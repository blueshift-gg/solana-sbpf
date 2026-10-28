//! Self contained runner for the SIMD experiment tests and benchmarks.

#![allow(clippy::arithmetic_side_effects, missing_docs)]

use crate::{
    aligned_memory::AlignedMemory,
    ebpf,
    elf::Executable,
    error::{EbpfError, ProgramResult},
    memory_region::{MemoryMapping, MemoryRegion},
    program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
    simd::VectorMemory,
    verifier::RequisiteVerifier,
    vm::{CallFrame, Config, ContextObject, EbpfVm, ExecutionMode, SimdJitStrategy},
};
use std::{ptr::NonNull, sync::Arc};

/// Instruction meter + memory mapping
pub struct HarnessContext {
    pub remaining: u64,
    pub mapping: MemoryMapping,
    /// The program `sol_test_invoke_` runs (a stand-in for a cross-program invocation)
    pub callee: Option<Callee>,
}

/// A program invoked through `sol_test_invoke_` in its own VM, like a CPI callee
pub struct Callee {
    pub runner: Box<Runner>,
    pub engine: Engine,
    /// Let the callee's VM allocate its own vector file (like a runtime which doesn't attach one)
    /// instead of attaching the runner's
    pub detached: bool,
    pub invocations: usize,
}

impl ContextObject for HarnessContext {
    fn consume(&mut self, amount: u64) {
        self.remaining = self.remaining.saturating_sub(amount);
    }
    fn get_remaining(&self) -> u64 {
        self.remaining
    }
    fn active_mapping_ptr(&mut self) -> NonNull<MemoryMapping> {
        NonNull::from_mut(&mut self.mapping)
    }
}

crate::declare_builtin_function!(
    /// `sol_memset_(dst, c, n)` with Agave's cost: max(10, n / 250) CUs
    SyscallMemset,
    fn rust(
        context: &mut HarnessContext,
        dst: u64,
        c: u64,
        n: u64,
        _arg4: u64,
        _arg5: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        context.consume(mem_op_cost(n));
        if n > 0 {
            let buf = Result::from(context.mapping.map(crate::memory_region::AccessType::Store, dst, n))?;
            unsafe { std::ptr::write_bytes(buf.ptr_mut().cast::<u8>(), c as u8, n as usize) };
        }
        Ok(0)
    }
);

crate::declare_builtin_function!(
    /// `sol_memcpy_(dst, src, n)` with Agave's cost: max(10, n / 250) CUs
    SyscallMemcpy,
    fn rust(
        context: &mut HarnessContext,
        dst: u64,
        src: u64,
        n: u64,
        _arg4: u64,
        _arg5: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        context.consume(mem_op_cost(n));
        if n > 0 {
            let s = Result::from(context.mapping.map(crate::memory_region::AccessType::Load, src, n))?;
            let d = Result::from(context.mapping.map(crate::memory_region::AccessType::Store, dst, n))?;
            unsafe { std::ptr::copy(s.ptr_mut().cast::<u8>().cast_const(), d.ptr_mut().cast::<u8>(), n as usize) };
        }
        Ok(0)
    }
);

crate::declare_builtin_function!(
    /// `sol_test_invoke_()`: runs [HarnessContext::callee] in a new VM, charges its CUs, returns its
    /// r0 (0 without a callee)
    SyscallTestInvoke,
    fn rust(
        context: &mut HarnessContext,
        _arg1: u64,
        _arg2: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let Some(mut callee) = context.callee.take() else {
            return Ok(0);
        };
        callee.runner.budget = context.remaining;
        let (cus, result) = if callee.detached {
            callee.runner.run_detached(callee.engine)
        } else {
            callee.runner.run(callee.engine)
        };
        callee.invocations += 1;
        context.consume(cus);
        context.callee = Some(callee);
        match result {
            ProgramResult::Ok(value) => Ok(value),
            ProgramResult::Err(err) => Err(Box::new(err)),
        }
    }
);

fn mem_op_cost(n: u64) -> u64 {
    10.max(n / 250)
}

/// Execution engine
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Interpreter,
    Jit,
}

pub use crate::simd::running_under_rosetta;

/// Production-like configuration with the experiment enabled
pub fn config(strategy: SimdJitStrategy) -> Config {
    Config {
        // V3 for compiled (ELF) programs with static syscalls, V4 for the generated ones
        enabled_sbpf_versions: SBPFVersion::V3..=SBPFVersion::V4,
        aligned_memory_mapping: true,
        enable_simd_experiment: true,
        simd_jit_strategy: strategy,
        simd_assume_avx2: running_under_rosetta()
            || std::env::var("SBPF_SIMD_ASSUME_AVX2").is_ok_and(|v| v == "1"),
        // SBPF_SIMD_AVX512=0 restricts the JIT to AVX2 (to compare both on one host)
        simd_avx512: !std::env::var("SBPF_SIMD_AVX512").is_ok_and(|v| v == "0"),
        ..Config::default()
    }
}

/// Owns an executable and all of its memory
pub struct Runner {
    pub executable: Executable<HarnessContext>,
    pub input: AlignedMemory<{ ebpf::HOST_ALIGN }>,
    pub heap: AlignedMemory<{ ebpf::HOST_ALIGN }>,
    pub stack: AlignedMemory<{ ebpf::HOST_ALIGN }>,
    pub vector: VectorMemory,
    context: Box<HarnessContext>,
    call_frames: Vec<CallFrame>,
    /// Instruction meter budget of every run
    pub budget: u64,
}

impl Runner {
    pub fn new(
        program: &[u8],
        input: &[u8],
        heap_size: usize,
        config: Config,
    ) -> Result<Self, EbpfError> {
        let executable = Executable::from_text_bytes(
            program,
            Self::loader(config)?,
            SBPFVersion::V4,
            FunctionRegistry::default(),
        )?;
        Self::with_executable(executable, input, heap_size)
    }

    /// Loads a compiled SBPF ELF (e.g. built with `cargo-build-sbf --arch v3`)
    pub fn from_elf(
        elf: &[u8],
        input: &[u8],
        heap_size: usize,
        config: Config,
    ) -> Result<Self, EbpfError> {
        let executable =
            Executable::load(elf, Self::loader(config)?).map_err(EbpfError::ElfError)?;
        Self::with_executable(executable, input, heap_size)
    }

    /// Loader with the vector intrinsics registered as static syscalls
    fn loader(config: Config) -> Result<Arc<BuiltinProgram<HarnessContext>>, EbpfError> {
        let mut loader = BuiltinProgram::new_loader(config);
        super::intrinsics::register(&mut loader).map_err(EbpfError::ElfError)?;
        // The memory syscalls compiled programs may use
        loader
            .register_definition::<SyscallMemset>("sol_memset_")
            .map_err(EbpfError::ElfError)?;
        loader
            .register_definition::<SyscallMemcpy>("sol_memcpy_")
            .map_err(EbpfError::ElfError)?;
        loader
            .register_definition::<SyscallTestInvoke>("sol_test_invoke_")
            .map_err(EbpfError::ElfError)?;
        Ok(Arc::new(loader))
    }

    fn with_executable(
        executable: Executable<HarnessContext>,
        input: &[u8],
        heap_size: usize,
    ) -> Result<Self, EbpfError> {
        executable.verify::<RequisiteVerifier>()?;
        let config = executable.get_config();
        let mut runner = Self {
            input: AlignedMemory::from_slice(input),
            heap: AlignedMemory::zero_filled(heap_size),
            stack: AlignedMemory::zero_filled(config.stack_size()),
            vector: VectorMemory::new(),
            call_frames: vec![CallFrame::default(); config.max_call_depth],
            context: Box::new(HarnessContext {
                callee: None,
                remaining: 0,
                mapping: unsafe {
                    MemoryMapping::new(vec![], &Config::default(), SBPFVersion::Reserved)?
                },
            }),
            budget: 1 << 40,
            executable,
        };
        let regions = vec![
            runner.executable.get_ro_region(),
            MemoryRegion::new(&raw mut *runner.stack.as_slice_mut(), ebpf::MM_STACK_START),
            MemoryRegion::new(&raw mut *runner.heap.as_slice_mut(), ebpf::MM_HEAP_START),
            MemoryRegion::new(&raw mut *runner.input.as_slice_mut(), ebpf::MM_INPUT_START),
        ];
        runner.context.mapping = unsafe {
            MemoryMapping::new(
                regions,
                runner.executable.get_config(),
                runner.executable.get_sbpf_version(),
            )?
        };
        Ok(runner)
    }

    /// JIT compiles, returns the compilation time
    pub fn jit_compile(&self) -> Result<std::time::Duration, EbpfError> {
        #[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
        {
            let start = std::time::Instant::now();
            self.executable.jit_compile()?;
            Ok(start.elapsed())
        }
        #[cfg(not(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64")))]
        Err(EbpfError::JitNotCompiled)
    }

    /// Runs the program once, returns the consumed CUs and the result
    pub fn run(&mut self, engine: Engine) -> (u64, ProgramResult) {
        let vector_ptr = self.vector.host_ptr();
        unsafe { self.run_with_vector(engine, vector_ptr) }
    }

    /// Runs with the vector file at `vector_ptr` (e.g. a [GuardedVectorFile])
    ///
    /// # Safety
    ///
    /// `vector_ptr` must point to [ebpf::VECTOR_MEMORY_SIZE] writable bytes.
    pub unsafe fn run_with_vector(
        &mut self,
        engine: Engine,
        vector_ptr: *mut u8,
    ) -> (u64, ProgramResult) {
        self.context.remaining = self.budget;
        let stack_len = self.stack.len();
        let mut vm = EbpfVm::new(
            self.executable.get_loader().clone(),
            self.executable.get_sbpf_version(),
            &mut *self.context,
            stack_len,
        );
        unsafe { vm.set_vector_memory(vector_ptr) };
        vm.registers[1] = ebpf::MM_INPUT_START;
        let mut mode = match engine {
            Engine::Interpreter => ExecutionMode::Interpreted,
            Engine::Jit => ExecutionMode::Jit,
        };
        vm.execute_program(&self.executable, &mut mode, &mut self.call_frames)
    }

    /// Interpreted run which also returns the pc of every executed instruction (profiling)
    #[cfg(feature = "tracer")]
    pub fn run_traced(&mut self) -> ((u64, ProgramResult), Vec<u64>) {
        self.context.remaining = self.budget;
        let stack_len = self.stack.len();
        let mut vm = EbpfVm::new(
            self.executable.get_loader().clone(),
            self.executable.get_sbpf_version(),
            &mut *self.context,
            stack_len,
        );
        unsafe { vm.set_vector_memory(self.vector.host_ptr()) };
        vm.registers[1] = ebpf::MM_INPUT_START;
        let r = vm.execute_program(
            &self.executable,
            &mut ExecutionMode::Interpreted,
            &mut self.call_frames,
        );
        let pcs = vm.register_trace.iter().map(|t| t[11]).collect();
        (r, pcs)
    }

    /// Runs without attaching [Self::vector], the VM allocates its own vector file
    pub fn run_detached(&mut self, engine: Engine) -> (u64, ProgramResult) {
        self.context.remaining = self.budget;
        let stack_len = self.stack.len();
        let mut vm = EbpfVm::new(
            self.executable.get_loader().clone(),
            self.executable.get_sbpf_version(),
            &mut *self.context,
            stack_len,
        );
        vm.registers[1] = ebpf::MM_INPUT_START;
        let mut mode = match engine {
            Engine::Interpreter => ExecutionMode::Interpreted,
            Engine::Jit => ExecutionMode::Jit,
        };
        vm.execute_program(&self.executable, &mut mode, &mut self.call_frames)
    }

    pub fn heap(&self) -> &[u8] {
        self.heap.as_slice()
    }

    /// Sets the program `sol_test_invoke_` runs
    pub fn set_callee(&mut self, callee: Option<Callee>) {
        self.context.callee = callee;
    }

    /// The callee after the run (to inspect its memory)
    pub fn take_callee(&mut self) -> Option<Callee> {
        self.context.callee.take()
    }
}

/// A vector file between inaccessible guard pages, so any access outside of it faults. `offset`
/// places the file that many bytes after the start of its pages (no alignment); the file ends at the
/// trailing guard page if `offset` is 0 and the size is a multiple of the page size, otherwise the
/// bytes after it up to the guard page are filled with a canary which [Self::canary_intact] checks.
pub struct GuardedVectorFile {
    map: *mut u8,
    map_len: usize,
    data: *mut u8,
    data_len: usize,
    offset: usize,
}

impl GuardedVectorFile {
    const CANARY: u8 = 0xa5;

    /// `at_end`: the file ends exactly at the trailing guard page (`offset` must be 0), otherwise
    /// it starts right after the leading guard page plus `offset`
    pub fn new(at_end: bool, offset: usize) -> Self {
        assert!(!at_end || offset == 0);
        // At least 16 KiB: Rosetta 2 reports 4 KiB pages but protects the host's 16 KiB pages
        let page = (unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize).max(16 << 10);
        let size = ebpf::VECTOR_MEMORY_SIZE;
        let data_len = (size + offset).div_ceil(page) * page;
        let map_len = data_len + 2 * page;
        unsafe {
            let map = libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert!(map != libc::MAP_FAILED, "mmap failed");
            let map = map.cast::<u8>();
            let data = map.add(page);
            assert_eq!(
                libc::mprotect(data.cast(), data_len, libc::PROT_READ | libc::PROT_WRITE),
                0
            );
            std::ptr::write_bytes(data, Self::CANARY, data_len);
            let offset = if at_end { data_len - size } else { offset };
            Self {
                map,
                map_len,
                data,
                data_len,
                offset,
            }
        }
    }

    pub fn ptr(&mut self) -> *mut u8 {
        unsafe { self.data.add(self.offset) }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.add(self.offset), ebpf::VECTOR_MEMORY_SIZE) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.data.add(self.offset), ebpf::VECTOR_MEMORY_SIZE)
        }
    }

    /// Whether the bytes around the file (inside the accessible pages) are unchanged
    pub fn canary_intact(&self) -> bool {
        let all = unsafe { std::slice::from_raw_parts(self.data, self.data_len) };
        let end = self.offset + ebpf::VECTOR_MEMORY_SIZE;
        all[..self.offset]
            .iter()
            .chain(&all[end..])
            .all(|b| *b == Self::CANARY)
    }
}

impl Drop for GuardedVectorFile {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.map.cast(), self.map_len) };
    }
}
