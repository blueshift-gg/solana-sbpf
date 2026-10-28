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

fn mem_op_cost(n: u64) -> u64 {
    10.max(n / 250)
}

/// Execution engine
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Interpreter,
    Jit,
}

/// Whether this x86_64 process is translated by Rosetta 2 (which executes AVX2 but hides it from
/// CPUID)
pub fn running_under_rosetta() -> bool {
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        let mut value: libc::c_int = 0;
        let mut size = std::mem::size_of::<libc::c_int>();
        let name = b"sysctl.proc_translated\0";
        let ret = unsafe {
            libc::sysctlbyname(
                name.as_ptr().cast(),
                (&raw mut value).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        ret == 0 && value == 1
    }
    #[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
    {
        false
    }
}

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
        self.context.remaining = self.budget;
        let stack_len = self.stack.len();
        let vector_ptr = self.vector.host_ptr();
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
}
