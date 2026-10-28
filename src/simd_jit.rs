//! AVX2 lowering of the experimental vector instructions (child module of the JIT)

#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::{
    simd::{self, Operand, VecInsn},
    vm::SimdJitStrategy,
    x86::{VexInstruction as V, VexRm},
};

/// Host YMM registers available to the register cache (ymm15 is reserved as scratch)
const CACHE_REGS: usize = 15;
const SCRATCH_YMM: u8 = 15;

const VPADDD: u8 = 0xfe;
const VPSUBD: u8 = 0xfa;
const VPMULLD: u8 = 0x40;
const VPMINUD: u8 = 0x3b;
const VPSLLVD: u8 = 0x47;
const VPSRLVD: u8 = 0x45;
const VPANDN: u8 = 0xdf;
const VPXOR: u8 = 0xef;
const VPADDW: u8 = 0xfd;
const VPSUBW: u8 = 0xf9;
const VPMULLW: u8 = 0xd5;
const VPMULHW: u8 = 0xe5;
const VPMADDWD: u8 = 0xf5;
const VPMINUW: u8 = 0x3a;
const VPAND: u8 = 0xdb;
const VPOR: u8 = 0xeb;
const VPUNPCKLDQ: u8 = 0x62;
const VPUNPCKHDQ: u8 = 0x6a;
const VPUNPCKLQDQ: u8 = 0x6c;
const VPUNPCKHQDQ: u8 = 0x6d;

/// Static statistics of the generated vector code
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimdJitStats {
    /// Guest vector instructions compiled
    pub vector_instructions: usize,
    /// Native instructions emitted (whole program)
    pub native_instructions: usize,
    /// Native 256 bit loads (vmovdqu ymm, m256), excluding memory source operands
    pub vector_loads: usize,
    /// Native 256 bit stores (vmovdqu m256, ymm)
    pub vector_stores: usize,
    /// ALU instructions which read a slot directly from memory (memory source operand)
    pub memory_operands: usize,
    /// Dirty slots written back because their register was evicted
    pub spills: usize,
    /// Slots loaded again after having been evicted within the same basic block
    pub reloads: usize,
    /// Operand accesses served from a cached register
    pub cache_hits: usize,
    /// Registers evicted to make room
    pub evictions: usize,
    /// Dirty slots written back at block boundaries, calls, memory accesses, meter checkpoints
    pub flush_stores: usize,
    /// Runtime operand translations (calls into the memory mapping)
    pub translations: usize,
    /// Intrinsic calls whose operands weren't known at compile time (regular syscall)
    pub intrinsic_slow_paths: usize,
    /// vload256 from a compile time known read-only address (translated at compile time)
    pub static_loads: usize,
}

#[derive(Clone, Copy, Default)]
struct SlotState {
    reg: Option<u8>,
    dirty: bool,
    evicted: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum R11 {
    Unknown,
    VectorBase,
    Scratch(usize),
}

/// Compile time state of the vector lowering
pub struct SimdJitState {
    pub(super) enabled: bool,
    cached: bool,
    /// Basic block starts (jump and call targets)
    jump_targets: Vec<bool>,
    /// R11 contents can be tracked across instructions (no callx into the middle of blocks)
    static_flow: bool,
    slots: [SlotState; ebpf::VECTOR_SLOTS],
    regs: [Option<u8>; CACHE_REGS],
    last_use: [u64; CACHE_REGS],
    clock: u64,
    pinned: [bool; CACHE_REGS],
    r11: R11,
    /// The previous instruction left R11 intact
    previous_keeps_r11: bool,
    /// Static call identifiers of the vector intrinsics, indexed by sub-operation
    intrinsic_hashes: [u32; simd::SUBOP_COUNT as usize],
    /// Block local constant tracking of the guest registers (for the intrinsic operands)
    known: [Option<u64>; 11],
    /// The previously compiled instruction, its effect on `known` is applied lazily
    previous: Option<ebpf::Insn>,
    pub(super) stats: SimdJitStats,
}

impl SimdJitState {
    pub(super) fn disabled() -> Self {
        Self {
            enabled: false,
            cached: false,
            jump_targets: Vec::new(),
            static_flow: false,
            slots: [SlotState::default(); ebpf::VECTOR_SLOTS],
            regs: [None; CACHE_REGS],
            last_use: [0; CACHE_REGS],
            clock: 0,
            pinned: [false; CACHE_REGS],
            r11: R11::Unknown,
            previous_keeps_r11: false,
            intrinsic_hashes: [0; simd::SUBOP_COUNT as usize],
            known: [None; 11],
            previous: None,
            stats: SimdJitStats::default(),
        }
    }

    /// Scans the program, returns the state and the number of vector instructions
    pub(super) fn new<C: ContextObject>(
        executable: &Executable<C>,
        program: &[u8],
        config: &Config,
    ) -> Result<(Self, usize), EbpfError> {
        let mut state = Self::disabled();
        if !config.enable_simd_experiment {
            return Ok((state, 0));
        }
        let _ = program;
        let analysis = executable.simd_analysis();
        let vector_instructions = analysis.vector_instructions;
        let intrinsic_hashes = simd::intrinsics::hashes();
        if vector_instructions > 0
            && !config.simd_assume_avx2
            && !std::arch::is_x86_feature_detected!("avx2")
        {
            return Err(EbpfError::SimdHostUnsupported);
        }
        if vector_instructions == 0 {
            // Programs without vector instructions compile exactly like without the experiment
            return Ok((state, 0));
        }
        state.enabled = true;
        // callx may only enter instructions outside runs of vector code (see simd::SimdAnalysis),
        // so block local state is sound even in programs which contain callx
        state.static_flow = true;
        state.cached = config.simd_jit_strategy == SimdJitStrategy::RegisterCached;
        state.jump_targets = analysis.jump_targets.clone();
        state.intrinsic_hashes = intrinsic_hashes;
        Ok((state, vector_instructions))
    }
}

use simd::is_pure_alu;

impl<'a, C: ContextObject> JitCompiler<'a, C> {
    /// Emits a VEX encoded instruction
    fn emit_vex(&mut self, instruction: V) {
        let address = unsafe {
            self.result
                .text_section()
                .as_ptr()
                .add(self.offset_in_text_section)
        };
        let (bytes, len) = instruction.encode(address);
        for byte in &bytes[..len] {
            self.emit::<u8>(*byte);
        }
        self.simd.stats.native_instructions += 1;
        match instruction.opcode {
            0x6f if matches!(instruction.rm, VexRm::Mem(..)) => self.simd.stats.vector_loads += 1,
            0x7f => self.simd.stats.vector_stores += 1,
            _ if matches!(instruction.rm, VexRm::Mem(..)) => self.simd.stats.memory_operands += 1,
            _ => {}
        }
    }

    /// Emits vzeroupper (before calling host code, which may use legacy SSE)
    pub(super) fn emit_vzeroupper(&mut self) {
        if self.simd.enabled {
            self.emit::<u8>(0xc5);
            self.emit::<u8>(0xf8);
            self.emit::<u8>(0x77);
            self.simd.stats.native_instructions += 1;
        }
    }

    /// Byte shuffle masks for the 8 and 16 bit rotations
    pub(super) fn emit_simd_constants(&mut self) {
        if !self.simd.enabled {
            return;
        }
        for (anchor, rotation) in [(ANCHOR_SIMD_ROT16, 2u8), (ANCHOR_SIMD_ROT8, 1u8)] {
            self.set_anchor(anchor);
            for i in 0..32u8 {
                let dword = i & !3;
                self.emit::<u8>(dword | ((i + rotation) & 3));
            }
        }
        for (anchor, value) in [(ANCHOR_SIMD_C32, 32u32), (ANCHOR_SIMD_ONES, u32::MAX)] {
            self.set_anchor(anchor);
            for _ in 0..8 {
                self.emit::<u32>(value);
            }
        }
    }

    /// Called before every guest instruction is compiled (before its pc offset is recorded)
    pub(super) fn simd_before_instruction(&mut self, insn: &ebpf::Insn) {
        if !self.simd.enabled {
            return;
        }
        let is_target = self.simd.jump_targets.get(self.pc).copied().unwrap_or(true);
        if let Some(previous) = self.simd.previous.take() {
            self.simd_track_constants(&previous);
        }
        if !self.simd.previous_keeps_r11 {
            // The previous scalar instruction may have used R11
            self.simd.r11 = R11::Unknown;
        }
        if is_target {
            // Fall-through into a block start, jumps arrive with an empty cache
            self.simd_flush(true);
            self.simd.r11 = R11::Unknown;
            self.simd.known = [None; 11];
        }
        let vector = insn.opc == ebpf::VOP || self.simd_is_intrinsic(insn);
        // `mov64 dst, imm32` only writes `dst`, even when the immediate is sanitized
        self.simd.previous_keeps_r11 = vector || insn.opc == ebpf::MOV64_IMM;
        if !vector && !is_pure_alu(insn.opc) {
            self.simd_flush(true);
        }
        self.simd.previous = Some(insn.clone());
    }

    fn simd_is_intrinsic(&self, insn: &ebpf::Insn) -> bool {
        insn.opc == ebpf::CALL_IMM
            && insn.src == 0
            && self.simd.intrinsic_hashes.contains(&(insn.imm as u32))
    }

    /// Applies the effect of an instruction on the known register constants
    fn simd_track_constants(&mut self, insn: &ebpf::Insn) {
        let class = insn.opc & ebpf::BPF_CLS_MASK;
        let dst = insn.dst as usize;
        match insn.opc {
            ebpf::MOV64_IMM if dst < 11 => self.simd.known[dst] = Some(insn.imm as u64),
            ebpf::LD_DW_IMM if dst < 11 && !self.executable.get_sbpf_version().disable_lddw() => {
                let mut full = insn.clone();
                ebpf::augment_lddw_unchecked(self.program, &mut full);
                self.simd.known[dst] = Some(full.imm as u64);
            }
            ebpf::ADD64_IMM if dst < 11 => {
                self.simd.known[dst] = self.simd.known[dst].map(|v| v.wrapping_add(insn.imm as u64))
            }
            ebpf::MOV64_REG if dst < 11 => {
                self.simd.known[dst] = self.simd.known.get(insn.src as usize).copied().flatten()
            }
            ebpf::CALL_IMM if self.simd_is_intrinsic(insn) => self.simd.known[0] = Some(0),
            // Syscalls write r0, internal calls may change anything
            ebpf::CALL_IMM | ebpf::CALL_REG => self.simd.known = [None; 11],
            ebpf::EXIT | ebpf::VOP => {}
            _ if class == ebpf::BPF_JMP64 || class == ebpf::BPF_JMP32 => {}
            _ => {
                // Conservatively treat everything else as writing dst (stores only read it)
                if dst < 11 {
                    self.simd.known[dst] = None;
                }
            }
        }
    }

    /// Codegen of a vector intrinsic call (SIMD-0645 style): inline AVX2 if the packed operands
    /// in r5 are a known valid constant, otherwise a regular syscall to `slow_path`
    pub(crate) fn emit_simd_intrinsic(&mut self, subop: u8, slow_path: BuiltinFunction<C>) {
        let op = if self.simd.enabled {
            self.simd.known[simd::intrinsics::OPERAND_REG as usize]
                .and_then(|packed| simd::intrinsics::unpack(subop, packed))
        } else {
            None
        };
        if let Some(op) = op {
            if op.subop == simd::VLOAD && self.emit_simd_static_load(&op) {
                self.emit_ins(X86Instruction::load_immediate(REGISTER_MAP[0], 0));
                return;
            }
            if self.emit_simd_op(&op).is_ok() {
                // The intrinsic returns 0 in r0
                self.emit_ins(X86Instruction::load_immediate(REGISTER_MAP[0], 0));
                return;
            }
        }
        if self.simd.enabled {
            // Host calls clobber all YMM registers and access the vector region in memory
            self.simd_flush(true);
            self.simd.stats.intrinsic_slow_paths += 1;
        }
        self.emit_external_call(slow_path);
        if self.simd.enabled {
            self.simd.r11 = R11::Unknown;
        }
    }

    /// `vload256` from a compile time known address in the read-only section (e.g. a table in
    /// .rodata): the section is immutable and always mapped at its address, so the translation is
    /// done here instead of at runtime, and the load leaves the register cache intact. Returns false
    /// (nothing emitted) if the address isn't known or the 32 bytes aren't all in the section.
    fn emit_simd_static_load(&mut self, op: &VecInsn) -> bool {
        let Some(base) = self.simd.known[simd::intrinsics::SCALAR_REG as usize] else {
            return false;
        };
        let addr = base.wrapping_add(op.off as i64 as u64);
        let ro = self.executable.get_ro_region();
        let Some(host) = ro.vm_to_host_buffer(addr, ebpf::VECTOR_WIDTH as u64) else {
            return false;
        };
        let host = host.ptr() as *const u8 as i64;
        self.simd.stats.vector_instructions += 1;
        self.simd.stats.static_loads += 1;
        // Validates the vector base like every slot mode instruction
        self.simd_ensure_base();
        if self.simd.cached {
            self.simd.pinned = [false; CACHE_REGS];
            let rd = self.simd_destination(op.vd);
            self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, host));
            self.emit_vex(V::load(rd, REGISTER_SCRATCH, 0));
        } else {
            self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, host));
            self.emit_vex(V::load(0, REGISTER_SCRATCH, 0));
            self.simd.r11 = R11::Unknown;
            self.simd_ensure_base();
            self.emit_vex(V::store(0, REGISTER_SCRATCH, Self::slot_disp(op.vd)));
        }
        self.simd.r11 = R11::Unknown;
        true
    }

    /// Called before an instruction meter checkpoint, which may throw
    pub(super) fn simd_before_checkpoint(&mut self) {
        if self.simd.enabled {
            self.simd_flush(false);
            // The checkpoint loads the pc into R11
            self.simd.r11 = R11::Unknown;
        }
    }

    /// Called before the final bumper which throws ExecutionOverrun
    pub(super) fn simd_before_end(&mut self) {
        if self.simd.enabled {
            self.simd_flush(true);
        }
    }

    fn simd_ensure_base(&mut self) {
        if self.simd.r11 == R11::VectorBase {
            return;
        }
        self.emit_ins(X86Instruction::load(
            OperandSize::S64,
            REGISTER_PTR_TO_VM,
            REGISTER_SCRATCH,
            X86IndirectAccess::Offset(self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase)),
        ));
        // if vector_base == 0 { throw UnsupportedInstruction }
        self.emit_ins(X86Instruction::test(OperandSize::S64, REGISTER_SCRATCH, REGISTER_SCRATCH, None));
        // jnz rel32, emitted raw so that no diversification noop can sneak in before the patch
        self.emit::<u8>(0x0f);
        self.emit::<u8>(0x85);
        self.emit::<i32>(0);
        let patch = self.offset_in_text_section;
        self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, self.pc as i64));
        self.emit_ins(X86Instruction::jump_immediate(
            self.relative_to_anchor(ANCHOR_CALL_UNSUPPORTED_INSTRUCTION, 5),
        ));
        let distance = (self.offset_in_text_section - patch) as i32;
        unsafe {
            let ptr = self.result.text_section_mut().as_mut_ptr().add(patch - 4);
            std::ptr::write_unaligned(ptr.cast::<i32>(), distance);
        }
        self.emit_ins(X86Instruction::load(
            OperandSize::S64,
            REGISTER_PTR_TO_VM,
            REGISTER_SCRATCH,
            X86IndirectAccess::Offset(self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase)),
        ));
        // Reset at the end of the instruction if R11 can not be tracked across instructions
        self.simd.r11 = R11::VectorBase;
    }

    fn simd_ensure_scratch(&mut self, index: usize) {
        if self.simd.r11 == R11::Scratch(index) {
            return;
        }
        self.emit_ins(X86Instruction::load(
            OperandSize::S64,
            REGISTER_PTR_TO_VM,
            REGISTER_SCRATCH,
            X86IndirectAccess::Offset(
                self.slot_in_vm(RuntimeEnvironmentSlot::SimdScratch) + 8 * index as i32,
            ),
        ));
        self.simd.r11 = R11::Scratch(index);
    }

    fn slot_disp(slot: u8) -> i32 {
        slot as i32 * ebpf::VECTOR_WIDTH as i32
    }

    /// Writes back all dirty cached slots, optionally forgetting all cached registers
    fn simd_flush(&mut self, invalidate: bool) {
        let dirty = self.simd.slots.iter().any(|slot| slot.dirty);
        if dirty {
            self.simd_ensure_base();
            for slot in 0..ebpf::VECTOR_SLOTS {
                let state = self.simd.slots[slot];
                if state.dirty {
                    let reg = state.reg.unwrap();
                    self.emit_vex(V::store(reg, REGISTER_SCRATCH, Self::slot_disp(slot as u8)));
                    self.simd.stats.flush_stores += 1;
                    self.simd.slots[slot].dirty = false;
                }
            }
        }
        if invalidate {
            for slot in self.simd.slots.iter_mut() {
                *slot = SlotState::default();
            }
            self.simd.regs = [None; CACHE_REGS];
        }
    }

    fn simd_touch(&mut self, reg: u8) {
        self.simd.clock += 1;
        self.simd.last_use[reg as usize] = self.simd.clock;
    }

    /// Picks a free or the least recently used unpinned register, writing back its slot if dirty
    fn simd_allocate(&mut self) -> u8 {
        let mut best = None;
        for reg in 0..CACHE_REGS {
            if self.simd.pinned[reg] {
                continue;
            }
            if self.simd.regs[reg].is_none() {
                best = Some(reg);
                break;
            }
            if best.is_none_or(|b: usize| self.simd.last_use[reg] < self.simd.last_use[b]) {
                best = Some(reg);
            }
        }
        let reg = best.expect("all vector registers pinned");
        if let Some(victim) = self.simd.regs[reg] {
            let state = self.simd.slots[victim as usize];
            if state.dirty {
                self.simd_ensure_base();
                self.emit_vex(V::store(reg as u8, REGISTER_SCRATCH, Self::slot_disp(victim)));
                self.simd.stats.spills += 1;
            }
            self.simd.slots[victim as usize] = SlotState {
                reg: None,
                dirty: false,
                evicted: true,
            };
            self.simd.stats.evictions += 1;
            self.simd.regs[reg] = None;
        }
        self.simd_touch(reg as u8);
        reg as u8
    }

    fn simd_pin(&mut self, reg: u8) {
        self.simd.pinned[reg as usize] = true;
    }

    /// Register holding `slot`, loading it if necessary
    fn simd_ensure_reg(&mut self, slot: u8) -> u8 {
        if let Some(reg) = self.simd.slots[slot as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_touch(reg);
            return reg;
        }
        let reg = self.simd_allocate();
        self.simd_ensure_base();
        self.emit_vex(V::load(reg, REGISTER_SCRATCH, Self::slot_disp(slot)));
        if self.simd.slots[slot as usize].evicted {
            self.simd.stats.reloads += 1;
        }
        self.simd.slots[slot as usize] = SlotState {
            reg: Some(reg),
            dirty: false,
            evicted: false,
        };
        self.simd.regs[reg as usize] = Some(slot);
        reg
    }

    /// Register or memory operand for a read-only source
    fn simd_source(&mut self, slot: u8) -> VexRm {
        if let Some(reg) = self.simd.slots[slot as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_touch(reg);
            self.simd_pin(reg);
            VexRm::Reg(reg)
        } else {
            self.simd_ensure_base();
            VexRm::Mem(REGISTER_SCRATCH, Self::slot_disp(slot))
        }
    }

    /// Register which will receive the new value of `slot`
    fn simd_destination(&mut self, slot: u8) -> u8 {
        let reg = match self.simd.slots[slot as usize].reg {
            Some(reg) => reg,
            None => {
                let reg = self.simd_allocate();
                self.simd.regs[reg as usize] = Some(slot);
                reg
            }
        };
        self.simd.slots[slot as usize] = SlotState {
            reg: Some(reg),
            dirty: true,
            evicted: false,
        };
        self.simd_touch(reg);
        reg
    }

    /// Emits the rotation `dst = src.rotate_right(amount)` (uses ymm15 as temporary)
    fn simd_emit_rotate(&mut self, dst: u8, src: u8, amount: u8) {
        match amount {
            0 => {
                if dst != src {
                    self.emit_vex(V::mov(dst, src));
                }
            }
            16 | 8 => {
                let anchor = if amount == 16 { ANCHOR_SIMD_ROT16 } else { ANCHOR_SIMD_ROT8 };
                self.emit_vex(V::pshufb(dst, src, VexRm::Rip(self.anchors[anchor])));
            }
            _ => {
                self.emit_vex(V::psrld(SCRATCH_YMM, src, amount));
                self.emit_vex(V::pslld(dst, src, 32 - amount));
                self.emit_vex(V::alu(VPOR, dst, dst, VexRm::Reg(SCRATCH_YMM)));
            }
        }
    }

    /// Lane-wise binary instruction of a sub-operation
    fn simd_binary(subop: u8, rd: u8, ra: u8, b: VexRm, imm: u8) -> V {
        use crate::simd::*;
        match subop {
            VADD32 => V::alu(VPADDD, rd, ra, b),
            VSUB32 => V::alu(VPSUBD, rd, ra, b),
            VMULLO32 => V::alu38(VPMULLD, rd, ra, b),
            VMINU32 => V::alu38(VPMINUD, rd, ra, b),
            VSLLV32 => V::alu38(VPSLLVD, rd, ra, b),
            VSRLV32 => V::alu38(VPSRLVD, rd, ra, b),
            VANDN => V::alu(VPANDN, rd, ra, b),
            VXOR => V::alu(VPXOR, rd, ra, b),
            VAND => V::alu(VPAND, rd, ra, b),
            VOR => V::alu(VPOR, rd, ra, b),
            VADD16 => V::alu(VPADDW, rd, ra, b),
            VSUB16 => V::alu(VPSUBW, rd, ra, b),
            VMULLO16 => V::alu(VPMULLW, rd, ra, b),
            VMULHI16 => V::alu(VPMULHW, rd, ra, b),
            VMADD16 => V::alu(VPMADDWD, rd, ra, b),
            VMINU16 => V::alu38(VPMINUW, rd, ra, b),
            VBLENDW => V::pblendw(rd, ra, b, imm),
            VBLENDD => V::pblendd(rd, ra, b, imm),
            _ => unreachable!(),
        }
    }

    /// `rd = ra <<v b | ra >>v (32 - b)`, rd may alias ra or b (uses ymm15)
    fn simd_emit_rolv(&mut self, rd: u8, ra: u8, b: VexRm) {
        self.emit_vex(V::load_rip(SCRATCH_YMM, self.anchors[ANCHOR_SIMD_C32]));
        self.emit_vex(V::alu(VPSUBD, SCRATCH_YMM, SCRATCH_YMM, b));
        self.emit_vex(V::alu38(VPSRLVD, SCRATCH_YMM, ra, VexRm::Reg(SCRATCH_YMM)));
        self.emit_vex(V::alu38(VPSLLVD, rd, ra, b));
        self.emit_vex(V::alu(VPOR, rd, rd, VexRm::Reg(SCRATCH_YMM)));
    }

    /// ymm15 = g(x, y) for the two input truth table `tt` (indexed by `x << 1 | y`)
    fn simd_emit_tt2(&mut self, tt: u8, x: u8, y: u8) {
        let s = SCRATCH_YMM;
        let ones = VexRm::Rip(self.anchors[ANCHOR_SIMD_ONES]);
        let (op, invert) = match tt & 0xf {
            0x0 => {
                self.emit_vex(V::alu(VPXOR, s, s, VexRm::Reg(s)));
                return;
            }
            0xf => {
                self.emit_vex(V::pcmpeqd(s, s, VexRm::Reg(s)));
                return;
            }
            0xc | 0x3 => {
                self.emit_vex(V::mov(s, x));
                if tt == 0x3 {
                    self.emit_vex(V::alu(VPXOR, s, s, ones));
                }
                return;
            }
            0xa | 0x5 => {
                self.emit_vex(V::mov(s, y));
                if tt == 0x5 {
                    self.emit_vex(V::alu(VPXOR, s, s, ones));
                }
                return;
            }
            0x8 => ((VPAND, x, y), false),
            0x7 => ((VPAND, x, y), true),
            0xe => ((VPOR, x, y), false),
            0x1 => ((VPOR, x, y), true),
            0x6 => ((VPXOR, x, y), false),
            0x9 => ((VPXOR, x, y), true),
            0x2 => ((VPANDN, x, y), false),
            0xd => ((VPANDN, x, y), true),
            0x4 => ((VPANDN, y, x), false),
            _ => ((VPANDN, y, x), true), // 0xb
        };
        self.emit_vex(V::alu(op.0, s, op.1, VexRm::Reg(op.2)));
        if invert {
            self.emit_vex(V::alu(VPXOR, s, s, ones));
        }
    }

    /// `rd = f(rd, ra, rb)` for the truth table `imm` (vpternlogd semantics, uses ymm15). Aliased
    /// operands are folded into a two input function.
    fn simd_emit_ternlog(&mut self, rd: u8, ra: u8, rb: u8, imm: u8) {
        let f = |d: u8, a: u8, b: u8| (imm >> (d << 2 | a << 1 | b)) & 1;
        let tt2 = |g: &dyn Fn(u8, u8) -> u8| (0..4u8).fold(0, |t, i| t | g(i >> 1, i & 1) << i);
        if rd == ra || rd == rb {
            let (tt, other) = if rd == ra { (tt2(&|x, y| f(x, x, y)), rb) } else { (tt2(&|x, y| f(x, y, x)), ra) };
            self.simd_emit_tt2(tt, rd, other);
            self.emit_vex(V::mov(rd, SCRATCH_YMM));
            return;
        }
        // f = g_lo(a, b) ^ (d & (g_lo ^ g_hi)(a, b))
        let (lo, hi) = (imm & 0xf, imm >> 4);
        let h = lo ^ hi;
        if h == 0 {
            self.simd_emit_tt2(lo, ra, rb);
            self.emit_vex(V::mov(rd, SCRATCH_YMM));
            return;
        }
        if h != 0xf {
            self.simd_emit_tt2(h, ra, rb);
            self.emit_vex(V::alu(VPAND, rd, rd, VexRm::Reg(SCRATCH_YMM)));
        }
        if lo != 0 {
            self.simd_emit_tt2(lo, ra, rb);
            self.emit_vex(V::alu(VPXOR, rd, rd, VexRm::Reg(SCRATCH_YMM)));
        }
    }

    /// Emits the 8x8 transpose of ymm0..ymm7, results in [ymm8..ymm14, ymm0]
    fn simd_emit_transpose_regs(&mut self) -> [u8; 8] {
        let r = |n: u8| VexRm::Reg(n);
        for (i, (lo, hi)) in [(0u8, 1u8), (2, 3), (4, 5), (6, 7)].iter().enumerate() {
            let i = i as u8;
            self.emit_vex(V::alu(VPUNPCKLDQ, 8 + 2 * i, *lo, r(*hi)));
            self.emit_vex(V::alu(VPUNPCKHDQ, 9 + 2 * i, *lo, r(*hi)));
        }
        // abcd_04, abcd_15, abcd_26, abcd_37, efgh_04, efgh_15, efgh_26, efgh_37 in ymm0..ymm7
        self.emit_vex(V::alu(VPUNPCKLQDQ, 0, 8, r(10)));
        self.emit_vex(V::alu(VPUNPCKHQDQ, 1, 8, r(10)));
        self.emit_vex(V::alu(VPUNPCKLQDQ, 2, 9, r(11)));
        self.emit_vex(V::alu(VPUNPCKHQDQ, 3, 9, r(11)));
        self.emit_vex(V::alu(VPUNPCKLQDQ, 4, 12, r(14)));
        self.emit_vex(V::alu(VPUNPCKHQDQ, 5, 12, r(14)));
        self.emit_vex(V::alu(VPUNPCKLQDQ, 6, 13, r(15)));
        self.emit_vex(V::alu(VPUNPCKHQDQ, 7, 13, r(15)));
        self.emit_vex(V::perm2i128(8, 0, r(4), 0x20));
        self.emit_vex(V::perm2i128(12, 0, r(4), 0x31));
        self.emit_vex(V::perm2i128(9, 1, r(5), 0x20));
        self.emit_vex(V::perm2i128(13, 1, r(5), 0x31));
        self.emit_vex(V::perm2i128(10, 2, r(6), 0x20));
        self.emit_vex(V::perm2i128(14, 2, r(6), 0x31));
        self.emit_vex(V::perm2i128(11, 3, r(7), 0x20));
        self.emit_vex(V::perm2i128(0, 3, r(7), 0x31));
        [8, 9, 10, 11, 12, 13, 14, 0]
    }

    /// Translates the operands which need runtime translation, host addresses end up in scratch
    fn simd_emit_translation(&mut self, op: &VecInsn, operands: &[Operand]) -> [Option<usize>; 3] {
        let mut scratch_index = [None; 3];
        let mut requests = Vec::new();
        for (i, operand) in operands.iter().enumerate() {
            if !operand.needs_translation(op.memory_mode) {
                continue;
            }
            let (reg, offset) = match operand {
                Operand::Slot(slot, _) | Operand::Transpose(slot) => {
                    (REGISTER_MAP[op.base_reg as usize], Self::slot_disp(*slot) as i64)
                }
                Operand::Scalar(_) => (REGISTER_MAP[op.src as usize], op.off as i64),
            };
            self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, offset));
            self.emit_ins(X86Instruction::alu(OperandSize::S64, 0x01, reg, REGISTER_SCRATCH, None));
            self.emit_ins(X86Instruction::store(
                OperandSize::S64,
                REGISTER_SCRATCH,
                REGISTER_PTR_TO_VM,
                X86IndirectAccess::Offset(
                    self.slot_in_vm(RuntimeEnvironmentSlot::SimdScratch) + 8 * requests.len() as i32,
                ),
            ));
            scratch_index[i] = Some(requests.len());
            requests.push((operand.len(), operand.access_type()));
        }
        if requests.is_empty() {
            return scratch_index;
        }
        self.simd.stats.translations += 1;
        self.emit_rust_call(
            Value::Constant64(simd::jit_translate_operands as *const u8 as i64, false),
            &[
                Argument { index: 3, value: Value::Constant64(simd::translation_spec(&requests) as i64, false) },
                Argument { index: 2, value: Value::RegisterPlusConstant32(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::SimdScratch), false) },
                Argument { index: 1, value: Value::RegisterIndirect(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::MemoryMapping), false) },
                Argument { index: 0, value: Value::RegisterPlusConstant32(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::ProgramResult), false) },
            ],
            None,
        );
        self.emit_check_result_is_ok();
        self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, self.pc as i64));
        self.emit_ins(X86Instruction::conditional_jump_immediate(0x85, self.relative_to_anchor(ANCHOR_THROW_EXCEPTION, 6)));
        self.simd.r11 = R11::Unknown;
        scratch_index
    }

    /// Memory operand of operand `i` for the uncached lowering
    fn simd_mem(&mut self, op: &VecInsn, operand: &Operand, scratch: Option<usize>, extra: i32) -> VexRm {
        match scratch {
            Some(index) => {
                self.simd_ensure_scratch(index);
                VexRm::Mem(REGISTER_SCRATCH, extra)
            }
            None => {
                debug_assert!(!op.memory_mode);
                self.simd_ensure_base();
                let slot = match operand {
                    Operand::Slot(slot, _) | Operand::Transpose(slot) => *slot,
                    Operand::Scalar(_) => unreachable!(),
                };
                VexRm::Mem(REGISTER_SCRATCH, Self::slot_disp(slot) + extra)
            }
        }
    }

    fn simd_load(&mut self, dst: u8, mem: VexRm) {
        let VexRm::Mem(base, disp) = mem else { unreachable!() };
        self.emit_vex(V::load(dst, base, disp));
    }

    fn simd_store(&mut self, src: u8, mem: VexRm) {
        let VexRm::Mem(base, disp) = mem else { unreachable!() };
        self.emit_vex(V::store(src, base, disp));
    }

    /// Compiles one vector instruction
    pub(super) fn emit_simd_instruction(&mut self, insn: &ebpf::Insn) -> Result<(), EbpfError> {
        if self.offset_in_text_section + 4096 >= self.result.text_section().len() {
            return Err(EbpfError::ExhaustedTextSegment(self.pc));
        }
        let Some(op) = VecInsn::decode(insn) else {
            return Err(EbpfError::UnsupportedInstruction);
        };
        self.emit_simd_op(&op)
    }

    /// Compiles one decoded vector operation (opcode or intrinsic call form)
    fn emit_simd_op(&mut self, op: &VecInsn) -> Result<(), EbpfError> {
        if self.offset_in_text_section + 4096 >= self.result.text_section().len() {
            return Err(EbpfError::ExhaustedTextSegment(self.pc));
        }
        let op = *op;
        self.simd.stats.vector_instructions += 1;
        let (operands, count) = op.operands();
        let operands = &operands[..count];
        let needs_translation = operands.iter().any(|o| o.needs_translation(op.memory_mode));
        if self.simd.cached && !needs_translation && op.subop != simd::VLOAD && op.subop != simd::VSTORE {
            self.emit_simd_cached(&op);
        } else {
            if self.simd.cached {
                // Host calls clobber all YMM registers and the operands may alias the vector region
                self.simd_flush(true);
            }
            let scratch = self.simd_emit_translation(&op, operands);
            self.emit_simd_direct(&op, operands, scratch);
        }
        if !self.simd.static_flow {
            self.simd.r11 = R11::Unknown;
        }
        Ok(())
    }

    /// Strategy A: every operand comes from memory and the result goes back to memory
    fn emit_simd_direct(&mut self, op: &VecInsn, operands: &[Operand], scratch: [Option<usize>; 3]) {
        use crate::simd::*;
        match op.subop {
            VTRANSPOSE8 => {
                for row in 0..8u8 {
                    let mem = self.simd_mem(op, &operands[0], scratch[0], Self::slot_disp(row));
                    self.simd_load(row, mem);
                }
                let out = self.simd_emit_transpose_regs();
                for (row, reg) in out.iter().enumerate() {
                    let mem = self.simd_mem(op, &operands[0], scratch[0], Self::slot_disp(row as u8));
                    self.simd_store(*reg, mem);
                }
            }
            VBCAST32 => {
                self.emit_vex(V::movd_from_gpr(0, REGISTER_MAP[op.src as usize]));
                self.emit_vex(V::pbroadcastd(0, 0));
                let mem = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.simd_store(0, mem);
            }
            VADD32 | VXOR | VAND | VOR | VBLENDD | VSUB32 | VMULLO32 | VMINU32 | VSLLV32 | VSRLV32 | VANDN | VADD16 | VSUB16
            | VMULLO16 | VMULHI16 | VMINU16 | VMADD16 | VBLENDW => {
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.simd_load(0, a);
                let b = self.simd_mem(op, &operands[1], scratch[1], 0);
                self.emit_vex(Self::simd_binary(op.subop, 0, 0, b, op.off as u8));
                let d = self.simd_mem(op, &operands[2], scratch[2], 0);
                self.simd_store(0, d);
            }
            VROLV32 => {
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.simd_load(0, a);
                let b = self.simd_mem(op, &operands[1], scratch[1], 0);
                self.simd_load(1, b);
                self.simd_emit_rolv(0, 0, VexRm::Reg(1));
                let d = self.simd_mem(op, &operands[2], scratch[2], 0);
                self.simd_store(0, d);
            }
            VTERNLOG => {
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.simd_load(1, a);
                let b = self.simd_mem(op, &operands[1], scratch[1], 0);
                self.simd_load(2, b);
                let d = self.simd_mem(op, &operands[2], scratch[2], 0);
                self.simd_load(0, d);
                // Same aliasing as the slots, so that the folding matches
                let ra = if op.va == op.vd { 0 } else { 1 };
                let rb = if op.vb == op.vd { 0 } else if op.vb == op.va { ra } else { 2 };
                self.simd_emit_ternlog(0, ra, rb, op.off as u8);
                let d = self.simd_mem(op, &operands[2], scratch[2], 0);
                self.simd_store(0, d);
            }
            VPERMD => {
                let b = self.simd_mem(op, &operands[1], scratch[1], 0);
                self.simd_load(1, b);
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.emit_vex(V::permd(0, 1, a));
                let d = self.simd_mem(op, &operands[2], scratch[2], 0);
                self.simd_store(0, d);
            }
            VSHUFD => {
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.emit_vex(V::pshufd(0, a, op.off as u8));
                let d = self.simd_mem(op, &operands[1], scratch[1], 0);
                self.simd_store(0, d);
            }
            VSRL32 | VSLL32 => {
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.simd_load(0, a);
                self.emit_vex(if op.subop == VSRL32 { V::psrld(0, 0, op.off as u8) } else { V::pslld(0, 0, op.off as u8) });
                let d = self.simd_mem(op, &operands[1], scratch[1], 0);
                self.simd_store(0, d);
            }
            VROR32 => {
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.simd_load(0, a);
                self.simd_emit_rotate(0, 0, op.off as u8);
                let d = self.simd_mem(op, &operands[1], scratch[1], 0);
                self.simd_store(0, d);
            }
            _ => {
                // VLOAD, VSTORE, VMOV
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.simd_load(0, a);
                let d = self.simd_mem(op, &operands[1], scratch[1], 0);
                self.simd_store(0, d);
            }
        }
    }

    /// Strategy B: slots of the dedicated region are cached in YMM registers
    fn emit_simd_cached(&mut self, op: &VecInsn) {
        use crate::simd::*;
        // Every slot mode instruction validates the vector base, like the interpreter
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        match op.subop {
            VTRANSPOSE8 => {
                self.simd_flush(true);
                self.simd_ensure_base();
                for row in 0..8u8 {
                    self.emit_vex(V::load(row, REGISTER_SCRATCH, Self::slot_disp(op.vd + row)));
                }
                let out = self.simd_emit_transpose_regs();
                // Keep the results resident (dirty) instead of storing them right away
                for (row, reg) in out.iter().enumerate() {
                    let slot = op.vd + row as u8;
                    self.simd.slots[slot as usize] = SlotState { reg: Some(*reg), dirty: true, evicted: false };
                    self.simd.regs[*reg as usize] = Some(slot);
                    self.simd_touch(*reg);
                }
            }
            VBCAST32 => {
                self.emit_vex(V::movd_from_gpr(SCRATCH_YMM, REGISTER_MAP[op.src as usize]));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pbroadcastd(rd, SCRATCH_YMM));
            }
            VADD32 | VXOR | VAND | VOR | VBLENDD | VSUB32 | VMULLO32 | VMINU32 | VSLLV32 | VSRLV32 | VANDN | VADD16 | VSUB16
            | VMULLO16 | VMULHI16 | VMINU16 | VMADD16 | VBLENDW => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_vex(Self::simd_binary(op.subop, rd, ra, b, op.off as u8));
            }
            VROLV32 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let rd = self.simd_destination(op.vd);
                self.simd_emit_rolv(rd, ra, VexRm::Reg(rb));
            }
            VTERNLOG => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let rd = self.simd_destination(op.vd);
                self.simd_emit_ternlog(rd, ra, rb, op.off as u8);
            }
            VPERMD => {
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let a = if op.va == op.vb { VexRm::Reg(rb) } else { self.simd_source(op.va) };
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::permd(rd, rb, a));
            }
            VSHUFD => {
                let a = self.simd_source(op.va);
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pshufd(rd, a, op.off as u8));
            }
            VROR32 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                self.simd_emit_rotate(rd, ra, op.off as u8);
            }
            VSRL32 | VSLL32 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                self.emit_vex(if op.subop == VSRL32 { V::psrld(rd, ra, op.off as u8) } else { V::pslld(rd, ra, op.off as u8) });
            }
            VMOV => {
                if op.vd != op.va {
                    let a = self.simd_source(op.va);
                    let rd = self.simd_destination(op.vd);
                    match a {
                        VexRm::Reg(ra) => self.emit_vex(V::mov(rd, ra)),
                        VexRm::Mem(base, disp) => self.emit_vex(V::load(rd, base, disp)),
                        VexRm::Rip(_) => unreachable!(),
                    }
                }
            }
            _ => unreachable!(),
        }
        self.simd.pinned = [false; CACHE_REGS];
    }
}
