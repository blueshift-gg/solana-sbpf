//! AVX2 lowering of the experimental vector instructions (child module of the JIT)

#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::{
    simd::{self, VecInsn},
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
    /// Runtime vload / vstore transfers (calls into the memory mapping)
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
    /// `(static call identifier, sub-operation, wide)` of the vector intrinsics
    intrinsics: Vec<(u32, u8, bool)>,
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
            intrinsics: Vec::new(),
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
        state.intrinsics = simd::intrinsics::hashes();
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

    /// `(sub-operation, wide)` if the instruction is a vector intrinsic call
    fn simd_intrinsic(&self, insn: &ebpf::Insn) -> Option<(u8, bool)> {
        if insn.opc != ebpf::CALL_IMM || insn.src != 0 {
            return None;
        }
        self.simd.intrinsics.iter().find(|(hash, _, _)| *hash == insn.imm as u32).map(|(_, subop, wide)| (*subop, *wide))
    }

    fn simd_is_intrinsic(&self, insn: &ebpf::Insn) -> bool {
        self.simd_intrinsic(insn).is_some()
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
            ebpf::CALL_IMM if self.simd_is_intrinsic(insn) => {
                let (subop, _) = self.simd_intrinsic(insn).unwrap();
                self.simd.known[0] = if simd::writes_dst(subop) { None } else { Some(0) };
            }
            // Syscalls write r0, internal calls may change anything
            ebpf::CALL_IMM | ebpf::CALL_REG => self.simd.known = [None; 11],
            ebpf::VOP => {
                if let Some(op) = VecInsn::decode(insn) {
                    if simd::writes_dst(op.subop) {
                        self.simd.known[op.dst as usize] = None;
                    }
                }
            }
            ebpf::EXIT => {}
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
    /// in r1 are a known valid constant, otherwise a regular syscall to `slow_path`
    pub(crate) fn emit_simd_intrinsic(&mut self, subop: u8, wide: bool, slow_path: BuiltinFunction<C>) {
        let op = if self.simd.enabled {
            self.simd.known[simd::intrinsics::OPERAND_REG as usize]
                .and_then(|packed| simd::intrinsics::unpack(subop, wide, packed))
        } else {
            None
        };
        if let Some(op) = op {
            if self.emit_simd_op(&op).is_ok() {
                // The intrinsic returns its scalar result (written by the op) or 0 in r0
                if !simd::writes_dst(op.subop) {
                    self.emit_ins(X86Instruction::load_immediate(REGISTER_MAP[0], 0));
                }
                return;
            }
        }
        if self.simd.enabled {
            // Host calls clobber all YMM registers and read or write the vector file in memory
            self.simd_flush(true);
            self.simd.stats.intrinsic_slow_paths += 1;
        }
        self.emit_external_call(slow_path);
        if self.simd.enabled {
            self.simd.r11 = R11::Unknown;
        }
    }

    /// `vload` from a compile time known address in the read-only section (e.g. a table in
    /// .rodata): the section is immutable and always mapped at its address, so the translation is
    /// done here instead of at runtime, and the load leaves the register cache intact. Returns false
    /// (nothing emitted) if the address isn't known or the bytes aren't all in the section.
    fn emit_simd_static_load(&mut self, op: &VecInsn) -> bool {
        let Some(base) = self.simd.known[op.src as usize] else {
            return false;
        };
        let addr = base.wrapping_add(op.off as i64 as u64);
        let slots = op.transfer_slots();
        let ro = self.executable.get_ro_region();
        let Some(host) = ro.vm_to_host_buffer(addr, (slots * ebpf::VECTOR_WIDTH) as u64) else {
            return false;
        };
        let host = host.ptr() as *const u8 as i64;
        self.simd.stats.static_loads += 1;
        // Validates the vector base like every vector instruction
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        for i in 0..slots {
            // The destination is allocated first: an eviction stores relative to the vector base
            let rd = self.simd_destination(op.vd + i as u8);
            self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, host + (i * ebpf::VECTOR_WIDTH) as i64));
            self.simd.r11 = R11::Unknown;
            self.emit_vex(V::load(rd, REGISTER_SCRATCH, 0));
        }
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

    /// Compiles one vector instruction
    pub(super) fn emit_simd_instruction(&mut self, insn: &ebpf::Insn) -> Result<(), EbpfError> {
        let Some(op) = VecInsn::decode(insn) else {
            return Err(EbpfError::UnsupportedInstruction);
        };
        self.emit_simd_op(&op)
    }

    /// Compiles one decoded vector operation (opcode or intrinsic call form)
    fn emit_simd_op(&mut self, op: &VecInsn) -> Result<(), EbpfError> {
        use crate::simd::*;
        if self.offset_in_text_section + 4096 >= self.result.text_section().len() {
            return Err(EbpfError::ExhaustedTextSegment(self.pc));
        }
        let op = *op;
        self.simd.stats.vector_instructions += 1;
        match op.subop {
            VLOAD if self.emit_simd_static_load(&op) => {}
            VLOAD | VSTORE => self.emit_simd_transfer(&op),
            VTRANSPOSE8 | VPERMI2D => self.emit_simd_fixed(&op),
            VPERMD if op.wide => self.emit_simd_fixed(&op),
            VMOVMSK8 if op.wide => self.emit_simd_movmsk_wide(&op),
            _ if op.wide => {
                // Lane-wise on the two halves. All slots are even, so the low half's result never
                // overwrites a source of the high half.
                for h in 0..2u8 {
                    let half = VecInsn { wide: false, vd: op.vd + h, va: op.va + h, vb: op.vb + h, ..op };
                    self.emit_simd_cached(&half);
                }
            }
            _ => self.emit_simd_cached(&op),
        }
        if !self.simd.cached {
            // Strategy A: nothing stays in registers across vector instructions
            self.simd_flush(true);
        }
        if !self.simd.static_flow {
            self.simd.r11 = R11::Unknown;
        }
        Ok(())
    }

    /// vload / vstore: one call which translates the guest range and copies all slots
    fn emit_simd_transfer(&mut self, op: &VecInsn) {
        // Host calls clobber all YMM registers, and vstore reads the vector file in memory
        self.simd_flush(true);
        let store = op.subop == simd::VSTORE;
        let slot = if store { op.va } else { op.vd };
        let spec = store as u64 | (slot as u64) << 8 | (op.transfer_slots() as u64) << 16;
        self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, op.off as i64));
        self.emit_ins(X86Instruction::alu(OperandSize::S64, 0x01, REGISTER_MAP[op.src as usize], REGISTER_SCRATCH, None));
        self.simd.stats.translations += 1;
        self.emit_rust_call(
            Value::Constant64(simd::jit_transfer as *const u8 as i64, false),
            &[
                // First, the other arguments may overwrite it
                Argument { index: 3, value: Value::Register(REGISTER_SCRATCH) },
                Argument { index: 4, value: Value::Constant64(spec as i64, false) },
                Argument { index: 2, value: Value::RegisterIndirect(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase), false) },
                Argument { index: 1, value: Value::RegisterIndirect(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::MemoryMapping), false) },
                Argument { index: 0, value: Value::RegisterPlusConstant32(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::ProgramResult), false) },
            ],
            None,
        );
        self.emit_check_result_is_ok();
        self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, self.pc as i64));
        self.emit_ins(X86Instruction::conditional_jump_immediate(0x85, self.relative_to_anchor(ANCHOR_THROW_EXCEPTION, 6)));
        self.simd.r11 = R11::Unknown;
    }

    /// Makes `slot` resident in `reg` (dirty), after a flush emptied the cache
    fn simd_assign(&mut self, slot: u8, reg: u8) {
        self.simd.slots[slot as usize] = SlotState { reg: Some(reg), dirty: true, evicted: false };
        self.simd.regs[reg as usize] = Some(slot);
        self.simd_touch(reg);
    }

    /// Operations on fixed registers (cross half permutations, transpose): flush, load every source
    /// into ymm0.., compute, leave the results resident and dirty
    fn emit_simd_fixed(&mut self, op: &VecInsn) {
        use crate::simd::*;
        self.simd_flush(true);
        self.simd_ensure_base();
        let load = |jit: &mut Self, reg: u8, slot: u8| jit.emit_vex(V::load(reg, REGISTER_SCRATCH, Self::slot_disp(slot)));
        match (op.subop, op.wide) {
            (VTRANSPOSE8, _) => {
                for row in 0..8u8 {
                    load(self, row, op.vd + row);
                }
                let out = self.simd_emit_transpose_regs();
                for (row, reg) in out.iter().enumerate() {
                    self.simd_assign(op.vd + row as u8, *reg);
                }
            }
            (VPERMD, true) => {
                // out_h = idx_h bit 3 ? a_hi[idx_h & 7] : a_lo[idx_h & 7]
                for (reg, slot) in [(0, op.va), (1, op.va + 1), (2, op.vb), (3, op.vb + 1)] {
                    load(self, reg, slot);
                }
                for h in 0..2u8 {
                    let idx = 2 + h;
                    self.emit_vex(V::permd(4, idx, VexRm::Reg(0)));
                    self.emit_vex(V::permd(5, idx, VexRm::Reg(1)));
                    self.emit_vex(V::pslld(6, idx, 28));
                    self.emit_vex(V::blendvps(7 + h, 4, 5, 6));
                }
                self.simd_assign(op.vd, 7);
                self.simd_assign(op.vd + 1, 8);
            }
            (VPERMI2D, false) => {
                // out = idx bit 3 ? b[idx & 7] : a[idx & 7]
                for (reg, slot) in [(0, op.va), (1, op.vb), (2, op.vd)] {
                    load(self, reg, slot);
                }
                self.emit_vex(V::permd(4, 2, VexRm::Reg(0)));
                self.emit_vex(V::permd(5, 2, VexRm::Reg(1)));
                self.emit_vex(V::pslld(6, 2, 28));
                self.emit_vex(V::blendvps(7, 4, 5, 6));
                self.simd_assign(op.vd, 7);
            }
            _ => {
                // VPERMI2D, 512 bit: bit 3 of the index picks the half, bit 4 the table
                for (reg, slot) in [(0, op.va), (1, op.va + 1), (2, op.vb), (3, op.vb + 1), (4, op.vd), (5, op.vd + 1)] {
                    load(self, reg, slot);
                }
                for h in 0..2u8 {
                    let idx = 4 + h;
                    self.emit_vex(V::pslld(8, idx, 28));
                    self.emit_vex(V::permd(6, idx, VexRm::Reg(0)));
                    self.emit_vex(V::permd(7, idx, VexRm::Reg(1)));
                    self.emit_vex(V::blendvps(6, 6, 7, 8));
                    self.emit_vex(V::permd(7, idx, VexRm::Reg(2)));
                    self.emit_vex(V::permd(9, idx, VexRm::Reg(3)));
                    self.emit_vex(V::blendvps(7, 7, 9, 8));
                    self.emit_vex(V::pslld(8, idx, 27));
                    self.emit_vex(V::blendvps(10 + h, 6, 7, 8));
                }
                self.simd_assign(op.vd, 10);
                self.simd_assign(op.vd + 1, 11);
            }
        }
    }

    /// Register holding `slot` for a read, ymm15 loaded from memory if it isn't cached
    fn simd_read_reg(&mut self, slot: u8) -> u8 {
        if let Some(reg) = self.simd.slots[slot as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_touch(reg);
            return reg;
        }
        self.simd_ensure_base();
        self.emit_vex(V::load(SCRATCH_YMM, REGISTER_SCRATCH, Self::slot_disp(slot)));
        SCRATCH_YMM
    }

    /// vmovmsk8, 512 bit: the masks of both halves combined in one 64 bit register
    fn emit_simd_movmsk_wide(&mut self, op: &VecInsn) {
        let dst = REGISTER_MAP[op.dst as usize];
        let lo = self.simd_read_reg(op.va);
        self.emit_vex(V::pmovmskb(dst, lo));
        let hi = self.simd_read_reg(op.va + 1);
        self.emit_vex(V::pmovmskb(REGISTER_SCRATCH, hi));
        self.simd.r11 = R11::Unknown;
        self.emit_ins(X86Instruction::alu_immediate(OperandSize::S64, 0xc1, 4, REGISTER_SCRATCH, 32, None));
        self.emit_ins(X86Instruction::alu(OperandSize::S64, 0x09, REGISTER_SCRATCH, dst, None));
    }

    /// Strategy B: slots of the vector file are cached in YMM registers (strategy A flushes after
    /// every instruction)
    fn emit_simd_cached(&mut self, op: &VecInsn) {
        use crate::simd::*;
        // Every vector instruction validates the vector base, like the interpreter
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        match op.subop {
            VBCAST32 => {
                self.emit_vex(V::movd_from_gpr(SCRATCH_YMM, REGISTER_MAP[op.src as usize]));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pbroadcastd(rd, SCRATCH_YMM));
            }
            VBCAST64 => {
                self.emit_vex(V::movq_from_gpr(SCRATCH_YMM, REGISTER_MAP[op.src as usize]));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pbroadcastq(rd, SCRATCH_YMM));
            }
            VINS64 => {
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                self.emit_vex(V::movq_from_gpr(SCRATCH_YMM, REGISTER_MAP[op.src as usize]));
                self.emit_vex(V::pbroadcastq(SCRATCH_YMM, SCRATCH_YMM));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pblendd(rd, rd, VexRm::Reg(SCRATCH_YMM), 3 << (2 * op.off)));
            }
            VEXT64 => {
                let dst = REGISTER_MAP[op.dst as usize];
                if let Some(reg) = self.simd.slots[op.va as usize].reg {
                    self.simd.stats.cache_hits += 1;
                    self.simd_touch(reg);
                    let lane = op.off as u8;
                    let src = if lane >= 2 {
                        self.emit_vex(V::extracti128(SCRATCH_YMM, reg, 1));
                        SCRATCH_YMM
                    } else {
                        reg
                    };
                    self.emit_vex(if lane & 1 == 0 { V::movq_to_gpr(dst, src) } else { V::pextrq(dst, src, 1) });
                } else {
                    let disp = Self::slot_disp(op.va) + 8 * op.off as i32;
                    self.emit_ins(X86Instruction::load(OperandSize::S64, REGISTER_SCRATCH, dst, X86IndirectAccess::Offset(disp)));
                }
            }
            VMOVMSK8 => {
                let ra = self.simd_read_reg(op.va);
                self.emit_vex(V::pmovmskb(REGISTER_MAP[op.dst as usize], ra));
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
