//! AVX2 / AVX-512 lowering of the experimental vector instructions (child module of the JIT)
//!
//! Slots are cached in YMM registers (ymm0..ymm14). With AVX-512, slot pairs used by 512 bit
//! operations are cached in a second set (zmm16..zmm30); a slot is never cached in both: an
//! operation of the other width writes it back and drops it first. Without AVX-512, 512 bit
//! operations run as two 256 bit halves.

#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::{
    simd::{self, VecInsn},
    vm::SimdJitStrategy,
    x86::{EvexInstruction as E, VexInstruction as V, VexRm},
};

/// ZMM registers of the slot pair cache (zmm16..zmm30), zmm31 is scratch
const ZREGS: usize = 15;
const ZBASE: u8 = 16;
const SCRATCH_ZMM: u8 = 31;
/// Opmask register used by blends and vmovmsk8
const KMASK: u8 = 1;

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

#[derive(Clone, Copy, Default)]
struct PairState {
    /// ZMM register (number)
    reg: Option<u8>,
    dirty: bool,
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
    /// AVX-512 lowering (512 bit operations in ZMM registers, AVX-512VL single instructions)
    pub(super) avx512: bool,
    /// AVX512_VBMI2 (vpcompressw) in addition
    vbmi2: bool,
    /// AVX512DQ (vpmullq) and AVX512_IFMA (vpmadd52*) in addition
    dq: bool,
    ifma: bool,
    pairs: [PairState; ebpf::VECTOR_SLOTS / 2],
    zregs: [Option<u8>; ZREGS],
    zlast_use: [u64; ZREGS],
    zpinned: [bool; ZREGS],
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
            avx512: false,
            vbmi2: false,
            dq: false,
            ifma: false,
            pairs: [PairState::default(); ebpf::VECTOR_SLOTS / 2],
            zregs: [None; ZREGS],
            zlast_use: [0; ZREGS],
            zpinned: [false; ZREGS],
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
        state.avx512 = config.simd_force_avx512
            || (config.simd_avx512
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl"));
        let (avx512, force) = (state.avx512, config.simd_force_avx512);
        let feature = |detected: bool| avx512 && (force || detected);
        state.vbmi2 = feature(std::arch::is_x86_feature_detected!("avx512vbmi2"));
        state.dq = feature(std::arch::is_x86_feature_detected!("avx512dq"));
        state.ifma = feature(std::arch::is_x86_feature_detected!("avx512ifma"));
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

    /// Emits an EVEX encoded instruction
    fn emit_evex(&mut self, instruction: E) {
        let (bytes, len) = instruction.encode();
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
        for (anchor, value) in [(ANCHOR_SIMD_C32, 32u32), (ANCHOR_SIMD_ONES, u32::MAX), (ANCHOR_SIMD_C31, 31)] {
            self.set_anchor(anchor);
            for _ in 0..8 {
                self.emit::<u32>(value);
            }
        }
        self.set_anchor(ANCHOR_SIMD_C63Q);
        for _ in 0..4 {
            self.emit::<u64>(63);
        }
        if self.simd.avx512 {
            // Opmasks of the 512 bit blends: the imm8 repeated for every 8 lanes
            self.set_anchor(ANCHOR_SIMD_BLENDD_MASKS);
            for imm in 0..256u32 {
                self.emit::<u16>((imm | imm << 8) as u16);
            }
            self.set_anchor(ANCHOR_SIMD_BLENDW_MASKS);
            for imm in 0..256u32 {
                self.emit::<u32>(imm * 0x0101_0101);
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
        for i in 0..slots {
            self.simd_drop_pair((op.vd as usize + i) as u8 / 2);
        }
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
        if self.simd.pairs.iter().any(|pair| pair.dirty) {
            self.simd_ensure_base();
            for pair in 0..ebpf::VECTOR_SLOTS / 2 {
                let state = self.simd.pairs[pair];
                if state.dirty {
                    self.emit_evex(E::store(state.reg.unwrap(), REGISTER_SCRATCH, Self::slot_disp(2 * pair as u8)));
                    self.simd.stats.flush_stores += 1;
                    self.simd.pairs[pair].dirty = false;
                }
            }
        }
        if invalidate {
            self.simd.pairs = [PairState::default(); ebpf::VECTOR_SLOTS / 2];
            self.simd.zregs = [None; ZREGS];
        }
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

    /// `rd = ra.rotate_left(rb & 31)` per lane, using `ra >> (32 - c) = (ra >> 1) >> (c ^ 31)`. `rd`
    /// may alias `ra` or `rb`, `t` is a free temporary (uses ymm15).
    fn simd_emit_rolv(&mut self, rd: u8, ra: u8, rb: u8, t: u8) {
        let c31 = VexRm::Rip(self.anchors[ANCHOR_SIMD_C31]);
        self.emit_vex(V::load_rip(SCRATCH_YMM, self.anchors[ANCHOR_SIMD_C31]));
        self.emit_vex(V::alu(VPAND, SCRATCH_YMM, SCRATCH_YMM, VexRm::Reg(rb)));
        self.emit_vex(V::alu38(VPSLLVD, t, ra, VexRm::Reg(SCRATCH_YMM)));
        self.emit_vex(V::alu(VPXOR, SCRATCH_YMM, SCRATCH_YMM, c31));
        self.emit_vex(V::psrld(rd, ra, 1));
        self.emit_vex(V::alu38(VPSRLVD, rd, rd, VexRm::Reg(SCRATCH_YMM)));
        self.emit_vex(V::alu(VPOR, rd, rd, VexRm::Reg(t)));
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
        if self.simd.avx512 && self.simd_avx512_native(&op) && !matches!(op.subop, VLOAD | VSTORE | VTRANSPOSE8 | VTRANSPOSE64) {
            // A slot is cached at one width only
            let (slots, count) = Self::simd_operand_slots(&op);
            for &slot in &slots[..count] {
                if op.wide {
                    self.simd_drop_slot(slot);
                    self.simd_drop_slot(slot + 1);
                } else {
                    self.simd_drop_pair(slot / 2);
                }
            }
            if op.wide {
                self.emit_simd_zmm(&op);
            } else {
                self.emit_simd_cached(&op);
            }
            if !self.simd.cached {
                self.simd_flush(true);
            }
            if !self.simd.static_flow {
                self.simd.r11 = R11::Unknown;
            }
            return Ok(());
        }
        match op.subop {
            VLOAD if self.emit_simd_static_load(&op) => {}
            VLOAD | VSTORE => self.emit_simd_transfer(&op),
            VTRANSPOSE8 | VTRANSPOSE64 => self.emit_simd_fixed(&op),
            VPERMI2D | VPERMT2D => self.emit_simd_cross(&op),
            VCOMPRESSLT16 => self.emit_simd_compress_scalar(&op),
            // No AVX2 equivalent (or a missing AVX-512 extension): the reference implementation
            VSRA64 | VMULLO64 | VMADD52LO | VMADD52HI | VCMP | VBLENDM | VCOMPRESS | VEXPAND => self.emit_simd_fallback(&op),
            VPERMD if op.wide => self.emit_simd_cross(&op),
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

    /// Transposes: flush, load the matrix into fixed registers, transpose, leave the results resident
    /// and dirty
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
            (VTRANSPOSE64, false) => {
                for row in 0..4u8 {
                    load(self, row, op.vd + row);
                }
                self.simd_emit_transpose4q([0, 1, 2, 3], [4, 5, 6, 7]);
                for row in 0..4u8 {
                    self.simd_assign(op.vd + row, row);
                }
            }
            (VTRANSPOSE64, true) if self.simd.avx512 => {
                // r0..r7 in zmm16..23, t in zmm24..31, u in zmm16..23, columns in zmm24..31
                for row in 0..8u8 {
                    self.emit_evex(E::load(16 + row, REGISTER_SCRATCH, Self::slot_disp(op.vd + 2 * row)));
                }
                for i in 0..4u8 {
                    self.emit_evex(E::alu_q(2, 1, VPUNPCKLQDQ, 24 + 2 * i, 16 + 2 * i, VexRm::Reg(17 + 2 * i)));
                    self.emit_evex(E::alu_q(2, 1, VPUNPCKHQDQ, 25 + 2 * i, 16 + 2 * i, VexRm::Reg(17 + 2 * i)));
                }
                for (base, t) in [(16u8, 24u8), (20, 28)] {
                    self.emit_evex(E::shufi64x2(base, t, t + 2, 0x88));
                    self.emit_evex(E::shufi64x2(base + 1, t + 1, t + 3, 0x88));
                    self.emit_evex(E::shufi64x2(base + 2, t, t + 2, 0xdd));
                    self.emit_evex(E::shufi64x2(base + 3, t + 1, t + 3, 0xdd));
                }
                for k in 0..4u8 {
                    self.emit_evex(E::shufi64x2(24 + k, 16 + k, 20 + k, 0x88));
                    self.emit_evex(E::shufi64x2(28 + k, 16 + k, 20 + k, 0xdd));
                }
                // Column k is the new row k; zmm31 is the scratch register, so that one is stored
                for k in 0..8u8 {
                    let (reg, pair) = (24 + k, op.vd / 2 + k);
                    if reg == SCRATCH_ZMM {
                        self.emit_evex(E::store(reg, REGISTER_SCRATCH, Self::slot_disp(2 * pair)));
                    } else {
                        self.simd.pairs[pair as usize] = PairState { reg: Some(reg), dirty: true };
                        self.simd.zregs[(reg - ZBASE) as usize] = Some(pair);
                        self.simd_ztouch(reg);
                    }
                }
            }
            _ => {
                // 8 x 8 as four 4 x 4 blocks of 256 bit halves: [A B; C D] -> [A' C'; B' D']
                let row = |r: u8, hi: u8| op.vd + 2 * r + hi;
                for (src, dst) in [((0u8, 0u8), (0u8, 0u8)), ((4, 1), (4, 1))] {
                    // A and D in place
                    for r in 0..4u8 {
                        load(self, r, row(src.0 + r, src.1));
                    }
                    self.simd_emit_transpose4q([0, 1, 2, 3], [4, 5, 6, 7]);
                    for r in 0..4u8 {
                        self.emit_vex(V::store(r, REGISTER_SCRATCH, Self::slot_disp(row(dst.0 + r, dst.1))));
                    }
                }
                // B (rows 0..4, high halves) and C (rows 4..8, low halves) swap places
                for r in 0..4u8 {
                    load(self, r, row(r, 1));
                    load(self, 8 + r, row(4 + r, 0));
                }
                self.simd_emit_transpose4q([0, 1, 2, 3], [4, 5, 6, 7]);
                self.simd_emit_transpose4q([8, 9, 10, 11], [4, 5, 6, 7]);
                for r in 0..4u8 {
                    self.simd_assign(row(4 + r, 0), r);
                    self.simd_assign(row(r, 1), 8 + r);
                }
            }
        }
    }

    /// 4 x 4 transpose of the u64 lanes of the registers `m` in place, `t` are temporaries
    fn simd_emit_transpose4q(&mut self, m: [u8; 4], t: [u8; 4]) {
        let r = VexRm::Reg;
        self.emit_vex(V::alu(VPUNPCKLQDQ, t[0], m[0], r(m[1])));
        self.emit_vex(V::alu(VPUNPCKHQDQ, t[1], m[0], r(m[1])));
        self.emit_vex(V::alu(VPUNPCKLQDQ, t[2], m[2], r(m[3])));
        self.emit_vex(V::alu(VPUNPCKHQDQ, t[3], m[2], r(m[3])));
        self.emit_vex(V::perm2i128(m[0], t[0], r(t[2]), 0x20));
        self.emit_vex(V::perm2i128(m[1], t[1], r(t[3]), 0x20));
        self.emit_vex(V::perm2i128(m[2], t[0], r(t[2]), 0x31));
        self.emit_vex(V::perm2i128(m[3], t[1], r(t[3]), 0x31));
    }

    /// Makes `reg` (a temporary) the new, dirty register of `slot`
    fn simd_rebind(&mut self, slot: u8, reg: u8) {
        if let Some(old) = self.simd.slots[slot as usize].reg {
            self.simd.regs[old as usize] = None;
        }
        self.simd_assign(slot, reg);
    }

    /// Cross half permutations without AVX-512, in the register cache: both halves are computed into
    /// temporaries before any destination is written (the operands may alias)
    fn emit_simd_cross(&mut self, op: &VecInsn) {
        use crate::simd::*;
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        let ensure = |jit: &mut Self, slot: u8| {
            let reg = jit.simd_ensure_reg(slot);
            jit.simd_pin(reg);
            reg
        };
        let temp = |jit: &mut Self| {
            let reg = jit.simd_allocate();
            jit.simd_pin(reg);
            reg
        };
        let s = SCRATCH_YMM;
        match (op.subop, op.wide) {
            (VPERMD, _) => {
                // out_h = idx_h bit 3 ? a_hi[idx_h & 7] : a_lo[idx_h & 7]
                let (a0, a1) = (ensure(self, op.va), ensure(self, op.va + 1));
                let idx = [ensure(self, op.vb), ensure(self, op.vb + 1)];
                let out = [temp(self), temp(self)];
                let m = temp(self);
                for h in 0..2 {
                    self.emit_vex(V::permd(out[h], idx[h], VexRm::Reg(a0)));
                    self.emit_vex(V::permd(s, idx[h], VexRm::Reg(a1)));
                    self.emit_vex(V::pslld(m, idx[h], 28));
                    self.emit_vex(V::blendvps(out[h], out[h], s, m));
                }
                self.simd_rebind(op.vd, out[0]);
                self.simd_rebind(op.vd + 1, out[1]);
            }
            (VPERMI2D | VPERMT2D, false) => {
                // out = idx bit 3 ? b[idx & 7] : a[idx & 7]
                let (ta, ti) = if op.subop == VPERMI2D { (op.va, op.vd) } else { (op.vd, op.va) };
                let (a, b, idx) = (ensure(self, ta), ensure(self, op.vb), ensure(self, ti));
                let (out, m) = (temp(self), temp(self));
                self.emit_vex(V::permd(out, idx, VexRm::Reg(a)));
                self.emit_vex(V::permd(s, idx, VexRm::Reg(b)));
                self.emit_vex(V::pslld(m, idx, 28));
                self.emit_vex(V::blendvps(out, out, s, m));
                self.simd_rebind(op.vd, out);
            }
            _ => {
                // 512 bit: bit 3 of the index picks the half, bit 4 the table
                let (ta, ti) = if op.subop == VPERMI2D { (op.va, op.vd) } else { (op.vd, op.va) };
                let (a0, a1) = (ensure(self, ta), ensure(self, ta + 1));
                let (b0, b1) = (ensure(self, op.vb), ensure(self, op.vb + 1));
                let idx = [ensure(self, ti), ensure(self, ti + 1)];
                let out = [temp(self), temp(self)];
                let (m, t) = (temp(self), temp(self));
                for h in 0..2 {
                    self.emit_vex(V::pslld(m, idx[h], 28));
                    self.emit_vex(V::permd(out[h], idx[h], VexRm::Reg(a0)));
                    self.emit_vex(V::permd(s, idx[h], VexRm::Reg(a1)));
                    self.emit_vex(V::blendvps(out[h], out[h], s, m));
                    self.emit_vex(V::permd(t, idx[h], VexRm::Reg(b0)));
                    self.emit_vex(V::permd(s, idx[h], VexRm::Reg(b1)));
                    self.emit_vex(V::blendvps(t, t, s, m));
                    self.emit_vex(V::pslld(m, idx[h], 27));
                    self.emit_vex(V::blendvps(out[h], out[h], t, m));
                }
                self.simd_rebind(op.vd, out[0]);
                self.simd_rebind(op.vd + 1, out[1]);
            }
        }
        self.simd.pinned = [false; CACHE_REGS];
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
            VROLV32 if self.simd.avx512 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu38(1, 0x15, rd, ra, b));
            }
            VTERNLOG if self.simd.avx512 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::pternlogd(1, rd, ra, b, op.off as u8));
            }
            VPERMI2D => {
                // AVX-512 only (without it, the fixed register lowering is used)
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::permi2d(1, rd, ra, b));
            }
            VADD64 | VSUB64 | VMULU32 | VSLLV64 | VSRLV64 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_vex(match op.subop {
                    VADD64 => V::alu(0xd4, rd, ra, b),
                    VSUB64 => V::alu(0xfb, rd, ra, b),
                    VMULU32 => V::alu(0xf4, rd, ra, b),
                    VSLLV64 => V::alu38w(0x47, rd, ra, b),
                    _ => V::alu38w(0x45, rd, ra, b),
                });
            }
            VSLL64 | VSRL64 | VSRA64 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                let imm = op.off as u8;
                match op.subop {
                    VSLL64 => self.emit_vex(V::psllq(rd, ra, imm)),
                    VSRL64 => self.emit_vex(V::psrlq(rd, ra, imm)),
                    _ => self.emit_evex(E::shift_imm_q(1, 0x72, 4, rd, ra, imm)),
                }
            }
            VMULLO64 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu_q(1, 2, 0x40, rd, ra, b));
            }
            VMADD52LO | VMADD52HI => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu_q(1, 2, if op.subop == VMADD52LO { 0xb4 } else { 0xb5 }, rd, ra, b));
            }
            VCMP => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                self.emit_evex(E::pcmp(1, (op.off >> 4) as u8, op.off & 8 != 0, KMASK, ra, b, (op.off & 7) as u8));
                self.emit_vex(V::kmovq_to_gpr(REGISTER_MAP[op.dst as usize], KMASK));
            }
            VBLENDM => {
                self.emit_vex(V::kmov_from_gpr(KMASK, REGISTER_MAP[op.src as usize], true));
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let a = if op.va == op.vb { VexRm::Reg(rb) } else { self.simd_source(op.va) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::pblendm(1, op.off as u8, rd, rb, a, KMASK));
            }
            VCOMPRESS | VEXPAND => {
                self.emit_vex(V::kmov_from_gpr(KMASK, REGISTER_MAP[op.src as usize], true));
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                if op.subop == VCOMPRESS {
                    self.emit_evex(E::pcompress(1, op.off as u8, rd, ra, KMASK));
                    self.emit_simd_lane_count(op, 32 >> op.off);
                } else {
                    self.emit_evex(E::pexpand(1, op.off as u8, rd, ra, KMASK));
                }
            }
            VSHUFB => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pshufb(rd, ra, b));
            }
            VPERMT2D => {
                // AVX-512 only (without it, emit_simd_cross)
                let ri = self.simd_ensure_reg(op.va);
                self.simd_pin(ri);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let b = if op.vb == op.va { VexRm::Reg(ri) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::permt2d(1, rd, ri, b));
            }
            VCOMPRESSLT16 => {
                // AVX-512 with VBMI2 only (otherwise emit_simd_compress_scalar)
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::pcmpuw(1, KMASK, ra, b, 1));
                self.emit_evex(E::pcompressw(1, rd, ra, KMASK));
                self.emit_simd_mask_count(REGISTER_MAP[op.dst as usize]);
            }
            VROLV64 if self.simd.avx512 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu_q(1, 2, 0x15, rd, ra, b));
            }
            VROLV64 => {
                // rd = ra << c | (ra >> 1) >> (c ^ 63) for c = rb & 63
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let t = self.simd_allocate();
                self.simd_pin(t);
                let rd = self.simd_destination(op.vd);
                let c63 = VexRm::Rip(self.anchors[ANCHOR_SIMD_C63Q]);
                self.emit_vex(V::load_rip(SCRATCH_YMM, self.anchors[ANCHOR_SIMD_C63Q]));
                self.emit_vex(V::alu(VPAND, SCRATCH_YMM, SCRATCH_YMM, VexRm::Reg(rb)));
                self.emit_vex(V::alu38w(VPSLLVD, t, ra, VexRm::Reg(SCRATCH_YMM)));
                self.emit_vex(V::alu(VPXOR, SCRATCH_YMM, SCRATCH_YMM, c63));
                self.emit_vex(V::psrlq(rd, ra, 1));
                self.emit_vex(V::alu38w(VPSRLVD, rd, rd, VexRm::Reg(SCRATCH_YMM)));
                self.emit_vex(V::alu(VPOR, rd, rd, VexRm::Reg(t)));
            }
            VROR32 if self.simd.avx512 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::shift_imm(1, 0, rd, ra, op.off as u8));
            }
            VROLV32 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                // A temporary register, not bound to a slot
                let t = self.simd_allocate();
                self.simd_pin(t);
                let rd = self.simd_destination(op.vd);
                self.simd_emit_rolv(rd, ra, rb, t);
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

    /// Slot operands of a vector operation (the first slot of each pair for 512 bit operations),
    /// for everything but vload / vstore / vtranspose8
    fn simd_operand_slots(op: &VecInsn) -> ([u8; 3], usize) {
        use crate::simd::*;
        match op.subop {
            VBCAST32 | VBCAST64 | VINS64 => ([op.vd, 0, 0], 1),
            VEXT64 | VMOVMSK8 => ([op.va, 0, 0], 1),
            VCMP => ([op.va, op.vb, 0], 2),
            VCOMPRESS | VEXPAND | VSLL64 | VSRL64 | VSRA64 => ([op.vd, op.va, 0], 2),
            VMOV | VROR32 | VSHUFD | VSRL32 | VSLL32 => ([op.vd, op.va, 0], 2),
            _ => ([op.vd, op.va, op.vb], 3),
        }
    }

    /// Writes `slot` back from the YMM cache (if dirty) and drops it
    fn simd_drop_slot(&mut self, slot: u8) {
        let state = self.simd.slots[slot as usize];
        if let Some(reg) = state.reg {
            if state.dirty {
                self.simd_ensure_base();
                self.emit_vex(V::store(reg, REGISTER_SCRATCH, Self::slot_disp(slot)));
                self.simd.stats.spills += 1;
            }
            self.simd.slots[slot as usize] = SlotState::default();
            self.simd.regs[reg as usize] = None;
        }
    }

    /// Writes `pair` back from the ZMM cache (if dirty) and drops it
    fn simd_drop_pair(&mut self, pair: u8) {
        let state = self.simd.pairs[pair as usize];
        if let Some(reg) = state.reg {
            if state.dirty {
                self.simd_ensure_base();
                self.emit_evex(E::store(reg, REGISTER_SCRATCH, Self::slot_disp(2 * pair)));
                self.simd.stats.spills += 1;
            }
            self.simd.pairs[pair as usize] = PairState::default();
            self.simd.zregs[(reg - ZBASE) as usize] = None;
        }
    }

    fn simd_ztouch(&mut self, reg: u8) {
        self.simd.clock += 1;
        self.simd.zlast_use[(reg - ZBASE) as usize] = self.simd.clock;
    }

    fn simd_zpin(&mut self, reg: u8) {
        self.simd.zpinned[(reg - ZBASE) as usize] = true;
    }

    /// Picks a free or the least recently used unpinned ZMM register, writing back its pair if dirty
    fn simd_zallocate(&mut self) -> u8 {
        let mut best = None;
        for i in 0..ZREGS {
            if self.simd.zpinned[i] {
                continue;
            }
            if self.simd.zregs[i].is_none() {
                best = Some(i);
                break;
            }
            if best.is_none_or(|b: usize| self.simd.zlast_use[i] < self.simd.zlast_use[b]) {
                best = Some(i);
            }
        }
        let i = best.expect("all zmm registers pinned");
        let reg = ZBASE + i as u8;
        if let Some(victim) = self.simd.zregs[i] {
            if self.simd.pairs[victim as usize].dirty {
                self.simd_ensure_base();
                self.emit_evex(E::store(reg, REGISTER_SCRATCH, Self::slot_disp(2 * victim)));
                self.simd.stats.spills += 1;
            }
            self.simd.pairs[victim as usize] = PairState::default();
            self.simd.stats.evictions += 1;
            self.simd.zregs[i] = None;
        }
        self.simd_ztouch(reg);
        reg
    }

    /// ZMM register holding `pair`, loading it if necessary
    fn simd_zensure(&mut self, pair: u8) -> u8 {
        if let Some(reg) = self.simd.pairs[pair as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_ztouch(reg);
            return reg;
        }
        let reg = self.simd_zallocate();
        self.simd_ensure_base();
        self.emit_evex(E::load(reg, REGISTER_SCRATCH, Self::slot_disp(2 * pair)));
        self.simd.pairs[pair as usize] = PairState { reg: Some(reg), dirty: false };
        self.simd.zregs[(reg - ZBASE) as usize] = Some(pair);
        reg
    }

    /// Register or memory operand for a read-only source pair
    fn simd_zsource(&mut self, pair: u8) -> VexRm {
        if let Some(reg) = self.simd.pairs[pair as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_ztouch(reg);
            self.simd_zpin(reg);
            VexRm::Reg(reg)
        } else {
            self.simd_ensure_base();
            VexRm::Mem(REGISTER_SCRATCH, Self::slot_disp(2 * pair))
        }
    }

    /// ZMM register which will receive the new value of `pair`
    fn simd_zdestination(&mut self, pair: u8) -> u8 {
        let reg = match self.simd.pairs[pair as usize].reg {
            Some(reg) => reg,
            None => {
                let reg = self.simd_zallocate();
                self.simd.zregs[(reg - ZBASE) as usize] = Some(pair);
                reg
            }
        };
        self.simd.pairs[pair as usize] = PairState { reg: Some(reg), dirty: true };
        self.simd_ztouch(reg);
        reg
    }

    /// 512 bit operations in ZMM registers (AVX-512F / BW)
    fn emit_simd_zmm(&mut self, op: &VecInsn) {
        use crate::simd::*;
        self.simd_ensure_base();
        self.simd.zpinned = [false; ZREGS];
        let (pd, pa, pb) = (op.vd / 2, op.va / 2, op.vb / 2);
        match op.subop {
            VBCAST32 | VBCAST64 => {
                let zd = self.simd_zdestination(pd);
                let src = REGISTER_MAP[op.src as usize];
                self.emit_evex(if op.subop == VBCAST32 { E::pbroadcastd_gpr(zd, src) } else { E::pbroadcastq_gpr(zd, src) });
            }
            VMOVMSK8 => {
                let za = match self.simd.pairs[pa as usize].reg {
                    Some(reg) => {
                        self.simd.stats.cache_hits += 1;
                        reg
                    }
                    None => {
                        self.emit_evex(E::load(SCRATCH_ZMM, REGISTER_SCRATCH, Self::slot_disp(op.va)));
                        SCRATCH_ZMM
                    }
                };
                self.emit_evex(E::pmovb2m(KMASK, za));
                self.emit_vex(V::kmovq_to_gpr(REGISTER_MAP[op.dst as usize], KMASK));
            }
            VADD32 | VXOR | VAND | VOR | VSUB32 | VMULLO32 | VMINU32 | VSLLV32 | VSRLV32 | VANDN | VROLV32 | VROLV64
            | VADD16 | VSUB16 | VMULLO16 | VMULHI16 | VMINU16 | VMADD16 | VBLENDD | VBLENDW | VSHUFB => {
                if matches!(op.subop, VBLENDD | VBLENDW) {
                    let (anchor, size) = if op.subop == VBLENDD { (ANCHOR_SIMD_BLENDD_MASKS, 2) } else { (ANCHOR_SIMD_BLENDW_MASKS, 4) };
                    let entry = unsafe { self.anchors[anchor].add(size * op.off as usize) };
                    self.emit_vex(V::kmov_load(KMASK, op.subop == VBLENDW, VexRm::Rip(entry)));
                }
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let b = if pb == pa { VexRm::Reg(za) } else { self.simd_zsource(pb) };
                let zd = self.simd_zdestination(pd);
                let insn = match op.subop {
                    VADD32 => E::alu(2, 0xfe, zd, za, b),
                    VSUB32 => E::alu(2, 0xfa, zd, za, b),
                    VXOR => E::alu(2, 0xef, zd, za, b),
                    VAND => E::alu(2, 0xdb, zd, za, b),
                    VOR => E::alu(2, 0xeb, zd, za, b),
                    VANDN => E::alu(2, 0xdf, zd, za, b),
                    VADD16 => E::alu(2, 0xfd, zd, za, b),
                    VSUB16 => E::alu(2, 0xf9, zd, za, b),
                    VMULLO16 => E::alu(2, 0xd5, zd, za, b),
                    VMULHI16 => E::alu(2, 0xe5, zd, za, b),
                    VMADD16 => E::alu(2, 0xf5, zd, za, b),
                    VMULLO32 => E::alu38(2, 0x40, zd, za, b),
                    VMINU32 => E::alu38(2, 0x3b, zd, za, b),
                    VSLLV32 => E::alu38(2, 0x47, zd, za, b),
                    VSRLV32 => E::alu38(2, 0x45, zd, za, b),
                    VROLV32 => E::alu38(2, 0x15, zd, za, b),
                    VROLV64 => E::alu_q(2, 2, 0x15, zd, za, b),
                    VMINU16 => E::alu38(2, 0x3a, zd, za, b),
                    VSHUFB => E::pshufb(2, zd, za, b),
                    VBLENDD => E::pblendmd(zd, za, b, KMASK),
                    _ => E::pblendmw(zd, za, b, KMASK),
                };
                self.emit_evex(insn);
            }
            VTERNLOG | VPERMI2D => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zensure(pd);
                self.simd_zpin(zd);
                let b = if pb == pa { VexRm::Reg(za) } else { self.simd_zsource(pb) };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(if op.subop == VTERNLOG { E::pternlogd(2, zd, za, b, op.off as u8) } else { E::permi2d(2, zd, za, b) });
            }
            VADD64 | VSUB64 | VMULU32 | VSLLV64 | VSRLV64 | VMULLO64 => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let b = if pb == pa { VexRm::Reg(za) } else { self.simd_zsource(pb) };
                let zd = self.simd_zdestination(pd);
                let (map, opcode) = match op.subop {
                    VADD64 => (1, 0xd4),
                    VSUB64 => (1, 0xfb),
                    VMULU32 => (1, 0xf4),
                    VSLLV64 => (2, 0x47),
                    VSRLV64 => (2, 0x45),
                    _ => (2, 0x40),
                };
                self.emit_evex(E::alu_q(2, map, opcode, zd, za, b));
            }
            VSLL64 | VSRL64 | VSRA64 => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zdestination(pd);
                let (opcode, ext) = match op.subop {
                    VSLL64 => (0x73, 6),
                    VSRL64 => (0x73, 2),
                    _ => (0x72, 4),
                };
                self.emit_evex(E::shift_imm_q(2, opcode, ext, zd, za, op.off as u8));
            }
            VMADD52LO | VMADD52HI => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zensure(pd);
                self.simd_zpin(zd);
                let b = if pb == pa { VexRm::Reg(za) } else { self.simd_zsource(pb) };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::alu_q(2, 2, if op.subop == VMADD52LO { 0xb4 } else { 0xb5 }, zd, za, b));
            }
            VCMP => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let b = if pb == pa { VexRm::Reg(za) } else { self.simd_zsource(pb) };
                self.emit_evex(E::pcmp(2, (op.off >> 4) as u8, op.off & 8 != 0, KMASK, za, b, (op.off & 7) as u8));
                self.emit_vex(V::kmovq_to_gpr(REGISTER_MAP[op.dst as usize], KMASK));
            }
            VBLENDM => {
                self.emit_vex(V::kmov_from_gpr(KMASK, REGISTER_MAP[op.src as usize], true));
                let zb = self.simd_zensure(pb);
                self.simd_zpin(zb);
                let a = if pa == pb { VexRm::Reg(zb) } else { self.simd_zsource(pa) };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::pblendm(2, op.off as u8, zd, zb, a, KMASK));
            }
            VCOMPRESS | VEXPAND => {
                self.emit_vex(V::kmov_from_gpr(KMASK, REGISTER_MAP[op.src as usize], true));
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zdestination(pd);
                if op.subop == VCOMPRESS {
                    self.emit_evex(E::pcompress(2, op.off as u8, zd, za, KMASK));
                    self.emit_simd_lane_count(op, 64 >> op.off);
                } else {
                    self.emit_evex(E::pexpand(2, op.off as u8, zd, za, KMASK));
                }
            }
            VPERMT2D => {
                let zidx = self.simd_zensure(pa);
                self.simd_zpin(zidx);
                let zd = self.simd_zensure(pd);
                self.simd_zpin(zd);
                let b = if pb == pa { VexRm::Reg(zidx) } else { self.simd_zsource(pb) };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::permt2d(2, zd, zidx, b));
            }
            VCOMPRESSLT16 => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let b = if pb == pa { VexRm::Reg(za) } else { self.simd_zsource(pb) };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::pcmpuw(2, KMASK, za, b, 1));
                self.emit_evex(E::pcompressw(2, zd, za, KMASK));
                self.emit_simd_mask_count(REGISTER_MAP[op.dst as usize]);
            }
            VPERMD => {
                let zidx = self.simd_zensure(pb);
                self.simd_zpin(zidx);
                let data = if pa == pb { VexRm::Reg(zidx) } else { self.simd_zsource(pa) };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::permd(zd, zidx, data));
            }
            VSHUFD => {
                let a = self.simd_zsource(pa);
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::pshufd(zd, a, op.off as u8));
            }
            VROR32 | VSRL32 | VSLL32 => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zdestination(pd);
                let ext = match op.subop {
                    VROR32 => 0,
                    VSRL32 => 2,
                    _ => 6,
                };
                self.emit_evex(E::shift_imm(2, ext, zd, za, op.off as u8));
            }
            VMOV => {
                if pd != pa {
                    let a = self.simd_zsource(pa);
                    let zd = self.simd_zdestination(pd);
                    match a {
                        VexRm::Reg(za) => self.emit_evex(E::mov(zd, za)),
                        VexRm::Mem(base, disp) => self.emit_evex(E::load(zd, base, disp)),
                        VexRm::Rip(_) => unreachable!(),
                    }
                }
            }
            _ => unreachable!(),
        }
        self.simd.zpinned = [false; ZREGS];
    }

    /// `dst = popcnt(k1)` (the number of lanes a compress kept)
    fn emit_simd_mask_count(&mut self, dst: X86Register) {
        self.emit_vex(V::kmovd_to_gpr(dst, KMASK));
        // popcnt r32, r32 (no helper for it in the emitter)
        let d = dst as u8;
        self.emit::<u8>(0xf3);
        if d >= 8 {
            self.emit::<u8>(0x45);
        }
        self.emit::<u8>(0x0f);
        self.emit::<u8>(0xb8);
        self.emit::<u8>(0xc0 | (d & 7) << 3 | (d & 7));
        self.simd.stats.native_instructions += 2;
    }

    /// compresslt16 without AVX512_VBMI2: a branchless scalar loop over the u16 lanes in the vector
    /// file (both lanes read, a[i] stored at the count, `count += a[i] < b[i]`), then the remaining
    /// lanes are zeroed. The store goes to index count <= i, after lane i is read, so it never
    /// clobbers a lane not yet read and the operands may alias.
    fn emit_simd_compress_scalar(&mut self, op: &VecInsn) {
        let n = 16 * op.halves() as i32;
        // The operands must be in memory
        for slot in [op.vd, op.va, op.vb] {
            for s in slot..slot + op.halves() as u8 {
                self.simd_drop_slot(s);
                self.simd_drop_pair(s / 2);
            }
        }
        self.simd_ensure_base();
        let count = REGISTER_MAP[op.dst as usize];
        let mut temps = [REGISTER_MAP[1], REGISTER_MAP[2], REGISTER_MAP[3]].iter().copied().filter(|r| *r != count);
        let (t, z) = (temps.next().unwrap(), temps.next().unwrap());
        self.emit_ins(X86Instruction::push(t, None));
        self.emit_ins(X86Instruction::push(z, None));
        self.emit_ins(X86Instruction::load_immediate(count, 0));
        let (da, db, dd) = (Self::slot_disp(op.va), Self::slot_disp(op.vb), Self::slot_disp(op.vd));
        for i in 0..n {
            // Both lanes are read before the store (which may hit b[i] when count == i)
            self.emit_ins(X86Instruction::load(OperandSize::S16, REGISTER_SCRATCH, t, X86IndirectAccess::Offset(da + 2 * i)));
            self.emit_ins(X86Instruction::load(OperandSize::S16, REGISTER_SCRATCH, z, X86IndirectAccess::Offset(db + 2 * i)));
            self.emit_ins(X86Instruction::store(OperandSize::S16, t, REGISTER_SCRATCH, X86IndirectAccess::OffsetIndexShift(dd, count, 1)));
            // CF = a[i] < b[i]
            self.emit_ins(X86Instruction::cmp(OperandSize::S32, z, t, None));
            self.emit_ins(X86Instruction::alu_immediate(OperandSize::S64, 0x81, 2, count, 0, None)); // adc count, 0
        }
        // for (t = count; t < n; t++) d[t] = 0
        self.emit_ins(X86Instruction::load_immediate(z, 0));
        self.emit_ins(X86Instruction::mov(OperandSize::S64, count, t));
        let top = self.offset_in_text_section;
        self.emit_ins(X86Instruction::cmp_immediate(OperandSize::S64, t, n as i64, None));
        // jae rel32 (patched), emitted raw so that no diversification noop separates it from the patch
        self.emit::<u8>(0x0f);
        self.emit::<u8>(0x83);
        self.emit::<i32>(0);
        let patch = self.offset_in_text_section;
        self.emit_ins(X86Instruction::store(OperandSize::S16, z, REGISTER_SCRATCH, X86IndirectAccess::OffsetIndexShift(dd, t, 1)));
        self.emit_ins(X86Instruction::alu_immediate(OperandSize::S64, 0x81, 0, t, 1, None));
        self.emit::<u8>(0xe9);
        let back = top as i64 - (self.offset_in_text_section as i64 + 4);
        self.emit::<i32>(back as i32);
        let distance = (self.offset_in_text_section - patch) as i32;
        unsafe {
            let ptr = self.result.text_section_mut().as_mut_ptr().add(patch - 4);
            std::ptr::write_unaligned(ptr.cast::<i32>(), distance);
        }
        self.emit_ins(X86Instruction::pop(z));
        self.emit_ins(X86Instruction::pop(t));
    }

    /// Whether the AVX-512 lowering has an instruction for the operation on this host
    fn simd_avx512_native(&self, op: &VecInsn) -> bool {
        use crate::simd::*;
        match op.subop {
            VCOMPRESSLT16 => self.simd.vbmi2,
            VCOMPRESS | VEXPAND => op.off >= 2 || self.simd.vbmi2,
            VMULLO64 => self.simd.dq,
            VMADD52LO | VMADD52HI => self.simd.ifma,
            _ => true,
        }
    }

    /// `dst = popcnt(src & low lanes)` for a compress over `lanes` lanes
    fn emit_simd_lane_count(&mut self, op: &VecInsn, lanes: u32) {
        let (dst, src) = (REGISTER_MAP[op.dst as usize], REGISTER_MAP[op.src as usize]);
        if lanes == 64 {
            self.emit_ins(X86Instruction::mov(OperandSize::S64, src, dst));
        } else {
            // A 32 bit mov zero extends
            self.emit_ins(X86Instruction::mov(OperandSize::S32, src, dst));
            if lanes < 32 {
                self.emit_ins(X86Instruction::alu_immediate(OperandSize::S32, 0x81, 4, dst, (1i64 << lanes) - 1, None));
            }
        }
        // popcnt r64, r64 (no helper for it in the emitter)
        let d = dst as u8;
        self.emit::<u8>(0xf3);
        self.emit::<u8>(0x48 | if d >= 8 { 0x05 } else { 0 });
        self.emit::<u8>(0x0f);
        self.emit::<u8>(0xb8);
        self.emit::<u8>(0xc0 | (d & 7) << 3 | (d & 7));
        self.simd.stats.native_instructions += 1;
    }

    /// Operations without an instruction on this host: one call to the reference implementation
    /// (the scalar result, if any, comes back in the destination register)
    fn emit_simd_fallback(&mut self, op: &VecInsn) {
        // Host calls clobber all vector registers and read and write the vector file in memory
        self.simd_flush(true);
        self.simd.stats.intrinsic_slow_paths += 1;
        let encoded = simd::encode_for_host(op);
        let scalar = if simd::uses_src(op.subop) { Value::Register(REGISTER_MAP[op.src as usize]) } else { Value::Constant64(0, false) };
        let dst = simd::writes_dst(op.subop).then(|| REGISTER_MAP[op.dst as usize]);
        self.emit_rust_call(
            Value::Constant64(simd::jit_interpret as *const u8 as i64, false),
            &[
                // First, the other arguments may overwrite it
                Argument { index: 2, value: scalar },
                Argument { index: 1, value: Value::Constant64(encoded as i64, false) },
                Argument { index: 0, value: Value::RegisterIndirect(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase), false) },
            ],
            dst,
        );
        self.simd.r11 = R11::Unknown;
    }
}
