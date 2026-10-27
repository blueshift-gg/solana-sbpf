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
const VPXOR: u8 = 0xef;
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
    previous_was_vector: bool,
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
            previous_was_vector: false,
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
        let sbpf_version = executable.get_sbpf_version();
        let len = program.len() / ebpf::INSN_SIZE;
        let mut jump_targets = vec![false; len + 1];
        let mut vector_instructions = 0;
        let mut has_callx = false;
        let mut pc = 0;
        while pc < len {
            let insn = ebpf::get_insn_unchecked(program, pc);
            let class = insn.opc & ebpf::BPF_CLS_MASK;
            match insn.opc {
                ebpf::VOP => vector_instructions += 1,
                ebpf::CALL_REG => has_callx = true,
                ebpf::CALL_IMM => {
                    if sbpf_version.static_syscalls() && insn.src == 1 {
                        let target = pc as i64 + insn.imm + 1;
                        if (0..len as i64).contains(&target) {
                            jump_targets[target as usize] = true;
                        }
                    } else if let Some((_, target)) =
                        executable.get_function_registry().lookup_by_key(insn.imm as u32)
                    {
                        if target < len {
                            jump_targets[target] = true;
                        }
                    }
                }
                ebpf::EXIT => {}
                _ if class == ebpf::BPF_JMP64
                    || (class == ebpf::BPF_JMP32 && sbpf_version.enable_jmp32()) =>
                {
                    let target = pc as i64 + insn.off as i64 + 1;
                    if (0..len as i64).contains(&target) {
                        jump_targets[target as usize] = true;
                    }
                }
                _ => {}
            }
            pc += if insn.opc == ebpf::LD_DW_IMM && !sbpf_version.disable_lddw() {
                2
            } else {
                1
            };
        }
        jump_targets[executable.get_entrypoint_instruction_offset().min(len)] = true;
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
        // A callx may enter any instruction, which rules out block local state
        state.static_flow = !has_callx;
        state.cached = state.static_flow && config.simd_jit_strategy == SimdJitStrategy::RegisterCached;
        state.jump_targets = jump_targets;
        Ok((state, vector_instructions))
    }
}

/// Opcodes which neither touch memory, branch, call, nor throw
fn is_pure_alu(opc: u8) -> bool {
    let class = opc & ebpf::BPF_CLS_MASK;
    let op = opc & ebpf::BPF_ALU_OP_MASK;
    if opc == ebpf::LD_DW_IMM {
        return true;
    }
    if class != ebpf::BPF_ALU32_LOAD && class != ebpf::BPF_ALU64_STORE {
        return false;
    }
    !matches!(
        opc,
        ebpf::LD_1B_REG
            | ebpf::LD_2B_REG
            | ebpf::LD_4B_REG
            | ebpf::LD_8B_REG
            | ebpf::ST_1B_IMM
            | ebpf::ST_2B_IMM
            | ebpf::ST_4B_IMM
            | ebpf::ST_8B_IMM
            | ebpf::ST_1B_REG
            | ebpf::ST_2B_REG
            | ebpf::ST_4B_REG
            | ebpf::ST_8B_REG
    ) && op != ebpf::BPF_DIV
        && op != ebpf::BPF_MOD
        && opc != ebpf::VOP
}

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
    }

    /// Called before every guest instruction is compiled (before its pc offset is recorded)
    pub(super) fn simd_before_instruction(&mut self, insn: &ebpf::Insn) {
        if !self.simd.enabled {
            return;
        }
        let is_target = self.simd.jump_targets.get(self.pc).copied().unwrap_or(true);
        if !self.simd.previous_was_vector {
            // The previous scalar instruction may have used R11
            self.simd.r11 = R11::Unknown;
        }
        if is_target {
            // Fall-through into a block start, jumps arrive with an empty cache
            self.simd_flush(true);
            self.simd.r11 = R11::Unknown;
        }
        self.simd.previous_was_vector = insn.opc == ebpf::VOP;
        if insn.opc != ebpf::VOP && !is_pure_alu(insn.opc) {
            self.simd_flush(true);
        }
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
            VADD32 | VXOR | VAND | VOR | VBLENDD => {
                let a = self.simd_mem(op, &operands[0], scratch[0], 0);
                self.simd_load(0, a);
                let b = self.simd_mem(op, &operands[1], scratch[1], 0);
                let ins = match op.subop {
                    VADD32 => V::alu(VPADDD, 0, 0, b),
                    VXOR => V::alu(VPXOR, 0, 0, b),
                    VAND => V::alu(VPAND, 0, 0, b),
                    VOR => V::alu(VPOR, 0, 0, b),
                    _ => V::pblendd(0, 0, b, op.off as u8),
                };
                self.emit_vex(ins);
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
            VADD32 | VXOR | VAND | VOR | VBLENDD => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va { VexRm::Reg(ra) } else { self.simd_source(op.vb) };
                let rd = self.simd_destination(op.vd);
                let ins = match op.subop {
                    VADD32 => V::alu(VPADDD, rd, ra, b),
                    VXOR => V::alu(VPXOR, rd, ra, b),
                    VAND => V::alu(VPAND, rd, ra, b),
                    VOR => V::alu(VPOR, rd, ra, b),
                    _ => V::pblendd(rd, ra, b, op.off as u8),
                };
                self.emit_vex(ins);
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
