//! Tiny macro assembler used to generate the experimental SBPFv4 test and benchmark programs.

#![allow(clippy::arithmetic_side_effects, missing_docs)]

use super::*;
use crate::ebpf::Insn;

/// Where the vector file lives
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VecMode {
    /// Dedicated vector region at [ebpf::MM_VECTOR_START]
    Slot,
    /// Ordinary VM memory pointed to by the given register
    Memory(u8),
    /// Dedicated vector region, every operation is a static call (`mov64 r5, operands` +
    /// `call sol_<op>`), see [super::intrinsics]. Clobbers r0 and r5.
    Call,
}

/// Branch target
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(usize);

/// Program builder
pub struct Asm {
    insns: Vec<Insn>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, Label, bool)>,
    /// Vector file mode of the emitted vector instructions
    pub mode: VecMode,
}

macro_rules! alu {
    ($($name:ident = $opc:ident;)*) => {
        $(pub fn $name(&mut self, dst: u8, src: u8) { self.raw(ebpf::$opc, dst, src, 0, 0); })*
    };
}
macro_rules! alu_imm {
    ($($name:ident = $opc:ident;)*) => {
        $(pub fn $name(&mut self, dst: u8, imm: i64) { self.raw(ebpf::$opc, dst, 0, 0, imm); })*
    };
}
macro_rules! jcc_imm {
    ($($name:ident = $opc:ident;)*) => {
        $(pub fn $name(&mut self, dst: u8, imm: i64, label: Label) { self.jump(ebpf::$opc, dst, 0, imm, label); })*
    };
}
macro_rules! jcc_reg {
    ($($name:ident = $opc:ident;)*) => {
        $(pub fn $name(&mut self, dst: u8, src: u8, label: Label) { self.jump(ebpf::$opc, dst, src, 0, label); })*
    };
}

impl Asm {
    pub fn new(mode: VecMode) -> Self {
        Self {
            insns: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
            mode,
        }
    }

    pub fn pc(&self) -> usize {
        self.insns.len()
    }

    pub fn raw(&mut self, opc: u8, dst: u8, src: u8, off: i16, imm: i64) {
        self.insns.push(Insn {
            ptr: 0,
            opc,
            dst,
            src,
            off,
            imm,
        });
    }

    pub fn new_label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        debug_assert!(self.labels[label.0].is_none());
        self.labels[label.0] = Some(self.pc());
    }

    fn jump(&mut self, opc: u8, dst: u8, src: u8, imm: i64, label: Label) {
        self.fixups.push((self.pc(), label, false));
        self.raw(opc, dst, src, 0, imm);
    }

    alu! {
        mov64 = MOV64_REG; add64 = ADD64_REG; sub64 = SUB64_REG; and64 = AND64_REG;
        or64 = OR64_REG; mov32 = MOV32_REG; add32 = ADD32_REG; xor32 = XOR32_REG; or32 = OR32_REG;
    }
    alu_imm! {
        mov64_imm = MOV64_IMM; add64_imm = ADD64_IMM; sub64_imm = SUB64_IMM; and64_imm = AND64_IMM;
        mul64_imm = MUL64_IMM; lsh64_imm = LSH64_IMM; rsh64_imm = RSH64_IMM; mov32_imm = MOV32_IMM;
        lsh32_imm = LSH32_IMM; rsh32_imm = RSH32_IMM; or64_imm = OR64_IMM;
    }
    jcc_imm! {
        jeq_imm = JEQ64_IMM; jne_imm = JNE64_IMM; jgt_imm = JGT64_IMM; jge_imm = JGE64_IMM;
        jlt_imm = JLT64_IMM; jle_imm = JLE64_IMM;
    }
    jcc_reg! {
        jeq = JEQ64_REG; jne = JNE64_REG; jgt = JGT64_REG; jge = JGE64_REG; jlt = JLT64_REG;
        jle = JLE64_REG;
    }

    /// Loads a 64 bit constant
    pub fn lddw(&mut self, dst: u8, imm: u64) {
        self.raw(ebpf::LD_DW_IMM, dst, 0, 0, imm as u32 as i32 as i64);
        self.raw(0, 0, 0, 0, (imm >> 32) as u32 as i32 as i64);
    }

    pub fn ja(&mut self, label: Label) {
        self.jump(ebpf::JA, 0, 0, 0, label);
    }

    /// Internal function call (SBPFv3+ static calls)
    pub fn call(&mut self, label: Label) {
        self.fixups.push((self.pc(), label, true));
        self.raw(ebpf::CALL_IMM, 0, 1, 0, 0);
    }

    pub fn exit(&mut self) {
        self.raw(ebpf::EXIT, 0, 0, 0, 0);
    }

    pub fn ldxb(&mut self, dst: u8, src: u8, off: i16) {
        self.raw(ebpf::LD_B_REG, dst, src, off, 0);
    }
    pub fn ldxw(&mut self, dst: u8, src: u8, off: i16) {
        self.raw(ebpf::LD_W_REG, dst, src, off, 0);
    }
    pub fn ldxdw(&mut self, dst: u8, src: u8, off: i16) {
        self.raw(ebpf::LD_DW_REG, dst, src, off, 0);
    }
    pub fn stxb(&mut self, dst: u8, off: i16, src: u8) {
        self.raw(ebpf::ST_B_REG, dst, src, off, 0);
    }
    pub fn stxw(&mut self, dst: u8, off: i16, src: u8) {
        self.raw(ebpf::ST_W_REG, dst, src, off, 0);
    }
    pub fn stxdw(&mut self, dst: u8, off: i16, src: u8) {
        self.raw(ebpf::ST_DW_REG, dst, src, off, 0);
    }
    pub fn stdw(&mut self, dst: u8, off: i16, imm: i64) {
        self.raw(ebpf::ST_DW_IMM, dst, 0, off, imm);
    }

    fn vop(&mut self, subop: u8, vd: u8, va: u8, vb: u8, src: u8, off: i16) {
        if self.mode == VecMode::Call {
            let op = VecInsn { subop, vd, va, vb, memory_mode: false, base_reg: 0, src, off };
            let packed = super::intrinsics::pack(&op);
            assert_eq!(
                super::intrinsics::unpack(subop, packed).map(|op| op.encode()),
                Some(op.encode()),
                "operands not encodable as an intrinsic call (scalar register must be r1..=r4)"
            );
            self.mov64_imm(super::intrinsics::OPERAND_REG, packed as i64);
            self.raw(ebpf::CALL_IMM, 0, 0, 0, super::intrinsics::hash(subop) as i32 as i64);
            return;
        }
        let (memory_mode, base_reg) = match self.mode {
            VecMode::Slot => (false, 0),
            VecMode::Memory(reg) => (true, reg),
            VecMode::Call => unreachable!(),
        };
        let insn = VecInsn {
            subop,
            vd,
            va,
            vb,
            memory_mode,
            base_reg,
            src,
            off,
        }
        .encode();
        debug_assert!(VecInsn::decode(&insn).is_some());
        self.insns.push(insn);
    }

    pub fn vload(&mut self, vd: u8, src: u8, off: i16) {
        self.vop(VLOAD, vd, 0, 0, src, off);
    }
    pub fn vstore(&mut self, src: u8, off: i16, va: u8) {
        self.vop(VSTORE, 0, va, 0, src, off);
    }
    pub fn vadd(&mut self, vd: u8, va: u8, vb: u8) {
        self.vop(VADD32, vd, va, vb, 0, 0);
    }
    pub fn vxor(&mut self, vd: u8, va: u8, vb: u8) {
        self.vop(VXOR, vd, va, vb, 0, 0);
    }
    pub fn vand(&mut self, vd: u8, va: u8, vb: u8) {
        self.vop(VAND, vd, va, vb, 0, 0);
    }
    pub fn vor(&mut self, vd: u8, va: u8, vb: u8) {
        self.vop(VOR, vd, va, vb, 0, 0);
    }
    pub fn vror(&mut self, vd: u8, va: u8, amount: i16) {
        self.vop(VROR32, vd, va, 0, 0, amount);
    }
    pub fn vshufd(&mut self, vd: u8, va: u8, imm: i16) {
        self.vop(VSHUFD, vd, va, 0, 0, imm);
    }
    pub fn vtranspose(&mut self, vd: u8) {
        self.vop(VTRANSPOSE8, vd, 0, 0, 0, 0);
    }
    pub fn vbcast(&mut self, vd: u8, src: u8) {
        self.vop(VBCAST32, vd, 0, 0, src, 0);
    }
    pub fn vmov(&mut self, vd: u8, va: u8) {
        self.vop(VMOV, vd, va, 0, 0, 0);
    }
    pub fn vpermd(&mut self, vd: u8, va: u8, vidx: u8) {
        self.vop(VPERMD, vd, va, vidx, 0, 0);
    }
    pub fn vblendd(&mut self, vd: u8, va: u8, vb: u8, mask: i16) {
        self.vop(VBLENDD, vd, va, vb, 0, mask);
    }

    /// Resolves labels and returns the bytecode
    pub fn finish(mut self) -> Vec<u8> {
        for (pc, label, is_call) in std::mem::take(&mut self.fixups) {
            let target = self.labels[label.0].expect("unbound label") as i64;
            let rel = target - pc as i64 - 1;
            if is_call {
                self.insns[pc].imm = rel;
            } else {
                self.insns[pc].off = <i16 as std::convert::TryFrom<i64>>::try_from(rel).expect("jump too far");
            }
        }
        self.insns.iter().flat_map(|insn| insn.to_array()).collect()
    }
}
