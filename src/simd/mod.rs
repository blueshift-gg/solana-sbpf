//! Experimental SIMD extension (research prototype, **not** a production API).
//!
//! All vector operations share the single experimental opcode [`ebpf::VOP`]. The JIT needs to know
//! the virtual vector operands at compile time in order to allocate host registers for them, so
//! the operands are encoded statically in the immediate instead of being passed in scalar
//! registers through the syscall ABI.
//!
//! ```text
//! opc = 0xe7 (VOP)
//! dst = vector file base register (memory mode only, must be 0 in slot mode)
//! src = scalar register (address for vload/vstore, value for vbcast32)
//! off = memory offset (vload/vstore) or 8 bit immediate (vror32x8, vshufd, vblendd)
//! imm = bits  0.. 5  vd   destination slot
//!       bits  6..11  va   first source slot
//!       bits 12..17  vb   second source slot
//!       bits 18..23  sub-operation
//!       bit  24      memory mode
//!       bits 25..31  reserved, must be zero
//! ```
//!
//! In *slot mode* the slot `vN` is the 32 byte vector at `MM_VECTOR_START + 32 * N` (the dedicated
//! vector region). In *memory mode* the slot `vN` is the 32 byte vector at `r[dst] + 32 * N` in
//! ordinary VM memory, which has to be translated and bounds checked on every access. The two modes
//! run the exact same programs and only differ in where the vector file lives, which allows to
//! measure whether the dedicated region is worth it.

#![allow(clippy::arithmetic_side_effects)]

pub mod asm;
pub mod blake3_program;
pub mod harness;
pub mod intrinsics;

use crate::{
    ebpf,
    error::{EbpfError, ProgramResult, StableResult},
    memory_region::{AccessType, MemoryMapping},
};

/// `v[vd] = load256(r[src] + off)`
pub const VLOAD: u8 = 0x00;
/// `store256(r[src] + off, v[va])`
pub const VSTORE: u8 = 0x01;
/// `v[vd] = v[va] + v[vb]` (eight wrapping u32 additions)
pub const VADD32: u8 = 0x02;
/// `v[vd] = v[va] ^ v[vb]`
pub const VXOR: u8 = 0x03;
/// `v[vd] = v[va] & v[vb]`
pub const VAND: u8 = 0x04;
/// `v[vd] = v[va] | v[vb]`
pub const VOR: u8 = 0x05;
/// `v[vd] = v[va].rotate_right(off)` (eight independent u32 rotations, `off` in 0..32)
pub const VROR32: u8 = 0x06;
/// `v[vd][h * 4 + i] = v[va][h * 4 + (off >> (2 * i)) & 3]` for both 128 bit halves `h` (pshufd)
pub const VSHUFD: u8 = 0x07;
/// Transposes the 8x8 u32 matrix formed by the eight slots `vd..vd + 8`
pub const VTRANSPOSE8: u8 = 0x08;
/// `v[vd] = [r[src] as u32; 8]`
pub const VBCAST32: u8 = 0x09;
/// `v[vd] = v[va]`
pub const VMOV: u8 = 0x0a;
/// `v[vd][i] = v[va][v[vb][i] & 7]` (cross lane permutation, vpermd)
pub const VPERMD: u8 = 0x0b;
/// `v[vd][i] = if (off >> i) & 1 != 0 { v[vb][i] } else { v[va][i] }` (vpblendd)
pub const VBLENDD: u8 = 0x0c;
/// Number of defined sub-operations
pub const SUBOP_COUNT: u8 = 0x0d;

/// Memory mode flag in the immediate
pub const MEMORY_MODE: u32 = 1 << 24;

/// The eight u32 lanes of a 256 bit vector, little-endian in memory
pub type Lanes = [u32; 8];

/// Decoded experimental vector instruction
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VecInsn {
    /// Sub-operation
    pub subop: u8,
    /// Destination slot
    pub vd: u8,
    /// First source slot
    pub va: u8,
    /// Second source slot
    pub vb: u8,
    /// Vector file lives in ordinary memory at `r[base_reg]`
    pub memory_mode: bool,
    /// Vector file base register (memory mode)
    pub base_reg: u8,
    /// Scalar register operand
    pub src: u8,
    /// Offset or immediate
    pub off: i16,
}

impl VecInsn {
    /// Decodes an instruction, returns `None` if it is not a well formed vector instruction
    pub fn decode(insn: &ebpf::Insn) -> Option<Self> {
        if insn.opc != ebpf::VOP {
            return None;
        }
        let imm = insn.imm as u32;
        if imm >> 25 != 0 {
            return None;
        }
        let op = Self {
            vd: (imm & 63) as u8,
            va: ((imm >> 6) & 63) as u8,
            vb: ((imm >> 12) & 63) as u8,
            subop: ((imm >> 18) & 63) as u8,
            memory_mode: imm & MEMORY_MODE != 0,
            base_reg: insn.dst,
            src: insn.src,
            off: insn.off,
        };
        if op.subop >= SUBOP_COUNT || insn.src > 10 || insn.dst > 10 {
            return None;
        }
        if !op.memory_mode && insn.dst != 0 {
            return None;
        }
        let uses_src = matches!(op.subop, VLOAD | VSTORE | VBCAST32);
        if !uses_src && insn.src != 0 {
            return None;
        }
        let off_ok = match op.subop {
            VLOAD | VSTORE => true,
            VROR32 => (0..32).contains(&op.off),
            VSHUFD | VBLENDD => (0..256).contains(&op.off),
            _ => op.off == 0,
        };
        if !off_ok {
            return None;
        }
        if op.subop == VTRANSPOSE8 && op.vd as usize > ebpf::VECTOR_SLOTS - 8 {
            return None;
        }
        Some(op)
    }

    /// Encodes the instruction
    pub fn encode(&self) -> ebpf::Insn {
        let imm = (self.vd as u32 & 63)
            | ((self.va as u32 & 63) << 6)
            | ((self.vb as u32 & 63) << 12)
            | ((self.subop as u32 & 63) << 18)
            | if self.memory_mode { MEMORY_MODE } else { 0 };
        ebpf::Insn {
            ptr: 0,
            opc: ebpf::VOP,
            dst: self.base_reg,
            src: self.src,
            off: self.off,
            imm: imm as i32 as i64,
        }
    }

    /// The vector operands this instruction touches in the order in which they are translated.
    ///
    /// The interpreter and the JIT translate in this exact order, so that the reported error of an
    /// instruction with multiple invalid operands is identical.
    pub fn operands(&self) -> ([Operand; 3], usize) {
        use Operand::*;
        let none = Slot(0, AccessType::Load);
        match self.subop {
            VLOAD => ([Scalar(AccessType::Load), Slot(self.vd, AccessType::Store), none], 2),
            VSTORE => ([Slot(self.va, AccessType::Load), Scalar(AccessType::Store), none], 2),
            VADD32 | VXOR | VAND | VOR | VPERMD | VBLENDD => (
                [
                    Slot(self.va, AccessType::Load),
                    Slot(self.vb, AccessType::Load),
                    Slot(self.vd, AccessType::Store),
                ],
                3,
            ),
            VROR32 | VSHUFD | VMOV => (
                [
                    Slot(self.va, AccessType::Load),
                    Slot(self.vd, AccessType::Store),
                    none,
                ],
                2,
            ),
            VTRANSPOSE8 => ([Transpose(self.vd), none, none], 1),
            VBCAST32 => ([Slot(self.vd, AccessType::Store), none, none], 1),
            _ => unreachable!(),
        }
    }
}

/// A memory operand of a vector instruction
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// Slot of the vector file (32 bytes)
    Slot(u8, AccessType),
    /// Eight consecutive slots (256 bytes), read and written
    Transpose(u8),
    /// `r[src] + off` (32 bytes)
    Scalar(AccessType),
}

impl Operand {
    /// Number of bytes accessed
    pub fn len(&self) -> u64 {
        match self {
            Operand::Transpose(_) => 8 * ebpf::VECTOR_WIDTH as u64,
            _ => ebpf::VECTOR_WIDTH as u64,
        }
    }

    /// Always false
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Access type of the translation
    pub fn access_type(&self) -> AccessType {
        match self {
            Operand::Slot(_, access_type) | Operand::Scalar(access_type) => *access_type,
            Operand::Transpose(_) => AccessType::Store,
        }
    }

    /// Whether the operand must be translated (and bounds checked) at runtime
    pub fn needs_translation(&self, memory_mode: bool) -> bool {
        memory_mode || matches!(self, Operand::Scalar(_))
    }

    /// VM address of the operand
    pub fn vm_addr(&self, op: &VecInsn, reg: &[u64]) -> u64 {
        match self {
            Operand::Slot(slot, _) | Operand::Transpose(slot) => {
                let base = if op.memory_mode {
                    reg[op.base_reg as usize]
                } else {
                    ebpf::MM_VECTOR_START
                };
                base.wrapping_add(*slot as u64 * ebpf::VECTOR_WIDTH as u64)
            }
            Operand::Scalar(_) => reg[op.src as usize].wrapping_add(op.off as i64 as u64),
        }
    }
}

/// Packs the translation request of up to three operands into one u64:
/// `len` in bits `16 * i .. 16 * i + 15` and the store flag in bit `16 * i + 15`.
pub fn translation_spec(requests: &[(u64, AccessType)]) -> u64 {
    let mut spec = 0u64;
    for (i, (len, access_type)) in requests.iter().enumerate() {
        debug_assert!(*len < 0x8000);
        let store = (*access_type == AccessType::Store) as u64;
        spec |= (len | (store << 15)) << (16 * i);
    }
    spec
}

/// Translates up to three vector operands, used by both the interpreter and the JIT.
///
/// `scratch[i]` holds the VM address on input and the host address on output. Translation happens
/// in two passes: The first pass may invoke the access violation handler which can replace regions
/// and thus invalidate host addresses translated before. The second pass only looks up the now
/// resident regions, so all returned host addresses are valid at the same time.
pub fn translate_operands(
    mapping: &mut MemoryMapping,
    scratch: &mut [u64; 4],
    spec: u64,
) -> Result<(), EbpfError> {
    for pass in 0..2 {
        for (i, slot) in scratch.iter_mut().enumerate().take(3) {
            let request = (spec >> (16 * i)) & 0xffff;
            if request == 0 {
                break;
            }
            let len = request & 0x7fff;
            let access_type = if request & 0x8000 != 0 {
                AccessType::Store
            } else {
                AccessType::Load
            };
            let vm_addr = if pass == 0 { *slot } else { scratch_vm_addr(slot) };
            let result = if pass == 0 {
                mapping.map_with_access_violation_handler(access_type, vm_addr, len)
            } else {
                mapping.map(access_type, vm_addr, len)
            };
            match result {
                StableResult::Ok(host) => {
                    if pass == 0 {
                        // Keep the VM address around for the second pass
                        *slot = vm_addr;
                    } else {
                        *slot = host.ptr().cast::<u8>() as usize as u64;
                    }
                }
                StableResult::Err(err) => return Err(err),
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn scratch_vm_addr(slot: &u64) -> u64 {
    *slot
}

/// Entry point of the JIT for [translate_operands]
pub(crate) extern "C" fn jit_translate_operands(
    result: *mut ProgramResult,
    mapping: *mut MemoryMapping,
    scratch: *mut [u64; 4],
    spec: u64,
) {
    unsafe {
        if let Err(err) = translate_operands(&mut *mapping, &mut *scratch, spec) {
            *result = ProgramResult::Err(err);
        }
    }
}

/// Reads eight little-endian u32 lanes
#[inline(always)]
pub fn read_lanes(ptr: *const u8) -> Lanes {
    let mut lanes = [0u32; 8];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = u32::from_le(unsafe { std::ptr::read_unaligned(ptr.add(4 * i).cast::<u32>()) });
    }
    lanes
}

/// Writes eight little-endian u32 lanes
#[inline(always)]
pub fn write_lanes(ptr: *mut u8, lanes: &Lanes) {
    for (i, lane) in lanes.iter().enumerate() {
        unsafe { std::ptr::write_unaligned(ptr.add(4 * i).cast::<u32>(), lane.to_le()) };
    }
}

/// Eight wrapping u32 additions
pub fn add_u32x8(dst: &mut Lanes, src: &Lanes) {
    for i in 0..8 {
        dst[i] = dst[i].wrapping_add(src[i]);
    }
}

/// Reference semantics of all lane-wise sub-operations
pub fn compute(op: &VecInsn, a: &Lanes, b: &Lanes) -> Lanes {
    let mut out = *a;
    match op.subop {
        VADD32 => add_u32x8(&mut out, b),
        VXOR => (0..8).for_each(|i| out[i] = a[i] ^ b[i]),
        VAND => (0..8).for_each(|i| out[i] = a[i] & b[i]),
        VOR => (0..8).for_each(|i| out[i] = a[i] | b[i]),
        VROR32 => (0..8).for_each(|i| out[i] = a[i].rotate_right(op.off as u32 & 31)),
        VSHUFD => (0..8).for_each(|i| {
            let half = i & !3;
            out[i] = a[half + ((op.off as usize >> (2 * (i & 3))) & 3)];
        }),
        VPERMD => (0..8).for_each(|i| out[i] = a[(b[i] & 7) as usize]),
        VBLENDD => (0..8).for_each(|i| {
            if (op.off >> i) & 1 != 0 {
                out[i] = b[i];
            }
        }),
        VMOV | VLOAD | VSTORE => {}
        _ => unreachable!(),
    }
    out
}

/// Transposes an 8x8 matrix of u32
pub fn transpose8(rows: &mut [Lanes; 8]) {
    for i in 0..8 {
        for j in (i + 1)..8 {
            let t = rows[i][j];
            rows[i][j] = rows[j][i];
            rows[j][i] = t;
        }
    }
}

/// Executes one vector instruction in the interpreter
pub fn interpret(
    op: &VecInsn,
    reg: &[u64],
    vector_base: *mut u8,
    mapping: &mut MemoryMapping,
) -> Result<(), EbpfError> {
    let (operands, count) = op.operands();
    let operands = &operands[..count];
    let mut scratch = [0u64; 4];
    let mut requests = [(0u64, AccessType::Load); 3];
    let mut translated = 0;
    for operand in operands {
        if operand.needs_translation(op.memory_mode) {
            scratch[translated] = operand.vm_addr(op, reg);
            requests[translated] = (operand.len(), operand.access_type());
            translated += 1;
        }
    }
    if translated > 0 {
        translate_operands(
            mapping,
            &mut scratch,
            translation_spec(&requests[..translated]),
        )?;
    }
    if !op.memory_mode && vector_base.is_null() {
        return Err(EbpfError::UnsupportedInstruction);
    }
    let mut host = [std::ptr::null_mut::<u8>(); 3];
    let mut next_translated = 0;
    for (i, operand) in operands.iter().enumerate() {
        host[i] = if operand.needs_translation(op.memory_mode) {
            next_translated += 1;
            scratch[next_translated - 1] as usize as *mut u8
        } else {
            match operand {
                Operand::Slot(slot, _) | Operand::Transpose(slot) => unsafe {
                    vector_base.add(*slot as usize * ebpf::VECTOR_WIDTH)
                },
                Operand::Scalar(_) => unreachable!(),
            }
        };
    }
    match op.subop {
        VTRANSPOSE8 => {
            let mut rows = [[0u32; 8]; 8];
            for (i, row) in rows.iter_mut().enumerate() {
                *row = read_lanes(unsafe { host[0].add(i * ebpf::VECTOR_WIDTH) });
            }
            transpose8(&mut rows);
            for (i, row) in rows.iter().enumerate() {
                write_lanes(unsafe { host[0].add(i * ebpf::VECTOR_WIDTH) }, row);
            }
        }
        VBCAST32 => write_lanes(host[0], &[reg[op.src as usize] as u32; 8]),
        VADD32 | VXOR | VAND | VOR | VPERMD | VBLENDD => {
            let a = read_lanes(host[0]);
            let b = read_lanes(host[1]);
            write_lanes(host[2], &compute(op, &a, &b));
        }
        _ => {
            // VLOAD, VSTORE, VROR32, VSHUFD, VMOV
            let a = read_lanes(host[0]);
            write_lanes(host[1], &compute(op, &a, &a));
        }
    }
    Ok(())
}

/// Textual representation of a vector instruction
pub fn disassemble(insn: &ebpf::Insn) -> String {
    let Some(op) = VecInsn::decode(insn) else {
        return format!("vop invalid imm={:#x}", insn.imm);
    };
    let slot = |n: u8| {
        if op.memory_mode {
            format!("[r{}+{}]", op.base_reg, n as usize * ebpf::VECTOR_WIDTH)
        } else {
            format!("v{n}")
        }
    };
    let addr = format!("[r{}{:+}]", op.src, op.off);
    let (vd, va, vb) = (slot(op.vd), slot(op.va), slot(op.vb));
    match op.subop {
        VLOAD => format!("vload256 {vd}, {addr}"),
        VSTORE => format!("vstore256 {addr}, {va}"),
        VADD32 => format!("vadd32x8 {vd}, {va}, {vb}"),
        VXOR => format!("vxor256 {vd}, {va}, {vb}"),
        VAND => format!("vand256 {vd}, {va}, {vb}"),
        VOR => format!("vor256 {vd}, {va}, {vb}"),
        VROR32 => format!("vror32x8 {vd}, {va}, {}", op.off),
        VSHUFD => format!("vshufd {vd}, {va}, {:#x}", op.off),
        VTRANSPOSE8 => format!("vtranspose8 {vd}..{}", slot(op.vd + 7)),
        VBCAST32 => format!("vbcast32 {vd}, r{}", op.src),
        VMOV => format!("vmov {vd}, {va}"),
        VPERMD => format!("vpermd {vd}, {va}, {vb}"),
        VBLENDD => format!("vblendd {vd}, {va}, {vb}, {:#x}", op.off),
        _ => unreachable!(),
    }
}

/// Host memory backing the experimental vector region.
///
/// The same buffer must be mapped at [ebpf::MM_VECTOR_START] (see [Self::region]) and attached to
/// the VM (see [crate::vm::EbpfVm::set_vector_memory]). It is zeroed by the VM at the beginning of
/// every program invocation.
pub struct VectorMemory {
    memory: crate::aligned_memory::AlignedMemory<{ ebpf::VECTOR_WIDTH }>,
}

impl Default for VectorMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorMemory {
    /// Allocates zeroed vector memory
    pub fn new() -> Self {
        Self {
            memory: crate::aligned_memory::AlignedMemory::zero_filled(ebpf::VECTOR_MEMORY_SIZE),
        }
    }

    /// The memory region to add to the memory mapping
    pub fn region(&mut self) -> crate::memory_region::MemoryRegion {
        crate::memory_region::MemoryRegion::new(
            &raw mut *self.memory.as_slice_mut(),
            ebpf::MM_VECTOR_START,
        )
    }

    /// Host address of the vector memory
    pub fn host_ptr(&mut self) -> *mut u8 {
        self.memory.as_slice_mut().as_mut_ptr()
    }

    /// Contents
    pub fn as_slice(&self) -> &[u8] {
        self.memory.as_slice()
    }

    /// Reads the lanes of a slot
    pub fn slot(&self, index: usize) -> Lanes {
        read_lanes(self.as_slice()[index * ebpf::VECTOR_WIDTH..].as_ptr())
    }
}
