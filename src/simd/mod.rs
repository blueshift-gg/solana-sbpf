//! Experimental SIMD extension (research prototype, **not** a production API).
//!
//! All vector operations share the single experimental opcode [`ebpf::VOP`]. The JIT needs to know
//! the virtual vector operands at compile time in order to allocate host registers for them, so
//! the operands are encoded statically in the immediate instead of being passed in scalar
//! registers through the syscall ABI.
//!
//! ```text
//! opc = 0xe7 (VOP)
//! dst = scalar destination register (vext64, vmovmsk8), otherwise 0
//! src = scalar source register (address of vload/vstore, value of vbcast / vins64)
//! off = memory offset (vload/vstore), 8 bit immediate (vror32x8, vshufd, vblendd, ...) or lane
//!       (vins64, vext64)
//! imm = bits  0.. 5  vd   destination slot
//!       bits  6..11  va   first source slot
//!       bits 12..17  vb   second source slot (slot count - 1 of vload/vstore)
//!       bits 18..23  sub-operation
//!       bit  24      512 bit operation on slot pairs
//!       bits 25..31  reserved, must be zero
//! ```
//!
//! The vector file (256 slots of 32 bytes) is private state of the VM, like the scalar registers.
//! It is not mapped into the guest address space: data enters and leaves it only through
//! `vload`/`vstore` (up to 32 consecutive slots from / to guest memory with one translation), the
//! 64 bit lane moves `vins64`/`vext64`, the broadcasts and `vmovmsk8`.
//!
//! 512 bit operations work on the slot pairs `(v2k, v2k + 1)` (low half in the even slot). All
//! of their slot operands must be even. They are defined lane by lane on the two halves, except
//! for the cross lane `vpermd` / `vpermi2d` (which index all 16 lanes) and `vmovmsk8` (64 bit mask).

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

/// `v[vd + i] = load256(r[src] + off + 32 i)` for the `vb + 1` slots `i` (at most 32)
pub const VLOAD: u8 = 0x00;
/// `store256(r[src] + off + 32 i, v[va + i])` for the `vb + 1` slots `i` (at most 32)
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
/// `v[vd] = v[va] - v[vb]` (eight wrapping u32 subtractions)
pub const VSUB32: u8 = 0x0d;
/// `v[vd] = v[va] * v[vb]` (low 32 bits of eight u32 products, vpmulld)
pub const VMULLO32: u8 = 0x0e;
/// `v[vd] = min(v[va], v[vb])` (eight unsigned u32 minimums, vpminud)
pub const VMINU32: u8 = 0x0f;
/// `v[vd] = v[va] >> off` (eight logical u32 shifts, `off` in 0..32)
pub const VSRL32: u8 = 0x10;
/// `v[vd] = v[va] << off` (eight u32 shifts, `off` in 0..32)
pub const VSLL32: u8 = 0x11;
/// `v[vd][i] = v[va][i] << v[vb][i]` (0 for counts above 31, vpsllvd)
pub const VSLLV32: u8 = 0x12;
/// `v[vd][i] = v[va][i] >> v[vb][i]` (0 for counts above 31, vpsrlvd)
pub const VSRLV32: u8 = 0x13;
/// `v[vd] = !v[va] & v[vb]` (vpandn)
pub const VANDN: u8 = 0x14;
/// `v[vd][i] = v[va][i].rotate_left(v[vb][i] % 32)` (vprolvd)
pub const VROLV32: u8 = 0x15;
/// `v[vd] = f(v[vd], v[va], v[vb])` bitwise, `f` given by the truth table `off` indexed by
/// `d << 2 | a << 1 | b` (vpternlogd)
pub const VTERNLOG: u8 = 0x16;
/// `v[vd] = v[va] + v[vb]` (sixteen wrapping u16 additions)
pub const VADD16: u8 = 0x17;
/// `v[vd] = v[va] - v[vb]` (sixteen wrapping u16 subtractions)
pub const VSUB16: u8 = 0x18;
/// `v[vd] = v[va] * v[vb]` (low 16 bits of sixteen products, vpmullw)
pub const VMULLO16: u8 = 0x19;
/// `v[vd] = (v[va] * v[vb]) >> 16` (high 16 bits of sixteen signed i16 products, vpmulhw)
pub const VMULHI16: u8 = 0x1a;
/// `v[vd] = min(v[va], v[vb])` (sixteen unsigned u16 minimums, vpminuw)
pub const VMINU16: u8 = 0x1b;
/// `v[vd][i] = a[2i] * b[2i] + a[2i + 1] * b[2i + 1]` (signed i16 products summed into eight
/// wrapping i32 lanes, vpmaddwd)
pub const VMADD16: u8 = 0x1c;
/// `v[vd][i] = if (off >> (i % 8)) & 1 != 0 { v[vb][i] } else { v[va][i] }` for the sixteen u16
/// lanes (vpblendw)
pub const VBLENDW: u8 = 0x1d;
/// `v[vd].u64[off] = r[src]` (64 bit lane `off` in 0..4)
pub const VINS64: u8 = 0x1e;
/// `r[dst] = v[va].u64[off]` (64 bit lane `off` in 0..4)
pub const VEXT64: u8 = 0x1f;
/// `v[vd] = [r[src]; 4]` (u64 lanes)
pub const VBCAST64: u8 = 0x20;
/// `r[dst] = sign bits of the 32 bytes of v[va]` (vpmovmskb, 64 bytes for the 512 bit form)
pub const VMOVMSK8: u8 = 0x21;
/// `v[vd][i] = (v[va] ++ v[vb])[v[vd][i] & (2 L - 1)]` for `L` u32 lanes per operand: two table
/// permutation, the index operand is overwritten (vpermi2d)
pub const VPERMI2D: u8 = 0x22;
/// `v[vd].u64[i] = v[va].u64[i].rotate_left(v[vb].u64[i] % 64)` (vprolvq)
pub const VROLV64: u8 = 0x23;
/// Transposes the matrix of u64 lanes whose rows are the slots `vd..vd + 4` (4 x 4), or for the 512
/// bit form the slot pairs `vd, vd + 2, .., vd + 14` (8 x 8)
pub const VTRANSPOSE64: u8 = 0x24;
/// `v[vd].u8[i] = v[vb].u8[i] & 0x80 != 0 ? 0 : v[va].u8[(i & !15) + (v[vb].u8[i] & 15)]` (byte
/// shuffle within each 128 bit lane, vpshufb)
pub const VSHUFB: u8 = 0x25;
/// `v[vd][i] = (v[vd] ++ v[vb])[v[va][i] & (2 L - 1)]` for `L` u32 lanes per operand: two table
/// permutation, the first table is overwritten (vpermt2d)
pub const VPERMT2D: u8 = 0x26;
/// The u16 lanes `i` of `v[va]` with `v[va][i] < v[vb][i]` (unsigned), in order, packed into the
/// low lanes of `v[vd]` (the others zero); `r[dst]` = their number (vpcmpuw + vpcompressw)
pub const VCOMPRESSLT16: u8 = 0x27;
/// Number of defined sub-operations
pub const SUBOP_COUNT: u8 = 0x28;

/// 512 bit flag in the immediate
pub const WIDE: u32 = 1 << 24;
/// Maximal slot count of one vload / vstore
pub const MAX_TRANSFER_SLOTS: usize = 32;

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
    /// Second source slot, slot count - 1 of vload / vstore
    pub vb: u8,
    /// 512 bit operation on slot pairs
    pub wide: bool,
    /// Scalar destination register (vext64, vmovmsk8)
    pub dst: u8,
    /// Scalar source register
    pub src: u8,
    /// Offset, immediate or lane
    pub off: i16,
}

/// Sub-operations which have a 512 bit form
pub fn has_wide_form(subop: u8) -> bool {
    !matches!(subop, VLOAD | VSTORE | VTRANSPOSE8 | VINS64 | VEXT64) && subop < SUBOP_COUNT
}

/// Sub-operations which read the scalar source register
pub fn uses_src(subop: u8) -> bool {
    matches!(subop, VLOAD | VSTORE | VBCAST32 | VBCAST64 | VINS64)
}

/// Sub-operations which write the scalar destination register
pub fn writes_dst(subop: u8) -> bool {
    matches!(subop, VEXT64 | VMOVMSK8 | VCOMPRESSLT16)
}

/// Operand validation shared by the opcode form and the intrinsic call form
pub(crate) fn validate_slot_op(op: &VecInsn) -> Option<()> {
    if op.subop >= SUBOP_COUNT || op.src > 10 || op.dst > 9 {
        return None;
    }
    if (!uses_src(op.subop) && op.src != 0) || (!writes_dst(op.subop) && op.dst != 0) {
        return None;
    }
    let off_ok = match op.subop {
        VLOAD | VSTORE => true,
        VROR32 | VSRL32 | VSLL32 => (0..32).contains(&op.off),
        VSHUFD | VBLENDD | VTERNLOG | VBLENDW => (0..256).contains(&op.off),
        VINS64 | VEXT64 => (0..4).contains(&op.off),
        _ => op.off == 0,
    };
    if !off_ok {
        return None;
    }
    match op.subop {
        VTRANSPOSE8 if op.vd as usize > ebpf::VECTOR_SLOTS - 8 => return None,
        VTRANSPOSE64 if op.vd as usize > ebpf::VECTOR_SLOTS - if op.wide { 16 } else { 4 } => return None,
        VLOAD | VSTORE => {
            let first = if op.subop == VLOAD { op.vd } else { op.va } as usize;
            if op.vb as usize >= MAX_TRANSFER_SLOTS || first + op.vb as usize >= ebpf::VECTOR_SLOTS {
                return None;
            }
        }
        _ => {}
    }
    if op.wide && (!has_wide_form(op.subop) || (op.vd | op.va | op.vb) & 1 != 0) {
        return None;
    }
    Some(())
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
            wide: imm & WIDE != 0,
            dst: insn.dst,
            src: insn.src,
            off: insn.off,
        };
        validate_slot_op(&op)?;
        Some(op)
    }

    /// Encodes the instruction
    pub fn encode(&self) -> ebpf::Insn {
        let imm = (self.vd as u32 & 63)
            | ((self.va as u32 & 63) << 6)
            | ((self.vb as u32 & 63) << 12)
            | ((self.subop as u32 & 63) << 18)
            | if self.wide { WIDE } else { 0 };
        ebpf::Insn {
            ptr: 0,
            opc: ebpf::VOP,
            dst: self.dst,
            src: self.src,
            off: self.off,
            imm: imm as i32 as i64,
        }
    }

    /// Slot count of vload / vstore
    pub fn transfer_slots(&self) -> usize {
        self.vb as usize + 1
    }

    /// Number of 256 bit halves of every vector operand (1, or 2 for the 512 bit form)
    pub fn halves(&self) -> usize {
        1 + self.wide as usize
    }
}

/// Copies `slots` vectors between guest memory at `vm_addr` and the vector file starting at
/// `slot` (vload if `!store`), with one translation of the whole guest range
pub fn transfer(
    store: bool,
    slot: u8,
    slots: usize,
    vm_addr: u64,
    vector_base: *mut u8,
    mapping: &mut MemoryMapping,
) -> Result<(), EbpfError> {
    let len = slots * ebpf::VECTOR_WIDTH;
    let access_type = if store { AccessType::Store } else { AccessType::Load };
    let host = match mapping.map_with_access_violation_handler(access_type, vm_addr, len as u64) {
        StableResult::Ok(host) => host.ptr().cast::<u8>() as usize as *mut u8,
        StableResult::Err(err) => return Err(err),
    };
    if vector_base.is_null() {
        return Err(EbpfError::UnsupportedInstruction);
    }
    let vector = unsafe { vector_base.add(slot as usize * ebpf::VECTOR_WIDTH) };
    unsafe {
        if store {
            std::ptr::copy_nonoverlapping(vector, host, len);
        } else {
            std::ptr::copy_nonoverlapping(host, vector, len);
        }
    }
    Ok(())
}

/// Entry point of the JIT for [transfer]: `spec` = `store | slot << 8 | slots << 16`
#[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_transfer(
    result: *mut ProgramResult,
    mapping: *mut MemoryMapping,
    vector_base: *mut u8,
    vm_addr: u64,
    spec: u64,
) {
    let store = spec & 1 != 0;
    let slot = (spec >> 8) as u8;
    let slots = (spec >> 16) as usize;
    unsafe {
        if let Err(err) = transfer(store, slot, slots, vm_addr, vector_base, &mut *mapping) {
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
        VSUB32 => (0..8).for_each(|i| out[i] = a[i].wrapping_sub(b[i])),
        VMULLO32 => (0..8).for_each(|i| out[i] = a[i].wrapping_mul(b[i])),
        VMINU32 => (0..8).for_each(|i| out[i] = a[i].min(b[i])),
        VSRL32 => (0..8).for_each(|i| out[i] = a[i] >> (op.off as u32 & 31)),
        VSLL32 => (0..8).for_each(|i| out[i] = a[i] << (op.off as u32 & 31)),
        VSLLV32 => (0..8).for_each(|i| out[i] = if b[i] > 31 { 0 } else { a[i] << b[i] }),
        VSRLV32 => (0..8).for_each(|i| out[i] = if b[i] > 31 { 0 } else { a[i] >> b[i] }),
        VANDN => (0..8).for_each(|i| out[i] = !a[i] & b[i]),
        VROLV32 => (0..8).for_each(|i| out[i] = a[i].rotate_left(b[i] % 32)),
        VROLV64 => (0..4).for_each(|i| {
            let q = |x: &Lanes| x[2 * i] as u64 | (x[2 * i + 1] as u64) << 32;
            let r = q(a).rotate_left((q(b) % 64) as u32);
            (out[2 * i], out[2 * i + 1]) = (r as u32, (r >> 32) as u32);
        }),
        VADD16 | VSUB16 | VMULLO16 | VMULHI16 | VMINU16 | VBLENDW => {
            let (a, b) = (words(a), words(b));
            let mut w = [0u16; 16];
            for i in 0..16 {
                w[i] = match op.subop {
                    VADD16 => a[i].wrapping_add(b[i]),
                    VSUB16 => a[i].wrapping_sub(b[i]),
                    VMULLO16 => a[i].wrapping_mul(b[i]),
                    VMULHI16 => ((a[i] as i16 as i32 * b[i] as i16 as i32) >> 16) as u16,
                    VMINU16 => a[i].min(b[i]),
                    _ => if (op.off >> (i % 8)) & 1 != 0 { b[i] } else { a[i] },
                };
            }
            out = from_words(&w);
        }
        VMADD16 => {
            let (a, b) = (words(a), words(b));
            let p = |i: usize| a[i] as i16 as i32 * b[i] as i16 as i32;
            (0..8).for_each(|i| out[i] = p(2 * i).wrapping_add(p(2 * i + 1)) as u32);
        }
        VSHUFB => {
            let bytes = |x: &Lanes| -> [u8; 32] { std::array::from_fn(|i| (x[i / 4] >> (8 * (i % 4))) as u8) };
            let (x, idx) = (bytes(a), bytes(b));
            let r: [u8; 32] = std::array::from_fn(|i| if idx[i] & 0x80 != 0 { 0 } else { x[(i & !15) + (idx[i] & 15) as usize] });
            out = std::array::from_fn(|i| u32::from_le_bytes([r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]]));
        }
        VMOV => {}
        _ => unreachable!(),
    }
    out
}

/// The sixteen u16 lanes of a vector
pub fn words(x: &Lanes) -> [u16; 16] {
    std::array::from_fn(|i| (x[i / 2] >> (16 * (i % 2))) as u16)
}

/// Inverse of [words]
pub fn from_words(w: &[u16; 16]) -> Lanes {
    std::array::from_fn(|i| w[2 * i] as u32 | (w[2 * i + 1] as u32) << 16)
}

/// `f(d, a, b)` bitwise for the truth table `imm` (vpternlogd)
pub fn ternlog(d: u32, a: u32, b: u32, imm: u8) -> u32 {
    let pick = |x: u32, set: bool| if set { x } else { !x };
    (0..8u32).filter(|i| (imm >> i) & 1 != 0).fold(0, |r, i| {
        r | (pick(d, i & 4 != 0) & pick(a, i & 2 != 0) & pick(b, i & 1 != 0))
    })
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

/// Lane-wise result of a two (three for vternlog) operand sub-operation on one 256 bit half
fn compute_half(op: &VecInsn, a: &Lanes, b: &Lanes, d: &Lanes) -> Lanes {
    if op.subop == VTERNLOG {
        std::array::from_fn(|i| ternlog(d[i], a[i], b[i], op.off as u8))
    } else {
        compute(op, a, b)
    }
}

/// Executes one vector instruction in the interpreter, returns the scalar result (vext64,
/// vmovmsk8, 0 otherwise). Never inlined: it would bloat the interpreter's instruction loop.
#[inline(never)]
pub fn interpret(
    op: &VecInsn,
    reg: &[u64],
    vector_base: *mut u8,
    mapping: &mut MemoryMapping,
) -> Result<u64, EbpfError> {
    if matches!(op.subop, VLOAD | VSTORE) {
        let vm_addr = reg[op.src as usize].wrapping_add(op.off as i64 as u64);
        let slot = if op.subop == VLOAD { op.vd } else { op.va };
        transfer(op.subop == VSTORE, slot, op.transfer_slots(), vm_addr, vector_base, mapping)?;
        return Ok(0);
    }
    if vector_base.is_null() {
        return Err(EbpfError::UnsupportedInstruction);
    }
    let slot = |n: u8| unsafe { vector_base.add(n as usize * ebpf::VECTOR_WIDTH) };
    let read = |n: u8| read_lanes(slot(n));
    let scalar = reg[op.src as usize];
    match op.subop {
        VTRANSPOSE8 => {
            let mut rows: [Lanes; 8] = std::array::from_fn(|i| read(op.vd + i as u8));
            transpose8(&mut rows);
            for (i, row) in rows.iter().enumerate() {
                write_lanes(slot(op.vd + i as u8), row);
            }
            return Ok(0);
        }
        VTRANSPOSE64 => {
            // Row r is slot vd + r (4 lanes) or pair vd + 2 r (8 lanes)
            let n = 4 * op.halves();
            let lane = |r: usize, c: usize| unsafe { slot(op.vd).add(r * 8 * n + 8 * c).cast::<u64>() };
            let m: Vec<u64> = (0..n * n).map(|i| unsafe { std::ptr::read_unaligned(lane(i / n, i % n)) }).collect();
            for r in 0..n {
                for c in 0..n {
                    unsafe { std::ptr::write_unaligned(lane(r, c), m[c * n + r]) };
                }
            }
            return Ok(0);
        }
        VINS64 => {
            unsafe { std::ptr::write_unaligned(slot(op.vd).add(8 * op.off as usize).cast::<u64>(), scalar.to_le()) };
            return Ok(0);
        }
        VEXT64 => {
            return Ok(u64::from_le(unsafe { std::ptr::read_unaligned(slot(op.va).add(8 * op.off as usize).cast::<u64>()) }));
        }
        VCOMPRESSLT16 => {
            let n = 16 * op.halves();
            let word = |s: u8, i: usize| unsafe { std::ptr::read_unaligned(slot(s).add(2 * i).cast::<u16>()) };
            let (a, b): (Vec<u16>, Vec<u16>) = (0..n).map(|i| (word(op.va, i), word(op.vb, i))).unzip();
            let kept: Vec<u16> = (0..n).filter(|&i| a[i] < b[i]).map(|i| a[i]).collect();
            for i in 0..n {
                unsafe { std::ptr::write_unaligned(slot(op.vd).add(2 * i).cast::<u16>(), kept.get(i).copied().unwrap_or(0)) };
            }
            return Ok(kept.len() as u64);
        }
        VMOVMSK8 => {
            let mut mask = 0u64;
            for i in 0..op.halves() * ebpf::VECTOR_WIDTH {
                mask |= ((unsafe { *slot(op.va).add(i) } >> 7) as u64) << i;
            }
            return Ok(mask);
        }
        _ => {}
    }
    let halves = op.halves();
    if !op.wide && !matches!(op.subop, VBCAST32 | VBCAST64 | VPERMI2D | VPERMT2D) {
        // The common case: one 256 bit lane-wise operation
        let (a, b) = (read(op.va), read(op.vb));
        let out = if op.subop == VTERNLOG { compute_half(op, &a, &b, &read(op.vd)) } else { compute(op, &a, &b) };
        write_lanes(slot(op.vd), &out);
        return Ok(0);
    }
    // Every source is read before the destination is written
    let reads_d = matches!(op.subop, VTERNLOG | VPERMI2D | VPERMT2D);
    let a: [Lanes; 2] = std::array::from_fn(|h| if h < halves { read(op.va + h as u8) } else { [0; 8] });
    let b: [Lanes; 2] = std::array::from_fn(|h| if h < halves { read(op.vb + h as u8) } else { [0; 8] });
    let d: [Lanes; 2] = std::array::from_fn(|h| if h < halves && reads_d { read(op.vd + h as u8) } else { [0; 8] });
    let lanes = 8 * halves;
    let get = |x: &[Lanes; 2], i: usize| x[i / 8][i % 8];
    let out: [Lanes; 2] = match op.subop {
        VBCAST32 => [[scalar as u32; 8]; 2],
        VBCAST64 => [std::array::from_fn(|i| (scalar >> (32 * (i % 2))) as u32); 2],
        // Cross lane over all lanes of the operand
        VPERMD => std::array::from_fn(|h| std::array::from_fn(|i| get(&a, (get(&b, 8 * h + i) as usize) & (lanes - 1)))),
        VPERMI2D => std::array::from_fn(|h| {
            std::array::from_fn(|i| {
                let index = get(&d, 8 * h + i) as usize & (2 * lanes - 1);
                if index < lanes { get(&a, index) } else { get(&b, index - lanes) }
            })
        }),
        VPERMT2D => std::array::from_fn(|h| {
            std::array::from_fn(|i| {
                let index = get(&a, 8 * h + i) as usize & (2 * lanes - 1);
                if index < lanes { get(&d, index) } else { get(&b, index - lanes) }
            })
        }),
        _ => std::array::from_fn(|h| compute_half(op, &a[h], &b[h], &d[h])),
    };
    for (h, half) in out.iter().enumerate().take(halves) {
        write_lanes(slot(op.vd + h as u8), half);
    }
    Ok(0)
}

/// Mnemonic of a sub-operation (the intrinsic name without `sol_<isa>_`, prefixed with `v`)
pub fn mnemonic(subop: u8, wide: bool) -> String {
    let name = if wide { intrinsics::NAMES_512[subop as usize] } else { Some(intrinsics::NAMES_256[subop as usize]) };
    match name {
        Some(name) => format!("v{}", name.splitn(3, '_').nth(2).unwrap()),
        None => "vinvalid".to_string(),
    }
}

/// Textual representation of a vector instruction
pub fn disassemble(insn: &ebpf::Insn) -> String {
    let Some(op) = VecInsn::decode(insn) else {
        return format!("vop invalid imm={:#x}", insn.imm);
    };
    let name = mnemonic(op.subop, op.wide);
    let addr = format!("[r{}{:+}]", op.src, op.off);
    let range = |first: u8| {
        if op.transfer_slots() == 1 { format!("v{first}") } else { format!("v{first}..v{}", first as usize + op.vb as usize) }
    };
    let (vd, va, vb) = (op.vd, op.va, op.vb);
    match op.subop {
        VLOAD => format!("{name} {}, {addr}", range(vd)),
        VSTORE => format!("{name} {addr}, {}", range(va)),
        VTRANSPOSE8 => format!("{name} v{vd}..v{}", vd + 7),
        VTRANSPOSE64 => format!("{name} v{vd}..v{}", vd as usize + if op.wide { 15 } else { 3 }),
        VBCAST32 | VBCAST64 => format!("{name} v{vd}, r{}", op.src),
        VINS64 => format!("{name} v{vd}, r{}, {}", op.src, op.off),
        VEXT64 => format!("{name} r{}, v{va}, {}", op.dst, op.off),
        VMOVMSK8 => format!("{name} r{}, v{va}", op.dst),
        VCOMPRESSLT16 => format!("{name} r{}, v{vd}, v{va}, v{vb}", op.dst),
        VMOV => format!("{name} v{vd}, v{va}"),
        VROR32 | VSRL32 | VSLL32 => format!("{name} v{vd}, v{va}, {}", op.off),
        VSHUFD => format!("{name} v{vd}, v{va}, {:#x}", op.off),
        VBLENDD | VBLENDW | VTERNLOG => format!("{name} v{vd}, v{va}, v{vb}, {:#x}", op.off),
        _ => format!("{name} v{vd}, v{va}, v{vb}"),
    }
}

/// Host memory of a vector file.
///
/// The VM allocates its own when it runs a program with vector instructions. An embedder may
/// provide one instead (see [crate::vm::EbpfVm::set_vector_memory]), e.g. to inspect the slots
/// after a run. It is zeroed by the VM at the beginning of every program invocation.
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

/// Opcodes which neither touch memory, branch, call, nor throw
pub(crate) fn is_pure_alu(opc: u8) -> bool {
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

/// Static control flow facts about a program, shared by the interpreter and the JIT
#[derive(Debug, Default)]
pub struct SimdAnalysis {
    /// Basic block starts (jump and call targets, the entry point)
    pub jump_targets: Vec<bool>,
    /// Instructions a `callx` may not enter: those inside a run of vector code, i.e. after a
    /// vector instruction with only pure scalar ALU instructions in between and no block start.
    /// There the JIT keeps vector state in registers. Entering one throws UnsupportedInstruction
    /// in every engine.
    pub callx_forbidden: Vec<bool>,
    /// Vector instructions (opcode form and intrinsic calls)
    pub vector_instructions: usize,
}

impl SimdAnalysis {
    /// Whether `callx` may not enter instruction `pc`
    pub fn callx_forbidden(&self, pc: usize) -> bool {
        self.callx_forbidden.get(pc).copied().unwrap_or(false)
    }
}

/// Computes [SimdAnalysis]
pub fn analyze<C: crate::vm::ContextObject>(executable: &crate::elf::Executable<C>) -> SimdAnalysis {
    let sbpf_version = executable.get_sbpf_version();
    let (_, program) = executable.get_text_bytes();
    let len = program.len() / ebpf::INSN_SIZE;
    let intrinsic_hashes: Vec<u32> = intrinsics::hashes().iter().map(|(hash, _, _)| *hash).collect();
    let is_vector = |insn: &ebpf::Insn| {
        insn.opc == ebpf::VOP
            || (insn.opc == ebpf::CALL_IMM && insn.src == 0 && intrinsic_hashes.contains(&(insn.imm as u32)))
    };
    let step = |insn: &ebpf::Insn| if insn.opc == ebpf::LD_DW_IMM && !sbpf_version.disable_lddw() { 2 } else { 1 };
    let mut jump_targets = vec![false; len + 1];
    let mut vector_instructions = 0;
    let mut pc = 0;
    while pc < len {
        let insn = ebpf::get_insn_unchecked(program, pc);
        let class = insn.opc & ebpf::BPF_CLS_MASK;
        if is_vector(&insn) {
            vector_instructions += 1;
        } else if insn.opc == ebpf::CALL_IMM {
            if sbpf_version.static_syscalls() && insn.src == 1 {
                let target = pc as i64 + insn.imm + 1;
                if (0..len as i64).contains(&target) {
                    jump_targets[target as usize] = true;
                }
            } else if let Some((_, target)) = executable.get_function_registry().lookup_by_key(insn.imm as u32) {
                if target < len {
                    jump_targets[target] = true;
                }
            }
        } else if insn.opc != ebpf::EXIT
            && insn.opc != ebpf::CALL_REG
            && (class == ebpf::BPF_JMP64 || (class == ebpf::BPF_JMP32 && sbpf_version.enable_jmp32()))
        {
            let target = pc as i64 + insn.off as i64 + 1;
            if (0..len as i64).contains(&target) {
                jump_targets[target as usize] = true;
            }
        }
        pc += step(&insn);
    }
    jump_targets[executable.get_entrypoint_instruction_offset().min(len)] = true;
    let mut callx_forbidden = vec![false; len + 1];
    let mut in_vector_run = false;
    let mut pc = 0;
    while pc < len {
        let insn = ebpf::get_insn_unchecked(program, pc);
        if jump_targets[pc] {
            in_vector_run = false;
        }
        callx_forbidden[pc] = in_vector_run;
        if is_vector(&insn) {
            in_vector_run = true;
        } else if !is_pure_alu(insn.opc) {
            in_vector_run = false;
        }
        pc += step(&insn);
    }
    SimdAnalysis { jump_targets, callx_forbidden, vector_instructions }
}
