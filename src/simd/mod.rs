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
//!       bits 18..23  sub-operation (low 6 bits)
//!       bit  24      512 bit operation on slot pairs
//!       bit  25      sub-operation (bit 6)
//!       bits 26..31  reserved, must be zero
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
// 0x27 is reserved (a removed fused compare + compress)
/// `v[vd].u64[i] = v[va].u64[i] + v[vb].u64[i]` (wrapping)
pub const VADD64: u8 = 0x28;
/// `v[vd].u64[i] = v[va].u64[i] - v[vb].u64[i]` (wrapping)
pub const VSUB64: u8 = 0x29;
/// `v[vd].u64[i] = v[va].u64[i] << off` (`off` in 0..64)
pub const VSLL64: u8 = 0x2a;
/// `v[vd].u64[i] = v[va].u64[i] >> off` (logical, `off` in 0..64)
pub const VSRL64: u8 = 0x2b;
/// `v[vd].i64[i] = v[va].i64[i] >> off` (arithmetic, `off` in 0..64)
pub const VSRA64: u8 = 0x2c;
/// `v[vd].u64[i] = v[va].u64[i] << v[vb].u64[i]` (0 for counts above 63)
pub const VSLLV64: u8 = 0x2d;
/// `v[vd].u64[i] = v[va].u64[i] >> v[vb].u64[i]` (0 for counts above 63)
pub const VSRLV64: u8 = 0x2e;
/// `v[vd].u64[i] = (v[va].u64[i] as u32) * (v[vb].u64[i] as u32)` (full 64 bit products, vpmuludq)
pub const VMULU32: u8 = 0x2f;
/// `v[vd].u64[i] = v[va].u64[i] * v[vb].u64[i]` (low 64 bits, vpmullq)
pub const VMULLO64: u8 = 0x30;
/// `v[vd].u64[i] += (lo52(a) * lo52(b)) mod 2^52` for the low 52 bits of the u64 lanes (vpmadd52luq)
pub const VMADD52LO: u8 = 0x31;
/// `v[vd].u64[i] += (lo52(a) * lo52(b)) >> 52` (vpmadd52huq)
pub const VMADD52HI: u8 = 0x32;
/// `r[dst]` = bit `i` set if `pred(v[va][i], v[vb][i])` for the lanes of size `8 << (off >> 4)` bits;
/// `off & 7` is the predicate (0 eq, 1 lt, 2 le, 3 false, 4 ne, 5 ge, 6 gt, 7 true, as vpcmp),
/// `off & 8` unsigned (vpcmp{b,w,d,q} / vpcmpu*)
pub const VCMP: u8 = 0x33;
/// `v[vd][i] = r[src] bit i ? v[va][i] : v[vb][i]` for lanes of `8 << off` bits (vpblendm*)
pub const VBLENDM: u8 = 0x34;
/// The lanes `i` of `v[va]` with `r[src]` bit `i` set, in order, packed into the low lanes of `v[vd]`
/// (the others zero); `r[dst]` = their number. Lanes of `8 << off` bits (vpcompress*)
pub const VCOMPRESS: u8 = 0x35;
/// `v[vd][i] = r[src] bit i ? v[va][k++] : 0` (the low lanes of `v[va]` spread to the set bits),
/// lanes of `8 << off` bits (vpexpand*)
pub const VEXPAND: u8 = 0x36;
/// Signed 16 bit Montgomery product: `v[vd] = mulhi(a, w) - mulhi(mullo(a, w'), q)` per i16 lane
/// with `a = v[va]`, `w = v[vb]`, `w' = v[vb + 2]` (w q^-1 mod 2^16), `q = v[off]` (4 AVX2
/// instructions: vpmullw, vpmulhw, vpmulhw, vpsubw)
pub const VMONT16: u8 = 0x37;
/// u16 lanes shifted by the immediate `off` (0..16): left / logical right / arithmetic right
pub const VSLL16: u8 = 0x38;
#[allow(missing_docs)]
pub const VSRL16: u8 = 0x39;
#[allow(missing_docs)]
pub const VSRA16: u8 = 0x3a;
/// i32 lanes shifted arithmetically right by the immediate `off` (0..32)
pub const VSRA32: u8 = 0x3b;
/// u16 lanes shifted by the lanes of `v[vb]`: left / logical right (0 above 15), arithmetic right
/// (sign fill above 15)
pub const VSLLV16: u8 = 0x3c;
#[allow(missing_docs)]
pub const VSRLV16: u8 = 0x3d;
#[allow(missing_docs)]
pub const VSRAV16: u8 = 0x3e;
/// i32 / i64 lanes shifted arithmetically right by the lanes of `v[vb]` (sign fill above 31 / 63)
pub const VSRAV32: u8 = 0x3f;
#[allow(missing_docs)]
pub const VSRAV64: u8 = 0x40;
/// u64 lanes rotated right by the immediate `off` (0..64)
pub const VROR64: u8 = 0x41;
/// Per 128 bit lane, the low (high) halves of `v[va]` and `v[vb]` interleaved, lanes of `8 << off`
/// bits (vpunpckl* / vpunpckh*)
pub const VUNPCKLO: u8 = 0x42;
#[allow(missing_docs)]
pub const VUNPCKHI: u8 = 0x43;
/// Per 128 bit lane, the lanes of `v[va]` then `v[vb]` narrowed with signed (unsigned) saturation:
/// `off` 0 i16 -> i8 (u8), 1 i32 -> i16 (u16) (vpackss* / vpackus*)
pub const VPACKSS: u8 = 0x44;
#[allow(missing_docs)]
pub const VPACKUS: u8 = 0x45;
/// The low lanes of `v[va]` zero (sign) extended to fill `v[vd]`; `off` = source lane size (0..3,
/// log2 bytes) | destination lane size << 2, destination larger (vpmovzx* / vpmovsx*)
pub const VZEXT: u8 = 0x46;
#[allow(missing_docs)]
pub const VSEXT: u8 = 0x47;
/// Per lane population count, lanes of `8 << off` bits
pub const VPOPCNT: u8 = 0x48;
/// Per lane count of leading zeros, lanes of `8 << off` bits (`off` 2 or 3)
pub const VLZCNT: u8 = 0x49;
/// `v[vd].u8[i] = v[va].u8[v[vb].u8[i] & (n - 1)]` over all `n` bytes (vpermb)
pub const VPERMB: u8 = 0x4a;
/// `v[vd].u8[i] = (v[va] ++ v[vb]).u8[v[vd].u8[i] & (2 n - 1)]` (vpermi2b)
pub const VPERMI2B: u8 = 0x4b;
/// `v[vd].u8[8 q + j] = v[va].u64[q].rotate_right(v[vb].u8[8 q + j] & 63) as u8` (vpmultishiftqb)
pub const VMULTISHIFT: u8 = 0x4c;
/// Per byte affine transformation over GF(2): bit `i` of the result = parity(A.u8[7 - i] & x) ^ bit
/// `i` of `off`, with x = a byte of `v[va]` and A the u64 of `v[vb]` holding it (gf2p8affineqb)
pub const VGF2P8AFFINE: u8 = 0x4d;
/// Per byte product in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1 (gf2p8mulb)
pub const VGF2P8MUL: u8 = 0x4e;
// 0x4f..=0x52 are reserved (removed AES rounds)
/// Per 128 bit lane, the carry-less product of u64 `off & 1` of `v[va]` and u64 `(off >> 4) & 1` of
/// `v[vb]` (pclmulqdq)
pub const VCLMUL: u8 = 0x53;
/// Per lane minimum / maximum; `off` = lane size (log2 bytes) | 4 if unsigned (vpmin* / vpmax*)
pub const VMIN: u8 = 0x54;
#[allow(missing_docs)]
pub const VMAX: u8 = 0x55;
/// Per lane absolute value (the minimum stays), lanes of `8 << off` bits (vpabs*)
pub const VABS: u8 = 0x56;
/// Saturating add / subtract; `off` = lane size (0: 8, 1: 16 bit) | 4 if unsigned (vpadds*,
/// vpaddus*, vpsubs*, vpsubus*)
pub const VADDS: u8 = 0x57;
#[allow(missing_docs)]
pub const VSUBS: u8 = 0x58;
/// Unsigned rounding average `(a + b + 1) >> 1`, lanes of `8 << off` bits (`off` 0 / 1) (vpavg*)
pub const VAVG: u8 = 0x59;
/// Wrapping u8 add / subtract
pub const VADD8: u8 = 0x5a;
#[allow(missing_docs)]
pub const VSUB8: u8 = 0x5b;
/// High half of unsigned 16 x 16 bit products (vpmulhuw)
pub const VMULHU16: u8 = 0x5c;
/// `(a * b + 0x4000) >> 15` on i16 (vpmulhrsw)
pub const VMULHRS16: u8 = 0x5d;
/// `(a.u64[i] as i32 as i64) * (b.u64[i] as i32 as i64)` (vpmuldq)
pub const VMULS32: u8 = 0x5e;
/// u8 x i8 products summed in pairs into i16 with signed saturation (vpmaddubsw)
pub const VMADDUBS16: u8 = 0x5f;
/// The lanes of `v[va]` truncated into the low part of `v[vd]` (the rest zero); `off` = source lane
/// size | destination lane size << 2, destination smaller (vpmov{wb,db,qb,dw,qw,qd})
pub const VTRUNC: u8 = 0x60;
/// u64 lanes permuted by the immediate within each 256 bit half (vpermq imm)
pub const VPERMQ: u8 = 0x61;
/// `v[vd].u16[i] = v[va].u16[v[vb].u16[i] & (n - 1)]` over all `n` u16 lanes (vpermw)
pub const VPERMW: u8 = 0x62;
/// Per 128 bit lane, bytes `off..off + 16` of (b lane ++ a lane), zeros beyond (vpalignr)
pub const VALIGNR: u8 = 0x63;
/// Per 128 bit lane, the low (high) four u16 permuted by the immediate (vpshuflw / vpshufhw)
pub const VSHUFLW: u8 = 0x64;
#[allow(missing_docs)]
pub const VSHUFHW: u8 = 0x65;
/// `v[vd][i]` = bit mask of the lanes `j < i` equal to lane `i`, lanes of `8 << off` bits (2 / 3)
/// (vpconflict*)
pub const VCONFLICT: u8 = 0x66;
/// `r[dst]` = bit `i` set if `v[va][i] & v[vb][i] != 0` (== 0 if `off & 4`), lanes of
/// `8 << (off & 3)` bits (vptestm* / vptestnm*)
pub const VTESTM: u8 = 0x67;
/// `v[vd].i32[i] += sum of the four u8 x i8 (i16 x i16 pairs for dpwssd) products of lane i`,
/// wrapping (vpdpbusd / vpdpwssd)
pub const VDPBUSD: u8 = 0x68;
#[allow(missing_docs)]
pub const VDPWSSD: u8 = 0x69;
/// Sum of absolute u8 differences of each 8 byte group into its u64 lane (vpsadbw)
pub const VSADBW: u8 = 0x6a;
/// Funnel shifts by the immediate: `off` = count | lane size << 6 (1..3: 16, 32, 64 bit);
/// shld `(a << c) | (b >> (w - c))`, shrd `(a >> c) | (b << (w - c))` (vpshld* / vpshrd*)
pub const VSHLD: u8 = 0x6b;
#[allow(missing_docs)]
pub const VSHRD: u8 = 0x6c;
/// Funnel shifts by the lanes of `v[vb]` (mod w) into the accumulator `v[vd]`: shldv
/// `(d << c) | (a >> (w - c))`, shrdv `(d >> c) | (a << (w - c))`; `off` = lane size (1..3)
/// (vpshldv* / vpshrdv*)
pub const VSHLDV: u8 = 0x6d;
#[allow(missing_docs)]
pub const VSHRDV: u8 = 0x6e;
/// Masked load / store of the lanes of `8 << vb` bits whose bit is set in `r[dst]` (the mask
/// register) at `r[src] + off`: only the span of the set lanes is translated, unset lanes load as
/// zero and are not written
pub const VLOADM: u8 = 0x6f;
#[allow(missing_docs)]
pub const VSTOREM: u8 = 0x70;
/// Number of defined sub-operations
pub const SUBOP_COUNT: u8 = 0x71;

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

/// Sub-operation numbers which are not (or no longer) defined
pub fn is_reserved(subop: u8) -> bool {
    matches!(subop, 0x27 | 0x4f..=0x52) || subop >= SUBOP_COUNT
}

/// Sub-operations which have a 512 bit form
pub fn has_wide_form(subop: u8) -> bool {
    !matches!(subop, VLOAD | VSTORE | VTRANSPOSE8 | VINS64 | VEXT64) && !is_reserved(subop)
}

/// Masked load / store with one translation of the span of the set lanes (lanes of `1 << size`
/// bytes, `halves` 256 bit halves)
#[allow(clippy::too_many_arguments)]
pub(crate) fn transfer_masked(
    store: bool,
    slot: u8,
    size: u32,
    halves: usize,
    mask: u64,
    vm_addr: u64,
    vector_base: *mut u8,
    mapping: &mut MemoryMapping,
) -> Result<(), EbpfError> {
    if vector_base.is_null() {
        return Err(EbpfError::UnsupportedInstruction);
    }
    let (s, n) = (1usize << size, (ebpf::VECTOR_WIDTH * halves) >> size);
    let mask = if n == 64 {
        mask
    } else {
        mask & ((1u64 << n) - 1)
    };
    let vector = unsafe { vector_base.add(slot as usize * ebpf::VECTOR_WIDTH) };
    if mask == 0 {
        if !store {
            unsafe { std::ptr::write_bytes(vector, 0, n * s) };
        }
        return Ok(());
    }
    let (first, last) = (
        mask.trailing_zeros() as usize,
        63 - mask.leading_zeros() as usize,
    );
    let access_type = if store {
        AccessType::Store
    } else {
        AccessType::Load
    };
    let span = ((last - first + 1) * s) as u64;
    let host = match mapping.map_with_access_violation_handler(
        access_type,
        vm_addr.wrapping_add((first * s) as u64),
        span,
    ) {
        StableResult::Ok(host) => host.ptr().cast::<u8>() as usize as *mut u8,
        StableResult::Err(err) => return Err(err),
    };
    for i in 0..n {
        let lane = unsafe { vector.add(i * s) };
        if (mask >> i) & 1 != 0 {
            let mem = unsafe { host.add((i - first) * s) };
            unsafe {
                if store {
                    std::ptr::copy_nonoverlapping(lane, mem, s);
                } else {
                    std::ptr::copy_nonoverlapping(mem, lane, s);
                }
            }
        } else if !store {
            unsafe { std::ptr::write_bytes(lane, 0, s) };
        }
    }
    Ok(())
}

/// Entry point of the JIT for [transfer_masked]: `spec` = `store | slot << 8 | size << 16 |
/// halves << 24`
#[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_transfer_masked(
    result: *mut ProgramResult,
    mapping: *mut MemoryMapping,
    vector_base: *mut u8,
    vm_addr: u64,
    spec: u64,
    mask: u64,
) {
    let (store, slot, size, halves) = (
        spec & 1 != 0,
        (spec >> 8) as u8,
        ((spec >> 16) & 0xff) as u32,
        (spec >> 24) as usize,
    );
    unsafe {
        if let Err(err) = transfer_masked(
            store,
            slot,
            size,
            halves,
            mask,
            vm_addr,
            vector_base,
            &mut *mapping,
        ) {
            *result = ProgramResult::Err(err);
        }
    }
}

/// Sub-operations which read the scalar source register
pub fn uses_src(subop: u8) -> bool {
    matches!(
        subop,
        VLOAD
            | VSTORE
            | VBCAST32
            | VBCAST64
            | VINS64
            | VBLENDM
            | VCOMPRESS
            | VEXPAND
            | VLOADM
            | VSTOREM
    )
}

/// Sub-operations which read a mask from the register in the `dst` field (masked load / store)
pub fn reads_mask_register(subop: u8) -> bool {
    matches!(subop, VLOADM | VSTOREM)
}

/// Sub-operations which write the scalar destination register
pub fn writes_dst(subop: u8) -> bool {
    matches!(subop, VEXT64 | VMOVMSK8 | VCMP | VCOMPRESS | VTESTM)
}

/// Operand validation shared by the opcode form and the intrinsic call form
pub(crate) fn validate_slot_op(op: &VecInsn) -> Option<()> {
    if is_reserved(op.subop) || op.src > 10 || op.dst > 9 {
        return None;
    }
    if (!uses_src(op.subop) && op.src != 0)
        || (!writes_dst(op.subop) && !reads_mask_register(op.subop) && op.dst != 0)
    {
        return None;
    }
    let off_ok = match op.subop {
        VLOAD | VSTORE => true,
        VROR32 | VSRL32 | VSLL32 => (0..32).contains(&op.off),
        VSHUFD | VBLENDD | VTERNLOG | VBLENDW => (0..256).contains(&op.off),
        VINS64 | VEXT64 => (0..4).contains(&op.off),
        VSLL64 | VSRL64 | VSRA64 | VROR64 => (0..64).contains(&op.off),
        VSLL16 | VSRL16 | VSRA16 => (0..16).contains(&op.off),
        VSRA32 => (0..32).contains(&op.off),
        VUNPCKLO | VUNPCKHI | VPOPCNT => (0..4).contains(&op.off),
        VPACKSS | VPACKUS => (0..2).contains(&op.off),
        VZEXT | VSEXT => {
            (0..16).contains(&op.off) && (op.off & 3) < 3 && (op.off >> 2) > (op.off & 3)
        }
        VLZCNT => (2..4).contains(&op.off),
        VGF2P8AFFINE => (0..256).contains(&op.off),
        VCLMUL => op.off & !0x11 == 0,
        VMIN | VMAX | VTESTM => (0..8).contains(&op.off),
        VABS => (0..4).contains(&op.off),
        VADDS | VSUBS => (0..8).contains(&op.off) && op.off & 3 < 2,
        VAVG => (0..2).contains(&op.off),
        VTRUNC => (0..16).contains(&op.off) && (op.off >> 2) < (op.off & 3),
        VPERMQ | VSHUFLW | VSHUFHW => (0..256).contains(&op.off),
        VALIGNR => (0..32).contains(&op.off),
        VCONFLICT => (2..4).contains(&op.off),
        VSHLD | VSHRD => {
            (1..4).contains(&(op.off >> 6)) && op.off < 256 && (op.off & 63) < 8 << (op.off >> 6)
        }
        VSHLDV | VSHRDV => (1..4).contains(&op.off),
        VLOADM | VSTOREM => true,
        VCMP => (0..64).contains(&op.off),
        VBLENDM | VCOMPRESS | VEXPAND => (0..4).contains(&op.off),
        VMONT16 => {
            (0..ebpf::VECTOR_SLOTS as i16 - op.halves() as i16 + 1).contains(&op.off)
                && op.vb as usize + 2 + op.halves() <= ebpf::VECTOR_SLOTS
        }
        _ => op.off == 0,
    };
    if !off_ok {
        return None;
    }
    match op.subop {
        VTRANSPOSE8 if op.vd as usize > ebpf::VECTOR_SLOTS - 8 => return None,
        VTRANSPOSE64 if op.vd as usize > ebpf::VECTOR_SLOTS - if op.wide { 16 } else { 4 } => {
            return None
        }
        VLOADM | VSTOREM if op.vb > 3 => return None,
        VLOAD | VSTORE => {
            let first = if op.subop == VLOAD { op.vd } else { op.va } as usize;
            if op.vb as usize >= MAX_TRANSFER_SLOTS || first + op.vb as usize >= ebpf::VECTOR_SLOTS
            {
                return None;
            }
        }
        _ => {}
    }
    let vb_slot = if reads_mask_register(op.subop) {
        0
    } else {
        op.vb
    };
    if op.wide
        && (!has_wide_form(op.subop)
            || (op.vd | op.va | vb_slot) & 1 != 0
            || (op.subop == VMONT16 && op.off & 1 != 0))
    {
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
        if imm >> 26 != 0 {
            return None;
        }
        let op = Self {
            vd: (imm & 63) as u8,
            va: ((imm >> 6) & 63) as u8,
            vb: ((imm >> 12) & 63) as u8,
            subop: ((imm >> 18) & 63) as u8 | (((imm >> 25) & 1) as u8 * 64),
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
            | ((self.subop as u32 >> 6) & 1) << 25
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
pub(crate) fn transfer(
    store: bool,
    slot: u8,
    slots: usize,
    vm_addr: u64,
    vector_base: *mut u8,
    mapping: &mut MemoryMapping,
) -> Result<(), EbpfError> {
    let len = slots * ebpf::VECTOR_WIDTH;
    let access_type = if store {
        AccessType::Store
    } else {
        AccessType::Load
    };
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

/// Packs an operation for [jit_interpret]: 8 bit slots (the intrinsic form reaches all 256),
/// sub-operation, width and offset
#[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
pub(crate) fn encode_for_host(op: &VecInsn) -> u64 {
    op.vd as u64
        | (op.va as u64) << 8
        | (op.vb as u64) << 16
        | (op.subop as u64) << 24
        | (op.wide as u64) << 31
        | (op.off as u16 as u64) << 32
}

/// Entry point of the JIT for operations without a native lowering on the host (no memory access,
/// so no errors): runs the reference implementation, returns the scalar result
#[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_interpret(vector_base: *mut u8, encoded: u64, scalar: u64) -> u64 {
    let op = VecInsn {
        vd: encoded as u8,
        va: (encoded >> 8) as u8,
        vb: (encoded >> 16) as u8,
        subop: (encoded >> 24) as u8 & 127,
        wide: (encoded >> 31) & 1 != 0,
        dst: 0,
        src: 1,
        off: (encoded >> 32) as u16 as i16,
    };
    interpret_registers(&op, &[0, scalar], vector_base).unwrap_or(0)
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

/// Reads eight little-endian u32 lanes from the first 32 bytes
pub fn read_lanes(bytes: &[u8]) -> Lanes {
    load_lanes(bytes[..ebpf::VECTOR_WIDTH].as_ptr())
}

/// Reads eight little-endian u32 lanes (32 readable bytes at `ptr`)
#[inline(always)]
fn load_lanes(ptr: *const u8) -> Lanes {
    let mut lanes = [0u32; 8];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = u32::from_le(unsafe { std::ptr::read_unaligned(ptr.add(4 * i).cast::<u32>()) });
    }
    lanes
}

/// Writes eight little-endian u32 lanes (32 writable bytes at `ptr`)
#[inline(always)]
fn store_lanes(ptr: *mut u8, lanes: &Lanes) {
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
                    _ => {
                        if (op.off >> (i % 8)) & 1 != 0 {
                            b[i]
                        } else {
                            a[i]
                        }
                    }
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
            let bytes = |x: &Lanes| -> [u8; 32] {
                std::array::from_fn(|i| (x[i / 4] >> (8 * (i % 4))) as u8)
            };
            let (x, idx) = (bytes(a), bytes(b));
            let r: [u8; 32] = std::array::from_fn(|i| {
                if idx[i] & 0x80 != 0 {
                    0
                } else {
                    x[(i & !15) + (idx[i] & 15) as usize]
                }
            });
            out = std::array::from_fn(|i| {
                u32::from_le_bytes([r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]])
            });
        }
        VMOV => {}
        VADD64 | VSUB64 | VSLL64 | VSRL64 | VSRA64 | VSLLV64 | VSRLV64 | VMULU32 | VMULLO64 => {
            let q = |x: &Lanes, i: usize| x[2 * i] as u64 | (x[2 * i + 1] as u64) << 32;
            for i in 0..4 {
                let (x, y, n) = (q(a, i), q(b, i), op.off as u32);
                let r = match op.subop {
                    VADD64 => x.wrapping_add(y),
                    VSUB64 => x.wrapping_sub(y),
                    VSLL64 => x << n,
                    VSRL64 => x >> n,
                    VSRA64 => ((x as i64) >> n) as u64,
                    VSLLV64 => {
                        if y > 63 {
                            0
                        } else {
                            x << y
                        }
                    }
                    VSRLV64 => {
                        if y > 63 {
                            0
                        } else {
                            x >> y
                        }
                    }
                    VMULU32 => (x as u32 as u64) * (y as u32 as u64),
                    _ => x.wrapping_mul(y),
                };
                (out[2 * i], out[2 * i + 1]) = (r as u32, (r >> 32) as u32);
            }
        }
        _ => unreachable!(),
    }
    out
}

/// Lanes of `size` bytes as u64 (little-endian)
fn lanes_of(x: &[u8], size: usize) -> Vec<u64> {
    x.chunks(size)
        .map(|c| c.iter().rev().fold(0u64, |v, b| v << 8 | *b as u64))
        .collect()
}

/// Inverse of [lanes_of]
fn from_lanes(lanes: &[u64], size: usize) -> Vec<u8> {
    lanes
        .iter()
        .flat_map(|v| (0..size).map(move |k| (v >> (8 * k)) as u8))
        .collect()
}

/// Product in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1
pub fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut r = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        a = (a << 1) ^ if a & 0x80 != 0 { 0x1b } else { 0 };
        b >>= 1;
    }
    r
}

/// Carry-less product of two u64
pub fn clmul(a: u64, b: u64) -> u128 {
    (0..64)
        .filter(|i| (b >> i) & 1 != 0)
        .fold(0u128, |r, i| r ^ (a as u128) << i)
}

/// Reference semantics of the operations from VSLL16 on, on whole operands as bytes (`d` the old
/// destination, read by vpermi2b)
fn compute_bytes(op: &VecInsn, a: &[u8], b: &[u8], d: &[u8]) -> Vec<u8> {
    use std::convert::TryInto;
    let n = a.len();
    let off = op.off as u32;
    let lanewise = |size: usize, f: &dyn Fn(u64, u64) -> u64| -> Vec<u8> {
        let (x, y) = (lanes_of(a, size), lanes_of(b, size));
        let mask = if size == 8 {
            u64::MAX
        } else {
            (1u64 << (8 * size)) - 1
        };
        from_lanes(
            &x.iter()
                .zip(&y)
                .map(|(x, y)| f(*x, *y) & mask)
                .collect::<Vec<_>>(),
            size,
        )
    };
    let sext = |v: u64, size: usize| ((v << (64 - 8 * size)) as i64 >> (64 - 8 * size)) as u64;
    match op.subop {
        VSLL16 => lanewise(2, &|x, _| x << off),
        VSRL16 => lanewise(2, &|x, _| x >> off),
        VSRA16 => lanewise(2, &|x, _| (sext(x, 2) as i64 >> off) as u64),
        VSRA32 => lanewise(4, &|x, _| (sext(x, 4) as i64 >> off) as u64),
        VSLLV16 => lanewise(2, &|x, c| if c > 15 { 0 } else { x << c }),
        VSRLV16 => lanewise(2, &|x, c| if c > 15 { 0 } else { x >> c }),
        VSRAV16 => lanewise(2, &|x, c| (sext(x, 2) as i64 >> c.min(15)) as u64),
        VSRAV32 => lanewise(4, &|x, c| (sext(x, 4) as i64 >> c.min(31)) as u64),
        VSRAV64 => lanewise(8, &|x, c| (x as i64 >> c.min(63)) as u64),
        VROR64 => lanewise(8, &|x, _| x.rotate_right(off)),
        VUNPCKLO | VUNPCKHI => {
            let size = 1usize << off;
            let mut out = Vec::with_capacity(n);
            for lane in 0..n / 16 {
                let base = 16 * lane + if op.subop == VUNPCKHI { 8 } else { 0 };
                for k in 0..8 / size {
                    out.extend_from_slice(&a[base + size * k..base + size * (k + 1)]);
                    out.extend_from_slice(&b[base + size * k..base + size * (k + 1)]);
                }
            }
            out
        }
        VPACKSS | VPACKUS => {
            let size = 2usize << off;
            let narrow = size / 2;
            let (lo, hi) = if op.subop == VPACKSS {
                (-(1i64 << (8 * narrow - 1)), (1i64 << (8 * narrow - 1)) - 1)
            } else {
                (0, (1i64 << (8 * narrow)) - 1)
            };
            let mut out = Vec::with_capacity(n);
            for lane in 0..n / 16 {
                for src in [a, b] {
                    for v in lanes_of(&src[16 * lane..16 * lane + 16], size) {
                        let clamped = (sext(v, size) as i64).clamp(lo, hi) as u64;
                        out.extend_from_slice(&from_lanes(&[clamped], narrow));
                    }
                }
            }
            out
        }
        VZEXT | VSEXT => {
            let (from, to) = (1usize << (off & 3), 1usize << (off >> 2));
            let lanes = lanes_of(&a[..n * from / to], from);
            let widened: Vec<u64> = lanes
                .iter()
                .map(|v| {
                    if op.subop == VSEXT {
                        sext(*v, from)
                    } else {
                        *v
                    }
                })
                .collect();
            from_lanes(&widened, to)
        }
        VPOPCNT => lanewise(1 << off, &|x, _| x.count_ones() as u64),
        VLZCNT => {
            let size = 1usize << off;
            lanewise(size, &|x, _| {
                (x.leading_zeros() - (64 - 8 * size as u32)) as u64
            })
        }
        VPERMB => (0..n).map(|i| a[b[i] as usize & (n - 1)]).collect(),
        VPERMI2B => (0..n)
            .map(|i| {
                let j = d[i] as usize & (2 * n - 1);
                if j < n {
                    a[j]
                } else {
                    b[j - n]
                }
            })
            .collect(),
        VMULTISHIFT => (0..n)
            .map(|i| {
                let q = u64::from_le_bytes(a[i & !7..(i & !7) + 8].try_into().unwrap());
                q.rotate_right(b[i] as u32 & 63) as u8
            })
            .collect(),
        VGF2P8AFFINE => (0..n)
            .map(|i| {
                let matrix = &b[i & !7..(i & !7) + 8];
                (0..8).fold(0u8, |r, bit| {
                    r | ((((matrix[7 - bit] & a[i]).count_ones() as u8) & 1)
                        ^ ((off >> bit) & 1) as u8)
                        << bit
                })
            })
            .collect(),
        VGF2P8MUL => (0..n).map(|i| gf_mul(a[i], b[i])).collect(),
        VMIN | VMAX => {
            let size = 1usize << (off & 3);
            let key = |v: u64| {
                if off & 4 != 0 {
                    v as i128
                } else {
                    sext(v, size) as i64 as i128
                }
            };
            lanewise(size, &|x, y| {
                if (key(x) < key(y)) == (op.subop == VMIN) {
                    x
                } else {
                    y
                }
            })
        }
        VABS => {
            let size = 1usize << off;
            lanewise(size, &|x, _| (sext(x, size) as i64).unsigned_abs())
        }
        VADDS | VSUBS => {
            let size = 1usize << (off & 3);
            let bits = 8 * size as u32;
            let (lo, hi, key): (i64, i64, &dyn Fn(u64) -> i64) = if off & 4 != 0 {
                (0, (1 << bits) - 1, &|v| v as i64)
            } else {
                (-(1 << (bits - 1)), (1 << (bits - 1)) - 1, &|v| {
                    sext(v, size) as i64
                })
            };
            lanewise(size, &|x, y| {
                let r = if op.subop == VADDS {
                    key(x) + key(y)
                } else {
                    key(x) - key(y)
                };
                r.clamp(lo, hi) as u64
            })
        }
        VAVG => lanewise(1 << off, &|x, y| (x + y + 1) >> 1),
        VADD8 => lanewise(1, &|x, y| x.wrapping_add(y)),
        VSUB8 => lanewise(1, &|x, y| x.wrapping_sub(y)),
        VMULHU16 => lanewise(2, &|x, y| (x * y) >> 16),
        VMULHRS16 => lanewise(2, &|x, y| {
            ((sext(x, 2) as i64 * sext(y, 2) as i64 + 0x4000) >> 15) as u64
        }),
        VMULS32 => lanewise(8, &|x, y| {
            (x as i32 as i64).wrapping_mul(y as i32 as i64) as u64
        }),
        VMADDUBS16 => (0..n / 2)
            .flat_map(|i| {
                let p = |k: usize| a[2 * i + k] as i32 * b[2 * i + k] as i8 as i32;
                ((p(0) + p(1)).clamp(i16::MIN as i32, i16::MAX as i32) as i16).to_le_bytes()
            })
            .collect(),
        VTRUNC => {
            let (from, to) = (1usize << (off & 3), 1usize << (off >> 2));
            let mut out: Vec<u8> = lanes_of(a, from)
                .iter()
                .flat_map(|v| v.to_le_bytes()[..to].to_vec())
                .collect();
            out.resize(n, 0);
            out
        }
        VPERMQ => {
            let q = lanes_of(a, 8);
            from_lanes(
                &(0..q.len())
                    .map(|i| q[(i & !3) + ((off as usize >> (2 * (i % 4))) & 3)])
                    .collect::<Vec<_>>(),
                8,
            )
        }
        VPERMW => {
            let (x, idx) = (lanes_of(a, 2), lanes_of(b, 2));
            from_lanes(
                &idx.iter()
                    .map(|j| x[*j as usize % x.len()])
                    .collect::<Vec<_>>(),
                2,
            )
        }
        VALIGNR => (0..n)
            .map(|i| {
                let (lane, k) = (i / 16, i % 16 + off as usize);
                match k {
                    0..=15 => b[16 * lane + k],
                    16..=31 => a[16 * lane + k - 16],
                    _ => 0,
                }
            })
            .collect(),
        VSHUFLW | VSHUFHW => {
            let w = lanes_of(a, 2);
            let high = op.subop == VSHUFHW;
            from_lanes(
                &(0..w.len())
                    .map(|i| {
                        let (quad, k) = (i / 4, i % 4);
                        if (quad % 2 == 1) == high {
                            w[4 * quad + ((off as usize >> (2 * k)) & 3)]
                        } else {
                            w[i]
                        }
                    })
                    .collect::<Vec<_>>(),
                2,
            )
        }
        VCONFLICT => {
            let size = 1usize << off;
            let x = lanes_of(a, size);
            from_lanes(
                &(0..x.len())
                    .map(|i| {
                        (0..i)
                            .filter(|&j| x[j] == x[i])
                            .fold(0u64, |m, j| m | 1 << j)
                    })
                    .collect::<Vec<_>>(),
                size,
            )
        }
        VDPBUSD => (0..n / 4)
            .flat_map(|i| {
                let acc = u32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as i32;
                (0..4)
                    .fold(acc, |s, k| {
                        s.wrapping_add(a[4 * i + k] as i32 * b[4 * i + k] as i8 as i32)
                    })
                    .to_le_bytes()
            })
            .collect(),
        VDPWSSD => (0..n / 4)
            .flat_map(|i| {
                let acc = u32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as i32;
                let w = |x: &[u8], k: usize| {
                    i16::from_le_bytes([x[4 * i + 2 * k], x[4 * i + 2 * k + 1]]) as i32
                };
                (0..2)
                    .fold(acc, |s, k| s.wrapping_add(w(a, k).wrapping_mul(w(b, k))))
                    .to_le_bytes()
            })
            .collect(),
        VSADBW => (0..n / 8)
            .flat_map(|g| {
                (0..8)
                    .map(|k| (a[8 * g + k] as i64 - b[8 * g + k] as i64).unsigned_abs())
                    .sum::<u64>()
                    .to_le_bytes()
            })
            .collect(),
        VSHLD | VSHRD => {
            let (size, c) = (1usize << (off >> 6), off & 63);
            let bits = 8 * size as u32;
            let mask = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            lanewise(size, &|x, y| {
                if c == 0 {
                    return x;
                }
                if op.subop == VSHLD {
                    (x << c | y >> (bits - c)) & mask
                } else {
                    (x >> c | y << (bits - c)) & mask
                }
            })
        }
        VSHLDV | VSHRDV => {
            let size = 1usize << off;
            let bits = 8 * size as u64;
            let mask = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            let (acc, x, cnt) = (lanes_of(d, size), lanes_of(a, size), lanes_of(b, size));
            from_lanes(
                &(0..acc.len())
                    .map(|i| {
                        let c = cnt[i] % bits;
                        if c == 0 {
                            return acc[i];
                        }
                        if op.subop == VSHLDV {
                            (acc[i] << c | x[i] >> (bits - c)) & mask
                        } else {
                            (acc[i] >> c | x[i] << (bits - c)) & mask
                        }
                    })
                    .collect::<Vec<_>>(),
                size,
            )
        }
        VCLMUL => (0..n / 16)
            .flat_map(|lane| {
                let q = |x: &[u8], k: u32| {
                    u64::from_le_bytes(
                        x[16 * lane + 8 * k as usize..16 * lane + 8 * k as usize + 8]
                            .try_into()
                            .unwrap(),
                    )
                };
                clmul(q(a, off & 1), q(b, (off >> 4) & 1)).to_le_bytes()
            })
            .collect(),
        _ => unreachable!(),
    }
}

/// vpmadd52luq / vpmadd52huq on one half: `d += lo / hi 52 bits of lo52(a) * lo52(b)`
fn madd52(op: &VecInsn, a: &Lanes, b: &Lanes, d: &Lanes) -> Lanes {
    let q = |x: &Lanes, i: usize| x[2 * i] as u64 | (x[2 * i + 1] as u64) << 32;
    let m = (1u64 << 52) - 1;
    let mut out = [0u32; 8];
    for i in 0..4 {
        let p = (q(a, i) & m) as u128 * (q(b, i) & m) as u128;
        let part = if op.subop == VMADD52LO {
            p as u64 & m
        } else {
            (p >> 52) as u64
        };
        let r = q(d, i).wrapping_add(part);
        (out[2 * i], out[2 * i + 1]) = (r as u32, (r >> 32) as u32);
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
#[allow(clippy::needless_range_loop)]
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
    if matches!(op.subop, VMADD52LO | VMADD52HI) {
        return madd52(op, a, b, d);
    }
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
    if matches!(op.subop, VLOADM | VSTOREM) {
        let vm_addr = reg[op.src as usize].wrapping_add(op.off as i64 as u64);
        let slot = if op.subop == VLOADM { op.vd } else { op.va };
        transfer_masked(
            op.subop == VSTOREM,
            slot,
            op.vb as u32,
            op.halves(),
            reg[op.dst as usize],
            vm_addr,
            vector_base,
            mapping,
        )?;
        return Ok(0);
    }
    if matches!(op.subop, VLOAD | VSTORE) {
        let vm_addr = reg[op.src as usize].wrapping_add(op.off as i64 as u64);
        let slot = if op.subop == VLOAD { op.vd } else { op.va };
        transfer(
            op.subop == VSTORE,
            slot,
            op.transfer_slots(),
            vm_addr,
            vector_base,
            mapping,
        )?;
        return Ok(0);
    }
    interpret_registers(op, reg, vector_base)
}

/// [interpret] for every operation but load / store (no guest memory access)
pub(crate) fn interpret_registers(
    op: &VecInsn,
    reg: &[u64],
    vector_base: *mut u8,
) -> Result<u64, EbpfError> {
    if vector_base.is_null() {
        return Err(EbpfError::UnsupportedInstruction);
    }
    if op.subop >= VSLL16 {
        // Byte level reference of the later operations
        let n = ebpf::VECTOR_WIDTH * op.halves();
        let bytes = |s: u8| -> Vec<u8> {
            unsafe {
                std::slice::from_raw_parts(vector_base.add(s as usize * ebpf::VECTOR_WIDTH), n)
            }
            .to_vec()
        };
        if op.subop == VTESTM {
            let size = 1usize << (op.off & 3);
            let (a, b) = (lanes_of(&bytes(op.va), size), lanes_of(&bytes(op.vb), size));
            let negate = op.off & 4 != 0;
            return Ok((0..a.len())
                .filter(|&i| ((a[i] & b[i]) != 0) != negate)
                .fold(0u64, |m, i| m | 1 << i));
        }
        let out = compute_bytes(op, &bytes(op.va), &bytes(op.vb), &bytes(op.vd));
        unsafe {
            std::ptr::copy_nonoverlapping(
                out.as_ptr(),
                vector_base.add(op.vd as usize * ebpf::VECTOR_WIDTH),
                n,
            )
        };
        return Ok(0);
    }
    let slot = |n: u8| unsafe { vector_base.add(n as usize * ebpf::VECTOR_WIDTH) };
    let read = |n: u8| load_lanes(slot(n));
    let scalar = reg[op.src as usize];
    match op.subop {
        VTRANSPOSE8 => {
            let mut rows: [Lanes; 8] = std::array::from_fn(|i| read(op.vd + i as u8));
            transpose8(&mut rows);
            for (i, row) in rows.iter().enumerate() {
                store_lanes(slot(op.vd + i as u8), row);
            }
            return Ok(0);
        }
        VTRANSPOSE64 => {
            // Row r is slot vd + r (4 lanes) or pair vd + 2 r (8 lanes)
            let n = 4 * op.halves();
            let lane =
                |r: usize, c: usize| unsafe { slot(op.vd).add(r * 8 * n + 8 * c).cast::<u64>() };
            let m: Vec<u64> = (0..n * n)
                .map(|i| unsafe { std::ptr::read_unaligned(lane(i / n, i % n)) })
                .collect();
            for r in 0..n {
                for c in 0..n {
                    unsafe { std::ptr::write_unaligned(lane(r, c), m[c * n + r]) };
                }
            }
            return Ok(0);
        }
        VINS64 => {
            unsafe {
                std::ptr::write_unaligned(
                    slot(op.vd).add(8 * op.off as usize).cast::<u64>(),
                    scalar.to_le(),
                )
            };
            return Ok(0);
        }
        VEXT64 => {
            return Ok(u64::from_le(unsafe {
                std::ptr::read_unaligned(slot(op.va).add(8 * op.off as usize).cast::<u64>())
            }));
        }
        VCMP | VBLENDM | VCOMPRESS | VEXPAND => {
            // Lanes of `size` bytes over the operand bytes
            let size = 1usize
                << (if op.subop == VCMP {
                    op.off >> 4
                } else {
                    op.off
                });
            let n = ebpf::VECTOR_WIDTH * op.halves() / size;
            let lane = |s: u8, i: usize| -> u64 {
                let mut v = 0u64;
                for k in 0..size {
                    v |= (unsafe { *slot(s).add(size * i + k) } as u64) << (8 * k);
                }
                v
            };
            let put = |i: usize, v: u64| {
                for k in 0..size {
                    unsafe { *slot(op.vd).add(size * i + k) = (v >> (8 * k)) as u8 };
                }
            };
            let bit = |i: usize| (scalar >> i) & 1 != 0;
            let a: Vec<u64> = (0..n).map(|i| lane(op.va, i)).collect();
            let b: Vec<u64> = (0..n).map(|i| lane(op.vb, i)).collect();
            match op.subop {
                VCMP => {
                    let signed = |x: u64| ((x << (64 - 8 * size)) as i64) >> (64 - 8 * size);
                    let mut mask = 0u64;
                    for i in 0..n {
                        let ord = if op.off & 8 != 0 {
                            a[i].cmp(&b[i])
                        } else {
                            signed(a[i]).cmp(&signed(b[i]))
                        };
                        let t = match op.off & 7 {
                            0 => ord.is_eq(),
                            1 => ord.is_lt(),
                            2 => ord.is_le(),
                            3 => false,
                            4 => ord.is_ne(),
                            5 => ord.is_ge(),
                            6 => ord.is_gt(),
                            _ => true,
                        };
                        mask |= (t as u64) << i;
                    }
                    return Ok(mask);
                }
                VBLENDM => (0..n).for_each(|i| put(i, if bit(i) { a[i] } else { b[i] })),
                VCOMPRESS => {
                    let kept: Vec<u64> = (0..n).filter(|&i| bit(i)).map(|i| a[i]).collect();
                    (0..n).for_each(|i| put(i, kept.get(i).copied().unwrap_or(0)));
                    return Ok(kept.len() as u64);
                }
                _ => {
                    let mut k = 0;
                    for i in 0..n {
                        put(
                            i,
                            if bit(i) {
                                k += 1;
                                a[k - 1]
                            } else {
                                0
                            },
                        );
                    }
                }
            }
            return Ok(0);
        }
        VMONT16 => {
            let w16 = |s: u8, i: usize| unsafe {
                std::ptr::read_unaligned(slot(s).add(2 * i).cast::<i16>())
            } as i32;
            let n = 16 * op.halves();
            let out: Vec<i16> = (0..n)
                .map(|i| {
                    let (a, half) = (w16(op.va, i), (i / 16) as u8);
                    let (w, wp, q) = (
                        w16(op.vb + half, i % 16),
                        w16(op.vb + 2 + half, i % 16),
                        w16(op.off as u8 + half, i % 16),
                    );
                    let hi = |x: i32, y: i32| ((x * y) >> 16) as i16 as i32;
                    let t = ((a * wp) as i16) as i32;
                    (hi(a, w) - hi(t, q)) as i16
                })
                .collect();
            for (i, v) in out.iter().enumerate() {
                unsafe { std::ptr::write_unaligned(slot(op.vd).add(2 * i).cast::<i16>(), *v) };
            }
            return Ok(0);
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
        let out = if matches!(op.subop, VTERNLOG | VMADD52LO | VMADD52HI) {
            compute_half(op, &a, &b, &read(op.vd))
        } else {
            compute(op, &a, &b)
        };
        store_lanes(slot(op.vd), &out);
        return Ok(0);
    }
    // Every source is read before the destination is written
    let reads_d = matches!(
        op.subop,
        VTERNLOG | VPERMI2D | VPERMT2D | VMADD52LO | VMADD52HI
    );
    let a: [Lanes; 2] = std::array::from_fn(|h| {
        if h < halves {
            read(op.va + h as u8)
        } else {
            [0; 8]
        }
    });
    let b: [Lanes; 2] = std::array::from_fn(|h| {
        if h < halves {
            read(op.vb + h as u8)
        } else {
            [0; 8]
        }
    });
    let d: [Lanes; 2] = std::array::from_fn(|h| {
        if h < halves && reads_d {
            read(op.vd + h as u8)
        } else {
            [0; 8]
        }
    });
    let lanes = 8 * halves;
    let get = |x: &[Lanes; 2], i: usize| x[i / 8][i % 8];
    let out: [Lanes; 2] = match op.subop {
        VBCAST32 => [[scalar as u32; 8]; 2],
        VBCAST64 => [std::array::from_fn(|i| (scalar >> (32 * (i % 2))) as u32); 2],
        // Cross lane over all lanes of the operand
        VPERMD => std::array::from_fn(|h| {
            std::array::from_fn(|i| get(&a, (get(&b, 8 * h + i) as usize) & (lanes - 1)))
        }),
        VPERMI2D => std::array::from_fn(|h| {
            std::array::from_fn(|i| {
                let index = get(&d, 8 * h + i) as usize & (2 * lanes - 1);
                if index < lanes {
                    get(&a, index)
                } else {
                    get(&b, index - lanes)
                }
            })
        }),
        VPERMT2D => std::array::from_fn(|h| {
            std::array::from_fn(|i| {
                let index = get(&a, 8 * h + i) as usize & (2 * lanes - 1);
                if index < lanes {
                    get(&d, index)
                } else {
                    get(&b, index - lanes)
                }
            })
        }),
        _ => std::array::from_fn(|h| compute_half(op, &a[h], &b[h], &d[h])),
    };
    for (h, half) in out.iter().enumerate().take(halves) {
        store_lanes(slot(op.vd + h as u8), half);
    }
    Ok(0)
}

/// Mnemonic of a sub-operation (the intrinsic name without `sol_<isa>_`, prefixed with `v`)
pub fn mnemonic(subop: u8, wide: bool) -> String {
    let name = intrinsics::name(subop, wide);
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
        if op.transfer_slots() == 1 {
            format!("v{first}")
        } else {
            format!("v{first}..v{}", first as usize + op.vb as usize)
        }
    };
    let (vd, va, vb) = (op.vd, op.va, op.vb);
    match op.subop {
        VLOAD => format!("{name} {}, {addr}", range(vd)),
        VSTORE => format!("{name} {addr}, {}", range(va)),
        VTRANSPOSE8 => format!("{name} v{vd}..v{}", vd + 7),
        VTRANSPOSE64 => format!(
            "{name} v{vd}..v{}",
            vd as usize + if op.wide { 15 } else { 3 }
        ),
        VBCAST32 | VBCAST64 => format!("{name} v{vd}, r{}", op.src),
        VINS64 => format!("{name} v{vd}, r{}, {}", op.src, op.off),
        VEXT64 => format!("{name} r{}, v{va}, {}", op.dst, op.off),
        VMOVMSK8 => format!("{name} r{}, v{va}", op.dst),
        VCMP => format!("{name} r{}, v{va}, v{vb}, {:#x}", op.dst, op.off),
        VBLENDM => format!("{name} v{vd}, v{va}, v{vb}, r{}, {}", op.src, op.off),
        VCOMPRESS => format!("{name} r{}, v{vd}, v{va}, r{}, {}", op.dst, op.src, op.off),
        VEXPAND => format!("{name} v{vd}, v{va}, r{}, {}", op.src, op.off),
        VSLL64 | VSRL64 | VSRA64 => format!("{name} v{vd}, v{va}, {}", op.off),
        VSLL16 | VSRL16 | VSRA16 | VSRA32 | VROR64 | VPOPCNT | VLZCNT | VZEXT | VSEXT => {
            format!("{name} v{vd}, v{va}, {:#x}", op.off)
        }
        VUNPCKLO | VUNPCKHI | VPACKSS | VPACKUS | VGF2P8AFFINE | VCLMUL => {
            format!("{name} v{vd}, v{va}, v{vb}, {:#x}", op.off)
        }
        VMONT16 => format!("{name} v{vd}, v{va}, v{vb}, q=v{}", op.off),
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
        read_lanes(&self.as_slice()[index * ebpf::VECTOR_WIDTH..])
    }
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

/// AVX-512 extensions beyond F / BW / VL / DQ... which the JIT uses when the host has them, as bits
/// of [crate::vm::Config::simd_disabled_features]
pub mod features {
    /// AVX-512 VBMI2
    pub const VBMI2: u32 = 1 << 0;
    /// AVX-512 DQ
    pub const DQ: u32 = 1 << 1;
    /// AVX-512 IFMA
    pub const IFMA: u32 = 1 << 2;
    /// AVX-512 CD
    pub const CD: u32 = 1 << 3;
    /// AVX-512 VBMI
    pub const VBMI: u32 = 1 << 4;
    /// AVX-512 BITALG
    pub const BITALG: u32 = 1 << 5;
    /// AVX-512 VPOPCNTDQ
    pub const VPOPCNTDQ: u32 = 1 << 6;
    /// GFNI (EVEX forms)
    pub const GFNI: u32 = 1 << 7;
    /// AVX-512 VNNI
    pub const VNNI: u32 = 1 << 8;
    /// VPCLMULQDQ (also the VEX forms)
    pub const VPCLMULQDQ: u32 = 1 << 9;
    /// Every feature
    pub const ALL: u32 = (1 << 10) - 1;
    /// Feature sets of real CPUs with AVX-512 (name, disabled features)
    pub const PROFILES: [(&str, u32); 4] = [
        (
            "skylake-sp",
            VBMI2 | IFMA | VBMI | BITALG | VPOPCNTDQ | GFNI | VNNI | VPCLMULQDQ,
        ),
        (
            "cascade-lake",
            VBMI2 | IFMA | VBMI | BITALG | VPOPCNTDQ | GFNI | VPCLMULQDQ,
        ),
        (
            "cannon-lake",
            VBMI2 | BITALG | VPOPCNTDQ | GFNI | VNNI | VPCLMULQDQ,
        ),
        ("ice-lake / zen 4", 0),
    ];
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
    /// compress / expand, whose lowering may be a scalar loop over up to 64 lanes (code size)
    pub lane_loops: usize,
}

impl SimdAnalysis {
    /// Whether `callx` may not enter instruction `pc`
    pub fn callx_forbidden(&self, pc: usize) -> bool {
        self.callx_forbidden.get(pc).copied().unwrap_or(false)
    }
}

/// Computes [SimdAnalysis]
pub fn analyze<C: crate::vm::ContextObject>(
    executable: &crate::elf::Executable<C>,
) -> SimdAnalysis {
    let sbpf_version = executable.get_sbpf_version();
    let (_, program) = executable.get_text_bytes();
    let len = program.len() / ebpf::INSN_SIZE;
    let intrinsic_hashes: Vec<u32> = intrinsics::hashes()
        .iter()
        .map(|(hash, _, _)| *hash)
        .collect();
    let lane_loop_hashes: Vec<u32> = intrinsics::hashes()
        .iter()
        .filter(|(_, subop, _)| matches!(*subop, VCOMPRESS | VEXPAND))
        .map(|(hash, _, _)| *hash)
        .collect();
    let is_lane_loop = |insn: &ebpf::Insn| {
        if insn.opc == ebpf::VOP {
            VecInsn::decode(insn).is_some_and(|op| matches!(op.subop, VCOMPRESS | VEXPAND))
        } else {
            lane_loop_hashes.contains(&(insn.imm as u32))
        }
    };
    let is_vector = |insn: &ebpf::Insn| {
        insn.opc == ebpf::VOP
            || (insn.opc == ebpf::CALL_IMM
                && insn.src == 0
                && intrinsic_hashes.contains(&(insn.imm as u32)))
    };
    let step = |insn: &ebpf::Insn| {
        if insn.opc == ebpf::LD_DW_IMM && !sbpf_version.disable_lddw() {
            2
        } else {
            1
        }
    };
    let mut jump_targets = vec![false; len + 1];
    let mut vector_instructions = 0;
    let mut lane_loops = 0;
    let mut pc = 0;
    while pc < len {
        let insn = ebpf::get_insn_unchecked(program, pc);
        let class = insn.opc & ebpf::BPF_CLS_MASK;
        if is_vector(&insn) {
            vector_instructions += 1;
            lane_loops += is_lane_loop(&insn) as usize;
        } else if insn.opc == ebpf::CALL_IMM {
            if sbpf_version.static_syscalls() && insn.src == 1 {
                let target = pc as i64 + insn.imm + 1;
                if (0..len as i64).contains(&target) {
                    jump_targets[target as usize] = true;
                }
            } else if let Some((_, target)) = executable
                .get_function_registry()
                .lookup_by_key(insn.imm as u32)
            {
                if target < len {
                    jump_targets[target] = true;
                }
            }
        } else if insn.opc != ebpf::EXIT
            && insn.opc != ebpf::CALL_REG
            && (class == ebpf::BPF_JMP64
                || (class == ebpf::BPF_JMP32 && sbpf_version.enable_jmp32()))
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
    SimdAnalysis {
        jump_targets,
        callx_forbidden,
        vector_instructions,
        lane_loops,
    }
}
