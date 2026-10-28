//! Vector intrinsics: static syscalls lowered inline by the JIT (SIMD-0645 style).
//!
//! Operands are packed into `r1` (8 bit slot numbers), the scalar operand goes in `r2`, a scalar
//! result comes back in `r0`. Called with constant slot numbers, every wrapper compiles to
//! `mov64 r1, imm` + `call`. The prefix of a name is the ISA level which has the operation as one
//! instruction (`sol_avx_`, `sol_avx2_`, `sol_avx512_`); the 512 bit forms work on slot pairs
//! (even slot numbers).

use solana_define_syscall::define_syscall;

#[inline(always)]
const fn ops3(d: u8, a: u8, b: u8) -> u64 {
    d as u64 | (a as u64) << 8 | (b as u64) << 16
}

/// Three operands with 7 bit slots and an imm8 (blend, ternlog)
#[inline(always)]
const fn ops7(d: u8, a: u8, b: u8, imm: u8) -> u64 {
    d as u64 | (a as u64) << 7 | (b as u64) << 14 | (imm as u64) << 21
}

define_syscall!(fn sol_avx_load256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx_store256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx2_transpose8x8(ops: u64) -> u64);
define_syscall!(fn sol_avx2_add32x8(ops: u64) -> u64);
define_syscall!(fn sol_avx2_add16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_sub16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_minu16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_madd16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_add16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_sub16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_minu16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_madd16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_add32x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_bcast32x16(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_avx512_xor512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_ternlog512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_permd512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_mov512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_rolv64x4(ops: u64) -> u64);
define_syscall!(fn sol_avx512_rolv64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_transpose8x8q(ops: u64) -> u64);
define_syscall!(fn sol_avx512_blendw512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_ror32x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_shufb512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_cmp512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_compress512(ops: u64, mask: u64) -> u64);
define_syscall!(fn sol_avx2_mont16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_permt2d512(ops: u64) -> u64);
define_syscall!(fn sol_avx_ins64(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_avx512_mont16x32(ops: u64) -> u64);

/// `n` consecutive slots from `src` (one translation)
#[inline(always)]
pub fn load(d: u8, n: u8, src: *const u32) {
    unsafe { sol_avx_load256(d as u64 | ((n - 1) as u64) << 24, src as u64) };
}
/// `n` consecutive slots to `dst` (one translation)
#[inline(always)]
pub fn store(dst: *mut u32, a: u8, n: u8) {
    unsafe { sol_avx_store256(a as u64 | ((n - 1) as u64) << 24, dst as u64) };
}
#[inline(always)]
pub fn transpose(d: u8) {
    unsafe { sol_avx2_transpose8x8(d as u64) };
}

macro_rules! words16 {
    ($($name:ident = $narrow:ident / $wide:ident;)*) => {
        $(
            /// 16 bit lanes, 512 bit form (slot pairs) if `wide`
            #[inline(always)]
            pub fn $name(wide: bool, d: u8, a: u8, b: u8) {
                if wide {
                    unsafe { $wide(ops3(d, a, b)) };
                } else {
                    unsafe { $narrow(ops3(d, a, b)) };
                }
            }
        )*
    };
}
words16! {
    add16 = sol_avx2_add16x16 / sol_avx512_add16x32;
    sub16 = sol_avx2_sub16x16 / sol_avx512_sub16x32;
    minu16 = sol_avx2_minu16x16 / sol_avx512_minu16x32;
    // Signed 16 bit products summed in pairs into u32 lanes
    madd16 = sol_avx2_madd16x16 / sol_avx512_madd16x32;
    add32 = sol_avx2_add32x8 / sol_avx512_add32x16;
}
/// Broadcast to a slot pair
#[inline(always)]
pub fn bcast512(d: u8, value: u32) {
    unsafe { sol_avx512_bcast32x16(d as u64, value as u64) };
}

/// 512 bit forms on slot pairs (even slots)
#[inline(always)]
pub fn xor512(d: u8, a: u8, b: u8) {
    unsafe { sol_avx512_xor512(ops3(d, a, b)) };
}
/// `d = f(d, a, b)` bitwise (see [tern]), slots below 128
#[inline(always)]
pub fn tern512(d: u8, a: u8, b: u8, imm: u8) {
    unsafe { sol_avx512_ternlog512(ops7(d, a, b, imm)) };
}
/// `d[i] = data[idx[i] & 15]` over the 16 u32 lanes of the pairs
#[inline(always)]
pub fn permd512(d: u8, data: u8, idx: u8) {
    unsafe { sol_avx512_permd512(ops3(d, data, idx)) };
}
#[inline(always)]
pub fn mov512(d: u8, a: u8) {
    unsafe { sol_avx512_mov512(ops3(d, a, 0)) };
}
/// u64 lanes: `d = a.rotate_left(b % 64)`, 512 bit form if `wide`
#[inline(always)]
pub fn rolv64(wide: bool, d: u8, a: u8, b: u8) {
    if wide {
        unsafe { sol_avx512_rolv64x8(ops3(d, a, b)) };
    } else {
        unsafe { sol_avx512_rolv64x4(ops3(d, a, b)) };
    }
}
/// Transposes the 8 x 8 u64 matrix of the pairs d, d + 2, .., d + 14
#[inline(always)]
pub fn transpose8x8q(d: u8) {
    unsafe { sol_avx512_transpose8x8q(d as u64) };
}
/// Word blend on pairs (mask bit i % 8 picks word i from b), slots below 128
#[inline(always)]
pub fn blendw512(d: u8, a: u8, b: u8, mask: u8) {
    unsafe { sol_avx512_blendw512(ops7(d, a, b, mask)) };
}
#[inline(always)]
pub fn ror512(d: u8, a: u8, n: u8) {
    unsafe { sol_avx512_ror32x16(ops3(d, a, n)) };
}
/// Byte shuffle within each 128 bit lane of the pairs (`idx` bytes, bit 7 zeroes)
#[inline(always)]
pub fn shufb512(d: u8, a: u8, idx: u8) {
    unsafe { sol_avx512_shufb512(ops3(d, a, idx)) };
}
/// Lane mask of `pred(a, b)` over the pairs; `imm` = predicate | 8 (unsigned) | lane size << 4
#[inline(always)]
pub fn cmp512(a: u8, b: u8, imm: u8) -> u64 {
    unsafe { sol_avx512_cmp512(a as u64 | (b as u64) << 8 | (imm as u64) << 16) }
}
/// The lanes of `a` (of `8 << size` bits) whose mask bit is set, packed into the low lanes of `d`
/// (rest zero); returns their number
#[inline(always)]
pub fn compress512(d: u8, a: u8, size: u8, mask: u64) -> u64 {
    unsafe { sol_avx512_compress512(d as u64 | (a as u64) << 8 | (size as u64) << 16, mask) }
}
/// Montgomery product of i16 lanes: `d = mulhi(a, w) - mulhi(mullo(a, w'), q)`, `w'` in slot `w + 2`,
/// `q` in slot `q` (slots below 128), 512 bit form if `wide`
#[inline(always)]
pub fn mont16(wide: bool, d: u8, a: u8, w: u8, q: u8) {
    let ops = d as u64 | (a as u64) << 7 | (w as u64) << 14 | (q as u64) << 21;
    if wide {
        unsafe { sol_avx512_mont16x32(ops) };
    } else {
        unsafe { sol_avx2_mont16x16(ops) };
    }
}
/// `d[i] = (d ++ b)[idx[i] & 31]` over the 16 u32 lanes of the pairs (the first table is overwritten)
#[inline(always)]
pub fn permt2d512(d: u8, idx: u8, b: u8) {
    unsafe { sol_avx512_permt2d512(ops3(d, idx, b)) };
}
/// u64 lane `lane` of slot `d` = `value`
#[inline(always)]
pub fn ins64(d: u8, value: u64, lane: u8) {
    unsafe { sol_avx_ins64(d as u64 | (lane as u64) << 8, value) };
}
