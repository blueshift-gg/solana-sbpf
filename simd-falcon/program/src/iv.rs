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
define_syscall!(fn sol_avx2_bcast32x8(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_avx2_transpose8x8(ops: u64) -> u64);
define_syscall!(fn sol_avx_mov256(ops: u64) -> u64);
define_syscall!(fn sol_avx2_add32x8(ops: u64) -> u64);
define_syscall!(fn sol_avx2_srl32x8(ops: u64) -> u64);
define_syscall!(fn sol_avx2_sll32x8(ops: u64) -> u64);
define_syscall!(fn sol_avx2_xor256(ops: u64) -> u64);
define_syscall!(fn sol_avx2_permd256(ops: u64) -> u64);
define_syscall!(fn sol_avx2_blendd256(ops: u64) -> u64);
define_syscall!(fn sol_avx512_ror32x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_rolv32x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_ternlog256(ops: u64) -> u64);
define_syscall!(fn sol_avx2_add16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_sub16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_mullo16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_mulhi16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_minu16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_madd16x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_blendw256(ops: u64) -> u64);
define_syscall!(fn sol_avx512_add16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_sub16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_mullo16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_mulhi16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_minu16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_madd16x32(ops: u64) -> u64);
define_syscall!(fn sol_avx512_add32x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_bcast32x16(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_avx512_permi2d512(ops: u64) -> u64);

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
pub fn add(d: u8, a: u8, b: u8) {
    unsafe { sol_avx2_add32x8(ops3(d, a, b)) };
}
#[inline(always)]
pub fn srl(d: u8, a: u8, n: u8) {
    unsafe { sol_avx2_srl32x8(ops3(d, a, n)) };
}
#[inline(always)]
pub fn sll(d: u8, a: u8, n: u8) {
    unsafe { sol_avx2_sll32x8(ops3(d, a, n)) };
}
#[inline(always)]
pub fn mov(d: u8, a: u8) {
    unsafe { sol_avx_mov256(ops3(d, a, 0)) };
}
#[inline(always)]
pub fn bcast(d: u8, value: u32) {
    unsafe { sol_avx2_bcast32x8(d as u64, value as u64) };
}
#[inline(always)]
pub fn transpose(d: u8) {
    unsafe { sol_avx2_transpose8x8(d as u64) };
}
#[inline(always)]
pub fn xor(d: u8, a: u8, b: u8) {
    unsafe { sol_avx2_xor256(ops3(d, a, b)) };
}
/// `d[i] = data[idx[i] & 7]`
#[inline(always)]
pub fn permd(d: u8, data: u8, idx: u8) {
    unsafe { sol_avx2_permd256(ops3(d, data, idx)) };
}
#[inline(always)]
pub fn ror(d: u8, a: u8, n: u8) {
    unsafe { sol_avx512_ror32x8(ops3(d, a, n)) };
}
/// Slots below 128 only (7 bit fields)
#[inline(always)]
pub fn blend(d: u8, a: u8, b: u8, mask: u8) {
    unsafe { sol_avx2_blendd256(ops7(d, a, b, mask)) };
}
/// `d[i] = a[i].rotate_left(b[i])` for counts in 0..=32
#[inline(always)]
pub fn rolv(d: u8, a: u8, b: u8) {
    unsafe { sol_avx512_rolv32x8(ops3(d, a, b)) };
}
/// `d = f(d, a, b)` bitwise for the truth table `imm` (index `d << 2 | a << 1 | b`), slots below 128
#[inline(always)]
pub fn tern(d: u8, a: u8, b: u8, imm: u8) {
    unsafe { sol_avx512_ternlog256(ops7(d, a, b, imm)) };
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
    mullo16 = sol_avx2_mullo16x16 / sol_avx512_mullo16x32;
    // High half of the signed 16 x 16 bit products
    mulhi16 = sol_avx2_mulhi16x16 / sol_avx512_mulhi16x32;
    minu16 = sol_avx2_minu16x16 / sol_avx512_minu16x32;
    // Signed 16 bit products summed in pairs into u32 lanes
    madd16 = sol_avx2_madd16x16 / sol_avx512_madd16x32;
    add32 = sol_avx2_add32x8 / sol_avx512_add32x16;
}
/// Word blend (mask bit i % 8 picks word i from b), slots below 128
#[inline(always)]
pub fn blendw(d: u8, a: u8, b: u8, mask: u8) {
    unsafe { sol_avx2_blendw256(ops7(d, a, b, mask)) };
}
/// Broadcast to a slot pair
#[inline(always)]
pub fn bcast512(d: u8, value: u32) {
    unsafe { sol_avx512_bcast32x16(d as u64, value as u64) };
}
/// `d[i] = (a ++ b)[d[i] & 31]` over slot pairs (two table permutation)
#[inline(always)]
pub fn permi2d512(d: u8, a: u8, b: u8) {
    unsafe { sol_avx512_permi2d512(ops3(d, a, b)) };
}
