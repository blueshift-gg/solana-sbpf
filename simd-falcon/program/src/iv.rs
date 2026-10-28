//! Vector intrinsics: static syscalls lowered inline by the JIT (SIMD-0645 style).
//!
//! Operands are packed into `r1` (8 bit slot numbers), the scalar operand goes in `r2`. Called
//! with constant slot numbers, every wrapper compiles to `mov64 r1, imm` + `call`.

use solana_define_syscall::define_syscall;

define_syscall!(fn sol_vload256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_vbcast32(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_vtranspose8(ops: u64) -> u64);
define_syscall!(fn sol_vmov256(ops: u64) -> u64);
define_syscall!(fn sol_vadd32x8(ops: u64) -> u64);
define_syscall!(fn sol_vsrl32x8(ops: u64) -> u64);

#[inline(always)]
const fn ops3(d: u8, a: u8, b: u8) -> u64 {
    d as u64 | (a as u64) << 8 | (b as u64) << 16
}

#[inline(always)]
pub fn add(d: u8, a: u8, b: u8) {
    unsafe { sol_vadd32x8(ops3(d, a, b)) };
}
#[inline(always)]
pub fn srl(d: u8, a: u8, n: u8) {
    unsafe { sol_vsrl32x8(ops3(d, a, n)) };
}
#[inline(always)]
pub fn mov(d: u8, a: u8) {
    unsafe { sol_vmov256(ops3(d, a, 0)) };
}
#[inline(always)]
pub fn bcast(d: u8, value: u32) {
    unsafe { sol_vbcast32(d as u64, value as u64) };
}
#[inline(always)]
pub fn transpose(d: u8) {
    unsafe { sol_vtranspose8(d as u64) };
}
#[inline(always)]
pub fn load(d: u8, src: *const u32) {
    unsafe { sol_vload256(d as u64, src as u64) };
}

define_syscall!(fn sol_vxor256(ops: u64) -> u64);
define_syscall!(fn sol_vpermd(ops: u64) -> u64);
define_syscall!(fn sol_vror32x8(ops: u64) -> u64);
define_syscall!(fn sol_vsll32x8(ops: u64) -> u64);
define_syscall!(fn sol_vblendd(ops: u64) -> u64);

#[inline(always)]
pub fn xor(d: u8, a: u8, b: u8) {
    unsafe { sol_vxor256(ops3(d, a, b)) };
}
/// `d[i] = data[idx[i] & 7]`
#[inline(always)]
pub fn permd(d: u8, data: u8, idx: u8) {
    unsafe { sol_vpermd(ops3(d, data, idx)) };
}
#[inline(always)]
pub fn ror(d: u8, a: u8, n: u8) {
    unsafe { sol_vror32x8(ops3(d, a, n)) };
}
#[inline(always)]
pub fn sll(d: u8, a: u8, n: u8) {
    unsafe { sol_vsll32x8(ops3(d, a, n)) };
}
/// Slots below 128 only (7 bit fields)
#[inline(always)]
pub fn blend(d: u8, a: u8, b: u8, mask: u8) {
    unsafe { sol_vblendd(d as u64 | (a as u64) << 7 | (b as u64) << 14 | (mask as u64) << 21) };
}

define_syscall!(fn sol_vrolv32x8(ops: u64) -> u64);
define_syscall!(fn sol_vternlog256(ops: u64) -> u64);

/// `d[i] = a[i].rotate_left(b[i])` for counts in 0..=32
#[inline(always)]
pub fn rolv(d: u8, a: u8, b: u8) {
    unsafe { sol_vrolv32x8(ops3(d, a, b)) };
}
/// `d = f(d, a, b)` bitwise for the truth table `imm` (index `d << 2 | a << 1 | b`), slots below 128
#[inline(always)]
pub fn tern(d: u8, a: u8, b: u8, imm: u8) {
    unsafe { sol_vternlog256(d as u64 | (a as u64) << 7 | (b as u64) << 14 | (imm as u64) << 21) };
}

define_syscall!(fn sol_vadd16x16(ops: u64) -> u64);
define_syscall!(fn sol_vsub16x16(ops: u64) -> u64);
define_syscall!(fn sol_vmullo16x16(ops: u64) -> u64);
define_syscall!(fn sol_vmulhi16x16(ops: u64) -> u64);
define_syscall!(fn sol_vminu16x16(ops: u64) -> u64);
define_syscall!(fn sol_vmadd16x16(ops: u64) -> u64);
define_syscall!(fn sol_vblendw(ops: u64) -> u64);

#[inline(always)]
pub fn add16(d: u8, a: u8, b: u8) {
    unsafe { sol_vadd16x16(ops3(d, a, b)) };
}
#[inline(always)]
pub fn sub16(d: u8, a: u8, b: u8) {
    unsafe { sol_vsub16x16(ops3(d, a, b)) };
}
#[inline(always)]
pub fn mullo16(d: u8, a: u8, b: u8) {
    unsafe { sol_vmullo16x16(ops3(d, a, b)) };
}
/// High half of the signed 16 x 16 bit products
#[inline(always)]
pub fn mulhi16(d: u8, a: u8, b: u8) {
    unsafe { sol_vmulhi16x16(ops3(d, a, b)) };
}
#[inline(always)]
pub fn minu16(d: u8, a: u8, b: u8) {
    unsafe { sol_vminu16x16(ops3(d, a, b)) };
}
/// Signed 16 bit products summed in pairs into u32 lanes
#[inline(always)]
pub fn madd16(d: u8, a: u8, b: u8) {
    unsafe { sol_vmadd16x16(ops3(d, a, b)) };
}
/// Word blend (mask bit i % 8 picks word i from b), slots below 128
#[inline(always)]
pub fn blendw(d: u8, a: u8, b: u8, mask: u8) {
    unsafe { sol_vblendw(d as u64 | (a as u64) << 7 | (b as u64) << 14 | (mask as u64) << 21) };
}
