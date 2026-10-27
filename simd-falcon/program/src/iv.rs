//! Vector intrinsics: static syscalls lowered inline by the JIT (SIMD-0645 style).
//!
//! Operands are packed into `r1` (8 bit slot numbers), the scalar operand goes in `r2`. Called
//! with constant slot numbers, every wrapper compiles to `mov64 r1, imm` + `call`.

#![allow(dead_code)]

use solana_define_syscall::define_syscall;

define_syscall!(fn sol_vload256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_vbcast32(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_vtranspose8(ops: u64) -> u64);
define_syscall!(fn sol_vmov256(ops: u64) -> u64);
define_syscall!(fn sol_vadd32x8(ops: u64) -> u64);
define_syscall!(fn sol_vand256(ops: u64) -> u64);
define_syscall!(fn sol_vsub32x8(ops: u64) -> u64);
define_syscall!(fn sol_vmullo32x8(ops: u64) -> u64);
define_syscall!(fn sol_vminu32x8(ops: u64) -> u64);
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
pub fn sub(d: u8, a: u8, b: u8) {
    unsafe { sol_vsub32x8(ops3(d, a, b)) };
}
#[inline(always)]
pub fn mullo(d: u8, a: u8, b: u8) {
    unsafe { sol_vmullo32x8(ops3(d, a, b)) };
}
#[inline(always)]
pub fn minu(d: u8, a: u8, b: u8) {
    unsafe { sol_vminu32x8(ops3(d, a, b)) };
}
#[inline(always)]
pub fn and(d: u8, a: u8, b: u8) {
    unsafe { sol_vand256(ops3(d, a, b)) };
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
define_syscall!(fn sol_vor256(ops: u64) -> u64);
define_syscall!(fn sol_vandn256(ops: u64) -> u64);
define_syscall!(fn sol_vpermd(ops: u64) -> u64);
define_syscall!(fn sol_vror32x8(ops: u64) -> u64);
define_syscall!(fn sol_vsll32x8(ops: u64) -> u64);
define_syscall!(fn sol_vsllv32x8(ops: u64) -> u64);
define_syscall!(fn sol_vsrlv32x8(ops: u64) -> u64);
define_syscall!(fn sol_vblendd(ops: u64) -> u64);

#[inline(always)]
pub fn xor(d: u8, a: u8, b: u8) {
    unsafe { sol_vxor256(ops3(d, a, b)) };
}
#[inline(always)]
pub fn or(d: u8, a: u8, b: u8) {
    unsafe { sol_vor256(ops3(d, a, b)) };
}
/// `!a & b`
#[inline(always)]
pub fn andn(d: u8, a: u8, b: u8) {
    unsafe { sol_vandn256(ops3(d, a, b)) };
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
#[inline(always)]
pub fn sllv(d: u8, a: u8, b: u8) {
    unsafe { sol_vsllv32x8(ops3(d, a, b)) };
}
#[inline(always)]
pub fn srlv(d: u8, a: u8, b: u8) {
    unsafe { sol_vsrlv32x8(ops3(d, a, b)) };
}
/// Slots below 128 only (7 bit fields)
#[inline(always)]
pub fn blend(d: u8, a: u8, b: u8, mask: u8) {
    unsafe { sol_vblendd(d as u64 | (a as u64) << 7 | (b as u64) << 14 | (mask as u64) << 21) };
}
