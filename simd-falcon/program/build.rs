//! Generates the vectorized part of the Falcon-512 verifier.
//!
//! Firedancer's algorithm (fd_falcon.c, branch drubin/falcon-syscall): a Cooley-Tukey forward NTT
//! and a Gentleman-Sande inverse NTT over Z/12289, a pointwise product with the (here compile time)
//! NTT of the public key, and the squared norm of (c - s2 * h, s2). Adapted to the vector
//! intrinsics the way current AVX2 NTTs (Kyber, Dilithium) are written:
//! - 16 signed 16 bit coefficients per vector, Montgomery products with R = 2^16 from 16 bit
//!   multiplies: `mulhi(a, w) - mulhi(mullo(a, w * q^-1), q)`, 4 ops for 16 products.
//! - Lazy reduction: the generator tracks a bound of every slot and reduces (a Montgomery product
//!   by R mod q) only where the next butterfly could overflow 16 bits.
//! - The last forward (first inverse) levels: levels 8, 4 and 2 between whole vectors after an
//!   8 x 8 transpose of u32 lanes (pairs of coefficients), level 1 between the two halves of a
//!   u32 with a 16 bit rotation and a word blend.
//!
//! The vector code must address slots with compile time constants for the JIT to lower each
//! intrinsic call inline, so all of it is emitted as straight-line code. The generated op stream is
//! simulated here and checked against schoolbook polynomial arithmetic before the program is built.

use std::fmt::Write;

const Q: u32 = 12289;
const N: usize = 512;
const QI: i64 = Q as i64;

// Slots of the vector file (16 coefficients each). Everything but P and C is a slot pair, for the
// 512 bit operations (256 bit operations use the low slot).
const P: u8 = 0; // working polynomial (s2, NTT, product, m), 32 slots
const C: u8 = 64; // hash_to_point candidates (u16 < 5Q), 32 slots
const QV: u8 = 96; // q in every word
const Q2V: u8 = 98;
const Q4V: u8 = 100;
const ONE: u8 = 102; // R mod q (Montgomery one) and its q^-1 multiple
const ONE_P: u8 = 104;
const TW: u8 = 106; // twiddle and its q^-1 multiple
const TW_P: u8 = 108;
const T0: u8 = 110;
const T1: u8 = 112;
const T2: u8 = 114;
const ACC1: u8 = 116; // ||s1||^2 partial sums (u32 lanes)
const ACC2: u8 = 118; // ||s2||^2
const I16_MAX: i64 = i16::MAX as i64;

fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    r
}

fn bitrev9(x: usize) -> usize {
    (0..9).fold(0, |r, i| r | (((x >> i) & 1) << (8 - i)))
}

/// Firedancer's twiddles: bit-reversed powers of psi (a primitive 2N-th root of unity) and psi^-1
fn twiddles() -> ([u32; N], [u32; N]) {
    let exp = ((Q - 1) / (2 * N as u32)) as u64;
    let psi = (1..Q as u64).map(|i| pow_mod(i, exp, Q as u64)).find(|g| pow_mod(*g, N as u64, Q as u64) != 1).unwrap();
    let psi_inv = pow_mod(psi, Q as u64 - 2, Q as u64);
    let powers = |x: u64| {
        let mut out = [1u32; N];
        let mut p = 1u64;
        for i in 1..N {
            p = p * x % Q as u64;
            out[bitrev9(i)] = p as u32;
        }
        out
    };
    (powers(psi), powers(psi_inv))
}

/// Reference forward NTT, Firedancer's loop structure with exact reduction
fn ntt_ref(a: &mut [u32; N], psi_pos: &[u32; N]) {
    let (mut t, mut m) = (N, 1);
    while m < N {
        t >>= 1;
        for i in 0..m {
            let s = psi_pos[m + i] as u64;
            for j in 2 * i * t..2 * i * t + t {
                let u = a[j];
                let v = (a[j + t] as u64 * s % Q as u64) as u32;
                a[j] = (u + v) % Q;
                a[j + t] = (u + Q - v) % Q;
            }
        }
        m <<= 1;
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Add(u8, u8, u8),
    Srl(u8, u8, u8),
    Mov(u8, u8),
    Bcast(u8, u32),
    Transpose(u8),
    Xor(u8, u8, u8),
    /// (dst, data, indices)
    Permd(u8, u8, u8),
    Ror(u8, u8, u8),
    Sll(u8, u8, u8),
    Blend(u8, u8, u8, u8),
    Rolv(u8, u8, u8),
    /// d = f(d, a, b) for the truth table imm
    Tern(u8, u8, u8, u8),
    Add16(u8, u8, u8),
    Sub16(u8, u8, u8),
    Mullo16(u8, u8, u8),
    Mulhi16(u8, u8, u8),
    Minu16(u8, u8, u8),
    Madd16(u8, u8, u8),
    Blendw(u8, u8, u8, u8),
    /// Load consecutive generated table rows: (first slot, table, first row, rows)
    Load(u8, &'static str, usize, u8),
    /// u64 lanes: d = a.rotate_left(b % 64)
    Rolv64(u8, u8, u8),
    /// 512 bit form of a lane-wise op on the slot pairs (all slots even)
    W(&'static Op),
    /// 512 bit cross lane permutation of the 16 u32 lanes of the pairs: (dst, data, indices)
    PermdW(u8, u8, u8),
    /// Transposes the 8 x 8 u64 matrix of the 8 pairs from the slot
    TransposeQW(u8),
}

/// The 512 bit form of a lane-wise op
fn w(op: Op) -> Op {
    Op::W(Box::leak(Box::new(op)))
}

/// A lane-wise op moved to the high halves of its pairs
fn shifted(op: Op, h: u8) -> Op {
    use Op::*;
    match op {
        Add(d, a, b) => Add(d + h, a + h, b + h),
        Xor(d, a, b) => Xor(d + h, a + h, b + h),
        Add16(d, a, b) => Add16(d + h, a + h, b + h),
        Sub16(d, a, b) => Sub16(d + h, a + h, b + h),
        Mullo16(d, a, b) => Mullo16(d + h, a + h, b + h),
        Mulhi16(d, a, b) => Mulhi16(d + h, a + h, b + h),
        Minu16(d, a, b) => Minu16(d + h, a + h, b + h),
        Madd16(d, a, b) => Madd16(d + h, a + h, b + h),
        Rolv64(d, a, b) => Rolv64(d + h, a + h, b + h),
        Tern(d, a, b, imm) => Tern(d + h, a + h, b + h, imm),
        Blendw(d, a, b, m) => Blendw(d + h, a + h, b + h, m),
        Ror(d, a, n) => Ror(d + h, a + h, n),
        Mov(d, a) => Mov(d + h, a + h),
        Bcast(d, v) => Bcast(d + h, v),
        _ => panic!("{op:?} has no 512 bit form here"),
    }
}


type Tables = std::collections::BTreeMap<&'static str, Vec<[u32; 8]>>;

/// q^-1 mod 2^16
fn qinv() -> i64 {
    (1..1i64 << 16).find(|x| (x * QI) & 0xffff == 1).unwrap()
}

/// Montgomery form of `x` (x * 2^16 mod q) centered in (-q/2, q/2]
fn mont_centered(x: i64) -> i64 {
    let m = (x.rem_euclid(QI) << 16) % QI;
    if m > QI / 2 { m - QI } else { m }
}

/// The q^-1 multiple of a Montgomery constant, as the i16 bit pattern
fn mont_prime(w: i64) -> i64 {
    ((w * qinv()) & 0xffff) as i16 as i64
}

fn pair(lo: i64, hi: i64) -> u32 {
    (lo as u16 as u32) | (hi as u16 as u32) << 16
}

/// Bound of |mont(a, w)| for |a| <= a_bound: |a w| / 2^16 + 1 (floor) + 2^15 q / 2^16 (rounded up)
fn mont_bound(a_bound: i64, w_abs: i64) -> i64 {
    assert!(a_bound <= I16_MAX && w_abs <= QI / 2 + 1);
    a_bound * w_abs / 65536 + 1 + 6145
}

/// Coefficient index -> (slot, word) after the transposes (layout of the last levels)
fn transposed_pos(j: usize) -> (usize, usize) {
    let (t, r, c, h) = (j / 128, (j / 16) % 8, (j / 2) % 8, j % 2);
    (8 * t + c, 2 * r + h)
}

struct Gen {
    ops: Vec<Op>,
    bound: [i64; 256],
    /// (ops executed, slot, bound) for the simulation to check
    checks: Vec<(usize, u8, i64)>,
    tables: std::collections::BTreeMap<&'static str, Vec<[u32; 8]>>,
}

/// Width of the generated operations: 256 bit (one slot) or 512 bit (a pair of slots)
#[derive(Clone, Copy, PartialEq, Eq)]
enum Width {
    Y,
    Z,
}
use Width::{Y, Z};

impl Gen {
    fn op(&mut self, op: Op, width: Width) {
        self.ops.push(if width == Z { w(op) } else { op });
    }

    /// Slots of an operand
    fn slots(s: u8, width: Width) -> std::ops::Range<u8> {
        s..s + if width == Z { 2 } else { 1 }
    }

    fn bound(&self, s: u8, width: Width) -> i64 {
        Self::slots(s, width).map(|x| self.bound[x as usize]).max().unwrap()
    }

    fn set_bound(&mut self, s: u8, bound: i64, width: Width) {
        for x in Self::slots(s, width) {
            self.bound[x as usize] = bound;
            self.checks.push((self.ops.len(), x, bound));
        }
    }

    /// Loads table rows (appended to `table`) into consecutive slots from `slot`
    fn rows(&mut self, slot: u8, table: &'static str, new_rows: &[[u32; 8]]) {
        let rows = self.tables.entry(table).or_default();
        let index = rows.len();
        rows.extend_from_slice(new_rows);
        self.ops.push(Op::Load(slot, table, index, new_rows.len() as u8));
    }

    /// Twiddle rows (TW, TW_P = TW + 2) for per word Montgomery constants: one row of 16 words per
    /// slot of the operation (bound returned)
    fn twiddle_rows(&mut self, table: &'static str, w: &[[i64; 16]]) -> i64 {
        let row = |w: &[i64; 16], f: &dyn Fn(i64) -> i64| -> [u32; 8] { std::array::from_fn(|i| pair(f(w[2 * i]), f(w[2 * i + 1]))) };
        let mut rows: Vec<[u32; 8]> = w.iter().map(|w| row(w, &|x| x)).collect();
        if w.len() == 1 {
            rows.push([0; 8]); // TW + 1 is unused by 256 bit operations
        }
        rows.extend(w.iter().map(|w| row(w, &mont_prime)));
        assert_eq!(TW_P, TW + 2);
        self.rows(TW, table, &rows);
        w.iter().flatten().map(|x| x.abs()).max().unwrap()
    }

    /// Broadcast twiddle (bound returned)
    fn twiddle_bcast(&mut self, w: i64, width: Width) -> i64 {
        self.op(Op::Bcast(TW, pair(w, w)), width);
        self.op(Op::Bcast(TW_P, pair(mont_prime(w), mont_prime(w))), width);
        w.abs()
    }

    /// dst = mont(a, W) (dst may be a)
    fn mont(&mut self, dst: u8, a: u8, w: u8, w_p: u8, w_abs: i64, width: Width) {
        self.op(Op::Mullo16(T0, a, w_p), width);
        self.op(Op::Mulhi16(T1, a, w), width);
        self.op(Op::Mulhi16(T0, T0, QV), width);
        self.op(Op::Sub16(dst, T1, T0), width);
        let b = mont_bound(self.bound(a, width), w_abs);
        self.set_bound(dst, b, width);
    }

    /// Transposes a tile of 8 slots: every slot may now hold values of any slot of the tile
    fn transpose(&mut self, tile: u8) {
        self.ops.push(Op::Transpose(tile));
        let max = (tile..tile + 8).map(|s| self.bound[s as usize]).max().unwrap();
        for s in tile..tile + 8 {
            self.set_bound(s, max, Y);
        }
    }

    fn reduce(&mut self, x: u8, width: Width) {
        let one = mont_centered(1).abs();
        self.mont(x, x, ONE, ONE_P, one, width);
    }

    fn reduce_if(&mut self, x: u8, limit: i64, width: Width) {
        if self.bound(x, width) > limit {
            self.reduce(x, width);
        }
    }

    /// Cooley-Tukey: (a, b) = (a + b w, a - b w), twiddle in TW / TW_P
    fn ct(&mut self, a: u8, b: u8, w_abs: i64, width: Width) {
        let bw = mont_bound(self.bound(b, width), w_abs);
        self.reduce_if(a, I16_MAX - bw, width);
        assert!(self.bound(a, width) + bw <= I16_MAX);
        self.mont(T2, b, TW, TW_P, w_abs, width);
        self.op(Op::Sub16(b, a, T2), width);
        self.op(Op::Add16(a, a, T2), width);
        let out = self.bound(a, width) + bw;
        self.set_bound(a, out, width);
        self.set_bound(b, out, width);
    }

    /// Gentleman-Sande: (a, b) = (a + b, (a - b) w), twiddle in TW / TW_P
    fn gs(&mut self, a: u8, b: u8, w_abs: i64, width: Width) {
        if self.bound(a, width) + self.bound(b, width) > I16_MAX {
            let (big, small) = if self.bound(a, width) >= self.bound(b, width) { (a, b) } else { (b, a) };
            self.reduce(big, width);
            if self.bound(a, width) + self.bound(b, width) > I16_MAX {
                self.reduce(small, width);
            }
        }
        let sum = self.bound(a, width) + self.bound(b, width);
        assert!(sum <= I16_MAX);
        self.op(Op::Sub16(T2, a, b), width);
        self.op(Op::Add16(a, a, b), width);
        self.set_bound(T2, sum, width);
        self.mont(b, T2, TW, TW_P, w_abs, width);
        self.set_bound(a, sum, width);
    }

    /// Level 1 on the slot (pair) `x` (pairs inside a u32): forward (a, b) = (a + b w, a - b w),
    /// inverse (a, b) = (a + b, (a - b) w); `w[s]` are the twiddles of slot x + s
    fn level1(&mut self, x: u8, w: &[[i64; 8]], forward: bool, width: Width) {
        let one = mont_centered(1);
        // Low words get the Montgomery one (a reduction), high words the twiddle
        let words: Vec<[i64; 16]> = w.iter().map(|w| std::array::from_fn(|i| if i % 2 == 0 { one } else { w[i / 2] })).collect();
        let w_abs = self.twiddle_rows(if forward { "TWF1" } else { "TWI1" }, &words);
        if forward {
            // M = (a', b w), Ms = (b w, a'), out = (M + Ms, Ms - M) blended
            self.mont(x, x, TW, TW_P, w_abs, width);
            let m = self.bound(x, width);
            self.op(Op::Ror(T2, x, 16), width);
            self.op(Op::Sub16(T1, T2, x), width);
            self.op(Op::Add16(x, x, T2), width);
            self.op(Op::Blendw(x, x, T1, 0xaa), width);
            assert!(2 * m <= I16_MAX);
            self.set_bound(x, 2 * m, width);
        } else {
            // Xs = (b, a), S = X + Xs = (a + b, a + b), D = Xs - X = (b - a, a - b),
            // out = mont((S, D) blended, (one, w))
            self.reduce_if(x, I16_MAX / 2, width);
            let b = self.bound(x, width);
            self.op(Op::Ror(T2, x, 16), width);
            self.op(Op::Sub16(T1, T2, x), width);
            self.op(Op::Add16(x, x, T2), width);
            self.op(Op::Blendw(x, x, T1, 0xaa), width);
            self.set_bound(x, 2 * b, width);
            self.mont(x, x, TW, TW_P, w_abs, width);
        }
    }
}

/// The vector core: ||s2||^2, NTT(s2), product with the prepared pubkey (table H_PREP), inverse
/// NTT, s1 = c - s2 h, ||s1||^2. Returns the op stream and the generated tables (H_PREP rows are
/// placeholders, filled by [prepare]). Most of it runs on slot pairs (512 bit operations); the
/// butterflies between the two slots of a pair (levels 16 and, in the transposed layout, 2) and
/// the transposes use 256 bit operations.
fn generate(psi_pos: &[u32; N], psi_neg: &[u32; N]) -> (Vec<Op>, Tables, Vec<(usize, u8, i64)>) {
    let mut g = Gen { ops: Vec::new(), bound: [0; 256], checks: Vec::new(), tables: Default::default() };
    let one = mont_centered(1);
    for (slot, value) in [(QV, QI), (Q2V, 2 * QI), (Q4V, 4 * QI), (ONE, one), (ONE_P, mont_prime(one))] {
        g.op(Op::Bcast(slot, pair(value, value)), Z);
    }
    // ||s2||^2 from the decoded (signed, |s2| < 2048) coefficients
    for s in (0..32).step_by(2) {
        g.op(Op::Madd16(T0, P + s, P + s), Z);
        g.op(Op::Add(ACC2, ACC2, T0), Z);
        g.set_bound(P + s, 2047, Z);
    }
    let twiddle = |psi: &[u32; N], t: usize, i: usize| mont_centered(psi[N / (2 * t) + i] as i64);
    // Forward NTT, levels 256..16 between whole slots
    let mut t = 256;
    while t >= 16 {
        let ts = (t / 16) as u8;
        let width = if ts >= 2 { Z } else { Y };
        for i in 0..N / (2 * t) {
            let w_abs = g.twiddle_bcast(twiddle(psi_pos, t, i), width);
            let base = (2 * i * t / 16) as u8;
            for k in (0..ts).step_by(if width == Z { 2 } else { 1 }) {
                g.ct(P + base + k, P + base + ts + k, w_abs, width);
            }
        }
        t /= 2;
    }
    for tile in 0..4 {
        g.transpose(P + 8 * tile);
    }
    // Levels 8, 4, 2: slot c and c + t / 2 of a tile, block index from (tile, lane, group)
    let tw_words = |psi: &[u32; N], t: usize, tile: usize, c: usize| -> [i64; 16] {
        std::array::from_fn(|word| {
            let (r, h) = (word / 2, word % 2);
            let j = 128 * tile + 16 * r + 2 * c + h;
            twiddle(psi, t, j / (2 * t))
        })
    };
    let transposed_level = |g: &mut Gen, psi: &[u32; N], t: usize, table: &'static str, forward: bool| {
        let d = t / 2; // slot distance
        let width = if d >= 2 { Z } else { Y };
        for tile in 0..4 {
            for grp in 0..4 / d {
                let c0 = grp * 2 * d;
                // The twiddles only depend on the group (the slot c enters as 2 c / (2 t) = 0)
                let words: Vec<[i64; 16]> = vec![tw_words(psi, t, tile, c0); if width == Z { 2 } else { 1 }];
                let w_abs = g.twiddle_rows(table, &words);
                for c in (c0..c0 + d).step_by(if width == Z { 2 } else { 1 }) {
                    let (a, b) = (P + 8 * tile as u8 + c as u8, P + 8 * tile as u8 + (c + d) as u8);
                    if forward { g.ct(a, b, w_abs, width) } else { g.gs(a, b, w_abs, width) }
                }
            }
        }
    };
    for t in [8usize, 4, 2] {
        transposed_level(&mut g, psi_pos, t, "TWF", true);
    }
    let level1_words = |psi: &[u32; N], slot: usize| -> [i64; 8] {
        let (tile, c) = (slot / 8, slot % 8);
        std::array::from_fn(|r| twiddle(psi, 1, (128 * tile + 16 * r + 2 * c) / 2))
    };
    for slot in (0..32u8).step_by(2) {
        let w = [level1_words(psi_pos, slot as usize), level1_words(psi_pos, slot as usize + 1)];
        g.level1(P + slot, &w, true, Z);
    }
    // Pointwise product with the prepared pubkey (NTT(h) / N, Montgomery form), rows of a pair:
    // H (2 slots), H q^-1 (2 slots)
    for s in (0..32u8).step_by(2) {
        g.rows(TW, "H_PREP", &[[0; 8]; 4]);
        g.mont(P + s, P + s, TW, TW_P, QI / 2 + 1, Z);
    }
    // Inverse NTT: level 1, then 2, 4, 8 in the transposed layout, then 16..256
    for slot in (0..32u8).step_by(2) {
        let w = [level1_words(psi_neg, slot as usize), level1_words(psi_neg, slot as usize + 1)];
        g.level1(P + slot, &w, false, Z);
    }
    for t in [2usize, 4, 8] {
        transposed_level(&mut g, psi_neg, t, "TWI", false);
    }
    for tile in 0..4 {
        g.transpose(P + 8 * tile);
    }
    let mut t = 16;
    while t <= 256 {
        let ts = (t / 16) as u8;
        let width = if ts >= 2 { Z } else { Y };
        for i in 0..N / (2 * t) {
            let w_abs = g.twiddle_bcast(twiddle(psi_neg, t, i), width);
            let base = (2 * i * t / 16) as u8;
            for k in (0..ts).step_by(if width == Z { 2 } else { 1 }) {
                g.gs(P + base + k, P + base + ts + k, w_abs, width);
            }
        }
        t *= 2;
    }
    // s1 = c - m: c (u16 < 5q) to [0, 2q), u = c + q - m in [0, 4q) (m reduced below q), u to
    // [0, q), |s1| = min(u, q - u), squares summed in pairs into u32 lanes
    for s in (0..32u8).step_by(2) {
        let (cs, m) = (C + s, P + s);
        g.reduce_if(m, QI, Z);
        assert!(g.bound(m, Z) <= QI);
        for op in [
            Op::Sub16(T2, cs, Q4V),
            Op::Minu16(T2, cs, T2), // [0, 4q) (5q > 4q values wrapped below q)
            Op::Sub16(T1, T2, Q2V),
            Op::Minu16(T2, T2, T1), // [0, 2q)
            Op::Add16(T2, T2, QV),
            Op::Sub16(T2, T2, m), // [0, 4q)
            Op::Sub16(T1, T2, Q2V),
            Op::Minu16(T2, T2, T1), // [0, 2q)
            Op::Sub16(T1, T2, QV),
            Op::Minu16(T2, T2, T1), // [0, q)
            Op::Sub16(T1, QV, T2),
            Op::Minu16(T2, T2, T1), // [0, q / 2]
            Op::Madd16(T1, T2, T2),
            Op::Add(ACC1, ACC1, T1),
        ] {
            g.op(op, Z);
        }
    }
    (g.ops, g.tables, g.checks)
}

/// Executes the op stream on a model of the vector region
fn simulate(ops: &[Op], region: &mut [[u32; 8]; 256], tables: &dyn Fn(&str, usize) -> [u32; 8]) {
    for op in ops {
        let lanes = |r: &[[u32; 8]; 256], a: u8, b: u8, f: &dyn Fn(u32, u32) -> u32| -> [u32; 8] {
            std::array::from_fn(|i| f(r[a as usize][i], r[b as usize][i]))
        };
        let words16 = |r: &[[u32; 8]; 256], a: u8, b: u8, f: &dyn Fn(u16, u16) -> u16| -> [u32; 8] {
            std::array::from_fn(|i| {
                let (x, y) = (r[a as usize][i], r[b as usize][i]);
                f(x as u16, y as u16) as u32 | (f((x >> 16) as u16, (y >> 16) as u16) as u32) << 16
            })
        };
        match *op {
            Op::Add16(d, a, b) => region[d as usize] = words16(region, a, b, &|x, y| x.wrapping_add(y)),
            Op::Sub16(d, a, b) => region[d as usize] = words16(region, a, b, &|x, y| x.wrapping_sub(y)),
            Op::Mullo16(d, a, b) => region[d as usize] = words16(region, a, b, &|x, y| x.wrapping_mul(y)),
            Op::Mulhi16(d, a, b) => region[d as usize] = words16(region, a, b, &|x, y| ((x as i16 as i32 * y as i16 as i32) >> 16) as u16),
            Op::Minu16(d, a, b) => region[d as usize] = words16(region, a, b, &|x, y| x.min(y)),
            Op::Madd16(d, a, b) => {
                let (x, y) = (region[a as usize], region[b as usize]);
                let p = |v: u32, w: u32, sh: u32| (v >> sh) as u16 as i16 as i32 * (w >> sh) as u16 as i16 as i32;
                region[d as usize] = std::array::from_fn(|i| p(x[i], y[i], 0).wrapping_add(p(x[i], y[i], 16)) as u32);
            }
            Op::Blendw(d, a, b, m) => {
                let (x, y) = (region[a as usize], region[b as usize]);
                region[d as usize] = std::array::from_fn(|i| {
                    let pick = |h: usize| if (m >> ((2 * i + h) % 8)) & 1 != 0 { y[i] } else { x[i] };
                    (pick(0) & 0xffff) | (pick(1) & 0xffff_0000)
                });
            }
            Op::Add(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x.wrapping_add(y)),
            Op::Srl(d, a, n) => region[d as usize] = region[a as usize].map(|x| x >> n),
            Op::Mov(d, a) => region[d as usize] = region[a as usize],
            Op::Bcast(d, v) => region[d as usize] = [v; 8],
            Op::Transpose(d) => {
                let d = d as usize;
                let tile: [[u32; 8]; 8] = std::array::from_fn(|r| region[d + r]);
                for r in 0..8 {
                    region[d + r] = std::array::from_fn(|c| tile[c][r]);
                }
            }
            Op::Rolv64(d, a, b) => {
                let q = |x: &[u32; 8], i: usize| x[2 * i] as u64 | (x[2 * i + 1] as u64) << 32;
                let (x, y) = (region[a as usize], region[b as usize]);
                let mut out = [0u32; 8];
                for i in 0..4 {
                    let r = q(&x, i).rotate_left((q(&y, i) % 64) as u32);
                    (out[2 * i], out[2 * i + 1]) = (r as u32, (r >> 32) as u32);
                }
                region[d as usize] = out;
            }
            Op::W(inner) => {
                for h in 0..2 {
                    simulate(&[shifted(*inner, h)], region, tables);
                }
            }
            Op::PermdW(d, a, idx) => {
                let get = |r: &[[u32; 8]; 256], s: u8, i: usize| r[s as usize + i / 8][i % 8];
                let out: [u32; 16] = std::array::from_fn(|i| get(region, a, (get(region, idx, i) & 15) as usize));
                for i in 0..16 {
                    region[d as usize + i / 8][i % 8] = out[i];
                }
            }
            Op::TransposeQW(d) => {
                let q = |r: &[[u32; 8]; 256], row: usize, c: usize| {
                    let s = &r[d as usize + 2 * row + c / 4];
                    s[2 * (c % 4)] as u64 | (s[2 * (c % 4) + 1] as u64) << 32
                };
                let m: Vec<u64> = (0..64).map(|i| q(region, i / 8, i % 8)).collect();
                for row in 0..8 {
                    for c in 0..8 {
                        let v = m[8 * c + row];
                        let s = &mut region[d as usize + 2 * row + c / 4];
                        (s[2 * (c % 4)], s[2 * (c % 4) + 1]) = (v as u32, (v >> 32) as u32);
                    }
                }
            }
            Op::Load(d, table, row, n) => {
                for i in 0..n as usize {
                    region[d as usize + i] = tables(table, row + i);
                }
            }
            Op::Xor(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x ^ y),
            Op::Permd(d, a, idx) => {
                let (data, idx) = (region[a as usize], region[idx as usize]);
                region[d as usize] = std::array::from_fn(|i| data[(idx[i] & 7) as usize]);
            }
            Op::Ror(d, a, n) => region[d as usize] = region[a as usize].map(|x| x.rotate_right(n as u32)),
            Op::Sll(d, a, n) => region[d as usize] = region[a as usize].map(|x| x << n),
            Op::Blend(d, a, b, m) => {
                let (x, y) = (region[a as usize], region[b as usize]);
                region[d as usize] = std::array::from_fn(|i| if (m >> i) & 1 != 0 { y[i] } else { x[i] });
            }
            Op::Rolv(d, a, b) => {
                let shl = |x: u32, n: u32| if n > 31 { 0 } else { x << n };
                let shr = |x: u32, n: u32| if n > 31 { 0 } else { x >> n };
                region[d as usize] = lanes(region, a, b, &|x, n| shl(x, n) | shr(x, 32u32.wrapping_sub(n)));
            }
            Op::Tern(d, a, b, imm) => {
                let pick = |x: u32, set: bool| if set { x } else { !x };
                let (dv, av, bv) = (region[d as usize], region[a as usize], region[b as usize]);
                region[d as usize] = std::array::from_fn(|i| {
                    (0..8u32).filter(|k| (imm >> k) & 1 != 0).fold(0, |r, k| {
                        r | (pick(dv[i], k & 4 != 0) & pick(av[i], k & 2 != 0) & pick(bv[i], k & 1 != 0))
                    })
                });
            }
        }
    }
}

fn decode_pubkey(pk: &[u8]) -> [u32; N] {
    assert_eq!(pk.len(), 897);
    assert_eq!(pk[0], 0x09);
    let mut h = [0u32; N];
    let (mut acc, mut bits, mut idx) = (0u64, 0, 0);
    for byte in &pk[1..] {
        acc = acc << 8 | *byte as u64;
        bits += 8;
        if bits >= 14 {
            bits -= 14;
            h[idx] = ((acc >> bits) & 0x3fff) as u32;
            assert!(h[idx] < Q);
            idx += 1;
        }
    }
    h
}


/// NTT(h) / N in Montgomery form, laid out like the forward NTT output
fn prepare(h: &[u32; N], psi_pos: &[u32; N]) -> Vec<[u32; 8]> {
    let mut hn = *h;
    ntt_ref(&mut hn, psi_pos);
    let n_inv = pow_mod(N as u64, Q as u64 - 2, Q as u64) as i64;
    let mut words = [[0i64; 16]; 32];
    for j in 0..N {
        let (slot, word) = transposed_pos(j);
        words[slot][word] = mont_centered(hn[j] as i64 * n_inv);
    }
    // Per slot pair: H of both slots, then H q^-1 of both slots
    let mut rows = Vec::new();
    for w in words.chunks(2) {
        for f in [&|x: i64| x, &mont_prime as &dyn Fn(i64) -> i64] {
            for w in w {
                rows.push(std::array::from_fn(|i| pair(f(w[2 * i]), f(w[2 * i + 1]))));
            }
        }
    }
    rows
}

fn self_test(ops: &[Op], tables: &Tables, checks: &[(usize, u8, i64)], psi_pos: &[u32; N]) {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rand = |m: u32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as u32
    };
    for round in 0..6 {
        let h: [u32; N] = std::array::from_fn(|_| rand(Q));
        // Signed s2 with |s2| < 2048: random, sparse, and the extremes
        let s2: [i64; N] = std::array::from_fn(|i| match round {
            0 if i > 3 => 0,
            1 => 2047,
            2 => if i % 2 == 0 { -2047 } else { 2047 },
            _ => rand(4095) as i64 - 2047,
        });
        let c: [u32; N] = std::array::from_fn(|i| if round == 1 { 5 * Q - 1 - (i as u32 % 3) } else { rand(5 * Q) });
        let h_prep = prepare(&h, psi_pos);
        let mut region = [[0u32; 8]; 256];
        let mut put = |slot: u8, i: usize, v: u16| {
            let lane = &mut region[slot as usize + i / 16][(i % 16) / 2];
            *lane = if i % 2 == 0 { (*lane & 0xffff_0000) | v as u32 } else { (*lane & 0xffff) | (v as u32) << 16 };
        };
        for i in 0..N {
            put(P, i, s2[i] as i16 as u16);
            put(C, i, c[i] as u16);
        }
        let lookup = |name: &str, row: usize| if name == "H_PREP" { h_prep[row] } else { tables[name][row] };
        // Signed 16 bit overflow check of the NTT part (the final stage wraps on purpose)
        let first_c = |op: &Op| matches!(*op, Op::Sub16(_, a, _) if a == C);
        let ntt_end = ops.iter().position(|op| first_c(op) || matches!(*op, Op::W(inner) if first_c(inner))).unwrap();
        for (k, op) in ops[..ntt_end].iter().enumerate() {
            let halves = match *op {
                Op::W(inner) => vec![shifted(*inner, 0), shifted(*inner, 1)],
                other => vec![other],
            };
            for half in halves {
                if let Op::Add16(_, a, b) | Op::Sub16(_, a, b) = half {
                    let sub = matches!(half, Op::Sub16(..));
                    for lane in 0..8 {
                        for h in 0..2 {
                            let x = (region[a as usize][lane] >> (16 * h)) as u16 as i16 as i32;
                            let y = (region[b as usize][lane] >> (16 * h)) as u16 as i16 as i32;
                            let r = if sub { x - y } else { x + y };
                            assert!(r == r as i16 as i32, "overflow at op {k} {op:?} round {round}: {x} {y}");
                        }
                    }
                }
            }
            simulate(std::slice::from_ref(op), &mut region, &lookup);
            for &(_, slot, bound) in checks.iter().filter(|c| c.0 == k + 1) {
                for lane in 0..8 {
                    for h in 0..2 {
                        let x = (region[slot as usize][lane] >> (16 * h)) as u16 as i16 as i64;
                        assert!(x.abs() <= bound, "slot {slot} = {x} exceeds bound {bound} after op {k} {op:?} round {round}");
                    }
                }
            }
        }
        simulate(&ops[ntt_end..], &mut region, &lookup);
        // Schoolbook s2 * h mod (x^N + 1, Q)
        let mut prod = [0i64; N];
        for i in 0..N {
            for j in 0..N {
                let v = s2[i] * h[j] as i64;
                if i + j < N { prod[i + j] += v } else { prod[i + j - N] -= v }
            }
        }
        let centered = |x: i64| {
            let x = x.rem_euclid(QI);
            if x > QI / 2 { x - QI } else { x }
        };
        let norm1: i64 = (0..N).map(|i| centered(c[i] as i64 - prod[i]).pow(2)).sum();
        let norm2: i64 = (0..N).map(|i| s2[i].pow(2)).sum();
        let got1: i64 = region[ACC1 as usize..ACC1 as usize + 2].iter().flatten().map(|x| *x as i64).sum();
        let got2: i64 = region[ACC2 as usize..ACC2 as usize + 2].iter().flatten().map(|x| *x as i64).sum();
        let word = |slot: u8, i: usize| (region[slot as usize + i / 16][(i % 16) / 2] >> (16 * (i % 2))) as u16 as i16 as i64;
        let bad: Vec<usize> = (0..N).filter(|&i| (word(P, i) - prod[i]).rem_euclid(QI) != 0).collect();
        if !bad.is_empty() {
            panic!("m mismatch round {round}: {} bad, first {:?}", bad.len(), &bad[..bad.len().min(20)]);
        }
        assert_eq!(got2, norm2, "s2 norm mismatch (round {round})");
        assert_eq!(got1, norm1, "s1 norm mismatch (round {round})");
    }
}

fn emit_table(s: &mut String, name: &str, rows: &[[u32; 8]]) {
    writeln!(s, "#[allow(non_camel_case_types)] #[repr(align(32))] pub struct {name}_ROWS(pub [[u32; 8]; {}]);", rows.len()).unwrap();
    writeln!(s, "pub static {name}: {name}_ROWS = {name}_ROWS([").unwrap();
    for r in rows {
        writeln!(s, "    {:?},", r).unwrap();
    }
    writeln!(s, "]);").unwrap();
}

fn emit_fn(s: &mut String, name: &str, ops: &[Op]) {
    writeln!(s, "/// {} vector intrinsic calls", ops.len()).unwrap();
    writeln!(s, "#[inline(never)]\npub fn {name}() {{\n    use crate::iv::*;").unwrap();
    for op in ops {
        match *op {
            Op::Add(d, a, b) => writeln!(s, "    add({d}, {a}, {b});"),
            Op::Srl(d, a, n) => writeln!(s, "    srl({d}, {a}, {n});"),
            Op::Mov(d, a) => writeln!(s, "    mov({d}, {a});"),
            Op::Bcast(d, v) => writeln!(s, "    bcast({d}, {v});"),
            Op::Transpose(d) => writeln!(s, "    transpose({d});"),
            Op::Load(d, t, row, n) => writeln!(s, "    load({d}, {n}, {t}.0[{row}].as_ptr());"),
            Op::Rolv64(d, a, b) => writeln!(s, "    rolv64(false, {d}, {a}, {b});"),
            Op::PermdW(d, a, i) => writeln!(s, "    permd512({d}, {a}, {i});"),
            Op::TransposeQW(d) => writeln!(s, "    transpose8x8q({d});"),
            Op::W(inner) => match *inner {
                Op::Xor(d, a, b) => writeln!(s, "    xor512({d}, {a}, {b});"),
                Op::Tern(d, a, b, imm) => writeln!(s, "    tern512({d}, {a}, {b}, {imm});"),
                Op::Rolv64(d, a, b) => writeln!(s, "    rolv64(true, {d}, {a}, {b});"),
                Op::Mov(d, a) => writeln!(s, "    mov512({d}, {a});"),
                Op::Add(d, a, b) => writeln!(s, "    add32(true, {d}, {a}, {b});"),
                Op::Add16(d, a, b) => writeln!(s, "    add16(true, {d}, {a}, {b});"),
                Op::Sub16(d, a, b) => writeln!(s, "    sub16(true, {d}, {a}, {b});"),
                Op::Mullo16(d, a, b) => writeln!(s, "    mullo16(true, {d}, {a}, {b});"),
                Op::Mulhi16(d, a, b) => writeln!(s, "    mulhi16(true, {d}, {a}, {b});"),
                Op::Minu16(d, a, b) => writeln!(s, "    minu16(true, {d}, {a}, {b});"),
                Op::Madd16(d, a, b) => writeln!(s, "    madd16(true, {d}, {a}, {b});"),
                Op::Blendw(d, a, b, m) => writeln!(s, "    blendw512({d}, {a}, {b}, {m});"),
                Op::Ror(d, a, n) => writeln!(s, "    ror512({d}, {a}, {n});"),
                Op::Bcast(d, v) => writeln!(s, "    bcast512({d}, {v});"),
                other => panic!("no 512 bit emission for {other:?}"),
            },
            Op::Xor(d, a, b) => writeln!(s, "    xor({d}, {a}, {b});"),
            Op::Permd(d, a, i) => writeln!(s, "    permd({d}, {a}, {i});"),
            Op::Ror(d, a, n) => writeln!(s, "    ror({d}, {a}, {n});"),
            Op::Sll(d, a, n) => writeln!(s, "    sll({d}, {a}, {n});"),
            Op::Blend(d, a, b, m) => writeln!(s, "    blend({d}, {a}, {b}, {m});"),
            Op::Rolv(d, a, b) => writeln!(s, "    rolv({d}, {a}, {b});"),
            Op::Add16(d, a, b) => writeln!(s, "    add16(false, {d}, {a}, {b});"),
            Op::Sub16(d, a, b) => writeln!(s, "    sub16(false, {d}, {a}, {b});"),
            Op::Mullo16(d, a, b) => writeln!(s, "    mullo16(false, {d}, {a}, {b});"),
            Op::Mulhi16(d, a, b) => writeln!(s, "    mulhi16(false, {d}, {a}, {b});"),
            Op::Minu16(d, a, b) => writeln!(s, "    minu16(false, {d}, {a}, {b});"),
            Op::Madd16(d, a, b) => writeln!(s, "    madd16(false, {d}, {a}, {b});"),
            Op::Blendw(d, a, b, m) => writeln!(s, "    blendw({d}, {a}, {b}, {m});"),
            Op::Tern(d, a, b, imm) => writeln!(s, "    tern({d}, {a}, {b}, {imm});"),
        }
        .unwrap();
    }
    writeln!(s, "}}").unwrap();
}

// ---------------------------------------------------------------------------------------------
// Keccak-f[1600] on the vector intrinsics, 512 bit operations on 64 bit lanes
//
// Every row of the state is one slot pair (u64 lane x of pair y = A[x, y]), in 8 pairs so that
// the 8 x 8 transpose applies; rows 5..8 and lanes 5..8 only ever hold values that never flow into
// rows / lanes 0..5. A round:
//   R layout (pair y = row y):    theta (column parity, two lane permutations, one rotation,
//                                  a three input xor per row), rho (one variable rotation per row)
//   pi:                           one lane permutation per pair turns the R into the C layout
//   C layout (pair x = column x): chi in place (three input logic), iota (one xor)
//   transpose:                    back to the R layout
// 30 operations per round. The rate rows go to and from guest memory with one load / store.

mod shake {
    #![allow(dead_code)]
    include!("src/shake.rs");
}

const K_STATE: u8 = 0; // 8 pairs
const K_C: u8 = 16;
const K_T0: u8 = 18;
const K_T1: u8 = 20;
const K_S0: u8 = 22;
const K_S1: u8 = 24;
const K_IN: u8 = 26; // absorb buffer, rate rows 0..4
const K_TABLE: u8 = 34; // first slot of the constant table (KC)
const K_IDX_M1: u8 = 34;
const K_IDX_P1: u8 = 36;
const K_ONE: u8 = 38;
const K_RHO: u8 = 40; // 5 pairs
const K_PI: u8 = 50; // 5 pairs
const K_RC: u8 = 60; // 24 pairs
/// d ^ a ^ b
const XOR3: u8 = 0x96;
/// d ^ (!a & b)
const CHI: u8 = 0xd2;

fn keccak_rho_offsets() -> [[u32; 5]; 5] {
    let mut r = [[0u32; 5]; 5];
    let (mut x, mut y) = (1usize, 0usize);
    for t in 0..24u32 {
        r[x][y] = ((t + 1) * (t + 2) / 2) % 64;
        (x, y) = (y, (2 * x + 3 * y) % 5);
    }
    r
}

fn keccak_round_constants() -> [u64; 24] {
    let mut rc = [0u64; 24];
    let mut lfsr = 1u8;
    for round in rc.iter_mut() {
        for j in 0..7 {
            if lfsr & 1 != 0 {
                *round ^= 1u64 << ((1 << j) - 1);
            }
            lfsr = if lfsr & 0x80 != 0 { (lfsr << 1) ^ 0x71 } else { lfsr << 1 };
        }
    }
    rc
}

fn keccak_f_ref(a: &mut [u64; 25]) {
    let r = keccak_rho_offsets();
    for rc in keccak_round_constants() {
        let c: [u64; 5] = std::array::from_fn(|x| (0..5).fold(0, |acc, y| acc ^ a[x + 5 * y]));
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(r[x][y]);
            }
        }
        for x in 0..5 {
            for y in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }
        a[0] ^= rc;
    }
}

/// The two slots (rows) of a pair holding the u64 lanes `lanes`
fn pair_rows(lanes: [u64; 8]) -> [[u32; 8]; 2] {
    std::array::from_fn(|h| std::array::from_fn(|i| (lanes[4 * h + i / 2] >> (32 * (i % 2))) as u32))
}

/// u32 indices of a permutation of u64 lanes: lane j <- lane src(j)
fn lane_permutation(src: &dyn Fn(usize) -> usize) -> [u64; 8] {
    std::array::from_fn(|j| {
        let s = src(j) as u64;
        (2 * s) | (2 * s + 1) << 32
    })
}

/// Rows of the Keccak constant table, loaded into the slots K_TABLE.. by keccak_init
fn keccak_table() -> Vec<[u32; 8]> {
    let mut pairs: Vec<[u64; 8]> = Vec::new();
    let cyclic = |j: usize, k: usize| if j < 5 { (j + k) % 5 } else { j };
    pairs.push(lane_permutation(&|j| cyclic(j, 4)));
    pairs.push(lane_permutation(&|j| cyclic(j, 1)));
    pairs.push([1; 8]);
    let r = keccak_rho_offsets();
    for y in 0..5 {
        pairs.push(std::array::from_fn(|x| if x < 5 { r[x][y] as u64 } else { 0 }));
    }
    for big_x in 0..5 {
        pairs.push(lane_permutation(&|y| if y < 5 { (big_x + 3 * y) % 5 } else { y }));
    }
    for rc in keccak_round_constants() {
        pairs.push(std::array::from_fn(|i| if i == 0 { rc } else { 0 }));
    }
    assert_eq!(K_TABLE as usize + 2 * pairs.len(), K_RC as usize + 48);
    pairs.into_iter().flat_map(pair_rows).collect()
}

fn keccak_init_ops() -> Vec<Op> {
    let rows = keccak_table().len();
    (0..rows).step_by(32).map(|row| Op::Load(K_TABLE + row as u8, "KC", row, (rows - row).min(32) as u8)).collect()
}

fn keccak_absorb_ops() -> Vec<Op> {
    (0..4).map(|y| w(Op::Xor(K_STATE + 2 * y, K_STATE + 2 * y, K_IN + 2 * y))).collect()
}

fn keccak_f_ops() -> Vec<Op> {
    let row = |y: u8| K_STATE + 2 * y;
    let mut ops = Vec::new();
    for round in 0..24u8 {
        // theta: C = column parities, D[x] = C[x - 1] ^ rotl(C[x + 1], 1), A[x, y] ^= D[x]
        ops.push(w(Op::Xor(K_C, row(0), row(1))));
        ops.push(w(Op::Tern(K_C, row(2), row(3), XOR3)));
        ops.push(w(Op::Xor(K_C, K_C, row(4))));
        ops.push(Op::PermdW(K_T0, K_C, K_IDX_M1));
        ops.push(Op::PermdW(K_T1, K_C, K_IDX_P1));
        ops.push(w(Op::Rolv64(K_T1, K_T1, K_ONE)));
        for y in 0..5 {
            ops.push(w(Op::Tern(row(y), K_T0, K_T1, XOR3)));
        }
        // rho
        for y in 0..5 {
            ops.push(w(Op::Rolv64(row(y), row(y), K_RHO + 2 * y)));
        }
        // pi: pair X lane Y <- lane (X + 3 Y) % 5, now pair x = column x
        for x in 0..5 {
            ops.push(Op::PermdW(row(x), row(x), K_PI + 2 * x));
        }
        // chi in place: columns 0 and 1 are needed after they are overwritten
        ops.push(w(Op::Mov(K_S0, row(0))));
        ops.push(w(Op::Mov(K_S1, row(1))));
        for x in 0..5u8 {
            let col = |c: u8| if x + c < 5 { row(x + c) } else if x + c == 5 { K_S0 } else { K_S1 };
            ops.push(w(Op::Tern(row(x), col(1), col(2), CHI)));
        }
        // iota
        ops.push(w(Op::Xor(row(0), row(0), K_RC + 2 * round)));
        // back to the R layout
        ops.push(Op::TransposeQW(K_STATE));
    }
    ops
}

/// Byte offsets of the 68 samples of a squeezed block in the stored rate rows (pairs 0..4), in
/// stream order: sample 4 L + j is bytes 2 j..2 j + 2 of lane L = 5 y + x, at byte 64 y + 8 x
fn sample_offsets() -> Vec<usize> {
    (0..shake::RATE_LANES * 4)
        .map(|i| {
            let (lane, j) = (i / 4, i % 4);
            64 * (lane / 5) + 8 * (lane % 5) + 2 * j
        })
        .collect()
}

fn emit_take(s: &mut String) {
    // A cold call keeps the rare rejection out of line, so the common path is a not taken branch
    // instead of a select
    writeln!(s, "#[cold]\n#[inline(never)]\nfn rejected(p: *mut u16) -> *mut u16 {{\n    unsafe {{ p.sub(1) }}\n}}").unwrap();
    writeln!(s, "/// Appends the accepted samples of a squeezed block (`b`: the stored rate rows) at `p` (see").unwrap();
    writeln!(s, "/// shake::Sponge::take)").unwrap();
    writeln!(s, "#[inline(always)]\npub unsafe fn take_candidates(b: *const u8, mut p: *mut u16) -> *mut u16 {{").unwrap();
    for off in sample_offsets() {
        writeln!(s, "    let v = u16::from_be((b.add({off}) as *const u16).read_volatile()); p.write_volatile(v); p = p.add(1); if v >= {} {{ p = rejected(p); }}", 5 * Q).unwrap();
    }
    writeln!(s, "    p\n}}").unwrap();
}

/// u64 lane `x` of pair `pair` in the region model
fn lane_mut(region: &mut [[u32; 8]; 256], pair: u8, x: usize) -> (&mut u32, &mut u32) {
    let (lo, hi) = region[pair as usize + x / 4].split_at_mut(2 * (x % 4) + 1);
    (&mut lo[2 * (x % 4)], &mut hi[0])
}

fn lane_get(region: &[[u32; 8]; 256], pair: u8, x: usize) -> u64 {
    let s = &region[pair as usize + x / 4];
    s[2 * (x % 4)] as u64 | (s[2 * (x % 4) + 1] as u64) << 32
}

fn lane_set(region: &mut [[u32; 8]; 256], pair: u8, x: usize, v: u64) {
    let (lo, hi) = lane_mut(region, pair, x);
    (*lo, *hi) = (v as u32, (v >> 32) as u32);
}

struct SimSponge<'a> {
    region: [[u32; 8]; 256],
    f: &'a [Op],
    absorb: &'a [Op],
    table: Vec<[u32; 8]>,
    out: Vec<u32>,
}

impl SimSponge<'_> {
    fn run(&mut self, ops: &[Op]) {
        let table = self.table.clone();
        simulate(ops, &mut self.region, &|_, row| table[row]);
    }
}

impl shake::Sponge for SimSponge<'_> {
    fn set_input(&mut self, lane: usize, value: u64) {
        lane_set(&mut self.region, K_IN + 2 * (lane / 5) as u8, lane % 5, value);
    }
    fn absorb_permute(&mut self) {
        let (a, f) = (self.absorb, self.f);
        self.run(a);
        self.run(f);
    }
    fn permute(&mut self) {
        let f = self.f;
        self.run(f);
    }
    fn squeeze(&mut self) {}
    fn take(&mut self, n: usize) -> usize {
        assert_eq!(n, self.out.len());
        let bytes: Vec<u8> = self.region[K_STATE as usize..K_STATE as usize + 8].iter().flat_map(|slot| slot.iter().flat_map(|w| w.to_le_bytes())).collect();
        for off in sample_offsets() {
            let v = u16::from_be_bytes([bytes[off], bytes[off + 1]]) as u32;
            if v < 5 * Q {
                self.out.push(v);
            }
        }
        self.out.len()
    }
}

fn shake256_ref(data: &[u8], out_len: usize) -> Vec<u8> {
    let mut a = [0u64; 25];
    let mut padded = data.to_vec();
    padded.push(0x1f);
    while padded.len() % 136 != 0 {
        padded.push(0);
    }
    *padded.last_mut().unwrap() |= 0x80;
    for block in padded.chunks(136) {
        for i in 0..17 {
            a[i] ^= u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
        }
        keccak_f_ref(&mut a);
    }
    let mut out = Vec::new();
    loop {
        for lane in &a[..17] {
            out.extend_from_slice(&lane.to_le_bytes());
        }
        if out.len() >= out_len {
            out.truncate(out_len);
            return out;
        }
        keccak_f_ref(&mut a);
    }
}

fn keccak_self_test(f: &[Op], absorb: &[Op], init: &[Op]) {
    assert_eq!(&shake256_ref(b"", 8), &[0x46, 0xb9, 0xdd, 0x2b, 0x0b, 0xa8, 0x8d, 0x13], "reference SHAKE256");
    let table = keccak_table();
    // The permutation alone, with garbage in the unused rows and lanes
    let mut region = [[0u32; 8]; 256];
    for (i, slot) in region[..16].iter_mut().enumerate() {
        *slot = std::array::from_fn(|l| (i as u32 * 8 + l as u32).wrapping_mul(0x9e37_79b9));
    }
    simulate(init, &mut region, &|_, row| table[row]);
    let mut a: [u64; 25] = std::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0x0123_4567_89ab_cdef).rotate_left(i as u32 * 7));
    for i in 0..25 {
        lane_set(&mut region, K_STATE + 2 * (i / 5) as u8, i % 5, a[i]);
    }
    simulate(f, &mut region, &|_, row| table[row]);
    keccak_f_ref(&mut a);
    for i in 0..25 {
        assert_eq!(lane_get(&region, K_STATE + 2 * (i / 5) as u8, i % 5), a[i], "keccak lane {i}");
    }
    // hash_to_point through the driver against the reference
    for msg_len in [0usize, 1, 7, 95, 96, 135, 136, 200, 271, 272, 1000] {
        let nonce: Vec<u8> = (0..40u8).map(|i| i.wrapping_mul(37)).collect();
        let msg: Vec<u8> = (0..msg_len).map(|i| (i as u8).wrapping_mul(13) ^ 0x5a).collect();
        let mut sponge = SimSponge { region: [[0; 8]; 256], f, absorb, table: table.clone(), out: Vec::new() };
        sponge.run(init);
        shake::hash_to_point(&mut sponge, &nonce, &msg);
        let stream = shake256_ref(&[nonce.clone(), msg].concat(), 8 * 136 * 4);
        let expected: Vec<u32> = stream.chunks(2).map(|p| (p[0] as u32) << 8 | p[1] as u32).filter(|w| *w < 5 * Q).take(N).collect();
        assert_eq!(sponge.out[..N], expected[..], "hash_to_point, message length {msg_len}");
    }
}

fn main() {
    let pk_path = std::env::var("FALCON_PK").unwrap_or_else(|_| "../fixtures/falcon.pk".to_string());
    println!("cargo:rerun-if-changed={pk_path}");
    println!("cargo:rerun-if-changed=build.rs");
    assert_eq!(ACC2, ACC1 + 2);
    let (psi_pos, psi_neg) = twiddles();
    let (ops, mut tables, checks) = generate(&psi_pos, &psi_neg);
    self_test(&ops, &tables, &checks, &psi_pos);
    let h = decode_pubkey(&std::fs::read(&pk_path).expect("pubkey fixture"));
    tables.insert("H_PREP", prepare(&h, &psi_pos));
    let (k_init, k_absorb, k_f) = (keccak_init_ops(), keccak_absorb_ops(), keccak_f_ops());
    keccak_self_test(&k_f, &k_absorb, &k_init);
    let mut s = String::new();
    for (name, rows) in &tables {
        emit_table(&mut s, name, rows);
    }
    emit_table(&mut s, "KC", &keccak_table());
    writeln!(s, "pub const ACC1: u8 = {ACC1};\npub const C_SLOT: u8 = {C};\npub const P_SLOT: u8 = {P};").unwrap();
    writeln!(s, "pub const K_IN: u8 = {K_IN};\npub const K_STATE: u8 = {K_STATE};").unwrap();
    emit_fn(&mut s, "vector_core", &ops);
    emit_fn(&mut s, "keccak_init", &k_init);
    emit_fn(&mut s, "keccak_absorb", &k_absorb);
    emit_fn(&mut s, "keccak_f", &k_f);
    emit_take(&mut s);
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector_core.rs");
    std::fs::write(out, s).unwrap();
    println!("cargo:rerun-if-changed=src/shake.rs");
}
