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

// Slots of the vector region (16 coefficients each)
const P: u8 = 0; // working polynomial (s2, NTT, product, m), 32 slots
const C: u8 = 64; // hash_to_point candidates (u16 < 5Q), 32 slots
const QV: u8 = 104; // q in every word
const Q2V: u8 = 105;
const Q4V: u8 = 106;
const ONE: u8 = 107; // R mod q (Montgomery one) and its q^-1 multiple
const ONE_P: u8 = 108;
const TW: u8 = 109; // twiddle and its q^-1 multiple
const TW_P: u8 = 110;
const T0: u8 = 111;
const T1: u8 = 112;
const T2: u8 = 113;
const ACC1: u8 = 114; // ||s1||^2 partial sums (u32 lanes)
const ACC2: u8 = 115; // ||s2||^2
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

impl Gen {
    fn op(&mut self, op: Op) {
        self.ops.push(op);
    }

    fn set_bound(&mut self, slot: u8, bound: i64) {
        self.bound[slot as usize] = bound;
        self.checks.push((self.ops.len(), slot, bound));
    }

    /// Loads table rows (appended to `table`) into consecutive slots from `slot`
    fn rows(&mut self, slot: u8, table: &'static str, new_rows: &[[u32; 8]]) {
        let rows = self.tables.entry(table).or_default();
        let index = rows.len();
        rows.extend_from_slice(new_rows);
        self.op(Op::Load(slot, table, index, new_rows.len() as u8));
    }

    /// Twiddle slots for per word Montgomery constants `w` (bound returned)
    fn twiddle_rows(&mut self, table: &'static str, w: [i64; 16]) -> i64 {
        let row = |f: &dyn Fn(i64) -> i64| -> [u32; 8] { std::array::from_fn(|i| pair(f(w[2 * i]), f(w[2 * i + 1]))) };
        let (plain, prime) = (row(&|x| x), row(&mont_prime));
        assert_eq!(TW_P, TW + 1);
        self.rows(TW, table, &[plain, prime]);
        w.iter().map(|x| x.abs()).max().unwrap()
    }

    /// Broadcast twiddle (bound returned)
    fn twiddle_bcast(&mut self, w: i64) -> i64 {
        self.op(Op::Bcast(TW, pair(w, w)));
        self.op(Op::Bcast(TW_P, pair(mont_prime(w), mont_prime(w))));
        w.abs()
    }

    /// dst = mont(a, W) (dst may be a)
    fn mont(&mut self, dst: u8, a: u8, w: u8, w_p: u8, w_abs: i64) {
        self.op(Op::Mullo16(T0, a, w_p));
        self.op(Op::Mulhi16(T1, a, w));
        self.op(Op::Mulhi16(T0, T0, QV));
        self.op(Op::Sub16(dst, T1, T0));
        self.set_bound(dst, mont_bound(self.bound[a as usize], w_abs));
    }

    /// Transposes a tile of 8 slots: every slot may now hold values of any slot of the tile
    fn transpose(&mut self, tile: u8) {
        self.op(Op::Transpose(tile));
        let max = (tile..tile + 8).map(|s| self.bound[s as usize]).max().unwrap();
        for s in tile..tile + 8 {
            self.set_bound(s, max);
        }
    }

    fn reduce(&mut self, x: u8) {
        let one = mont_centered(1).abs();
        self.mont(x, x, ONE, ONE_P, one);
    }

    fn reduce_if(&mut self, x: u8, limit: i64) {
        if self.bound[x as usize] > limit {
            self.reduce(x);
        }
    }

    /// Cooley-Tukey: (a, b) = (a + b w, a - b w), twiddle in TW / TW_P
    fn ct(&mut self, a: u8, b: u8, w_abs: i64) {
        let bw = mont_bound(self.bound[b as usize], w_abs);
        self.reduce_if(a, I16_MAX - bw);
        assert!(self.bound[a as usize] + bw <= I16_MAX);
        self.mont(T2, b, TW, TW_P, w_abs);
        self.op(Op::Sub16(b, a, T2));
        self.op(Op::Add16(a, a, T2));
        let out = self.bound[a as usize] + bw;
        self.set_bound(a, out);
        self.set_bound(b, out);
    }

    /// Gentleman-Sande: (a, b) = (a + b, (a - b) w), twiddle in TW / TW_P
    fn gs(&mut self, a: u8, b: u8, w_abs: i64) {
        if self.bound[a as usize] + self.bound[b as usize] > I16_MAX {
            let (big, small) = if self.bound[a as usize] >= self.bound[b as usize] { (a, b) } else { (b, a) };
            self.reduce(big);
            if self.bound[a as usize] + self.bound[b as usize] > I16_MAX {
                self.reduce(small);
            }
        }
        let sum = self.bound[a as usize] + self.bound[b as usize];
        assert!(sum <= I16_MAX);
        self.op(Op::Sub16(T2, a, b));
        self.op(Op::Add16(a, a, b));
        self.set_bound(T2, sum);
        self.mont(b, T2, TW, TW_P, w_abs);
        self.set_bound(a, sum);
    }

    /// Level 1 on every slot (pairs inside a u32): forward (a, b) = (a + b w, a - b w),
    /// inverse (a, b) = (a + b, (a - b) w)
    fn level1(&mut self, slot: u8, w: [i64; 8], forward: bool) {
        let one = mont_centered(1);
        // Low words get the Montgomery one (a reduction), high words the twiddle
        let words: [i64; 16] = std::array::from_fn(|i| if i % 2 == 0 { one } else { w[i / 2] });
        let w_abs = self.twiddle_rows(if forward { "TWF1" } else { "TWI1" }, words);
        let x = P + slot;
        if forward {
            // M = (a', b w), Ms = (b w, a'), out = (M + Ms, Ms - M) blended
            self.mont(x, x, TW, TW_P, w_abs);
            let m = self.bound[x as usize];
            self.op(Op::Ror(T2, x, 16));
            self.op(Op::Sub16(T1, T2, x));
            self.op(Op::Add16(x, x, T2));
            self.op(Op::Blendw(x, x, T1, 0xaa));
            assert!(2 * m <= I16_MAX);
            self.set_bound(x, 2 * m);
        } else {
            // Xs = (b, a), S = X + Xs = (a + b, a + b), D = Xs - X = (b - a, a - b),
            // out = mont((S, D) blended, (one, w))
            self.reduce_if(x, I16_MAX / 2);
            let b = self.bound[x as usize];
            self.op(Op::Ror(T2, x, 16));
            self.op(Op::Sub16(T1, T2, x));
            self.op(Op::Add16(x, x, T2));
            self.op(Op::Blendw(x, x, T1, 0xaa));
            self.set_bound(x, 2 * b);
            self.mont(x, x, TW, TW_P, w_abs);
        }
    }
}

/// The vector core: ||s2||^2, NTT(s2), product with the prepared pubkey (table H_PREP), inverse
/// NTT, s1 = c - s2 h, ||s1||^2. Returns the op stream and the generated tables (H_PREP rows are
/// placeholders, filled by [prepare]).
fn generate(psi_pos: &[u32; N], psi_neg: &[u32; N]) -> (Vec<Op>, Tables, Vec<(usize, u8, i64)>) {
    let mut g = Gen { ops: Vec::new(), bound: [0; 256], checks: Vec::new(), tables: Default::default() };
    let one = mont_centered(1);
    for (slot, value) in [(QV, QI), (Q2V, 2 * QI), (Q4V, 4 * QI), (ONE, one), (ONE_P, mont_prime(one))] {
        g.op(Op::Bcast(slot, pair(value, value)));
    }
    // ||s2||^2 from the decoded (signed, |s2| < 2048) coefficients
    for s in 0..32 {
        g.op(Op::Madd16(T0, P + s, P + s));
        g.op(Op::Add(ACC2, ACC2, T0));
        g.set_bound(P + s, 2047);
    }
    let twiddle = |psi: &[u32; N], t: usize, i: usize| mont_centered(psi[N / (2 * t) + i] as i64);
    // Forward NTT, levels 256..16 between whole slots
    let mut t = 256;
    while t >= 16 {
        let ts = (t / 16) as u8;
        for i in 0..N / (2 * t) {
            let w_abs = g.twiddle_bcast(twiddle(psi_pos, t, i));
            let base = (2 * i * t / 16) as u8;
            for k in 0..ts {
                g.ct(P + base + k, P + base + ts + k, w_abs);
            }
        }
        t /= 2;
    }
    for tile in 0..4 {
        g.transpose(P + 8 * tile);
    }
    // Levels 8, 4, 2: slot c and c + t / 2 of a tile, block index from (tile, lane, group)
    for t in [8usize, 4, 2] {
        let d = t / 2; // slot distance
        for tile in 0..4 {
            for grp in 0..4 / d {
                let w: [i64; 16] = std::array::from_fn(|word| {
                    let (r, h) = (word / 2, word % 2);
                    let j = 128 * tile + 16 * r + 2 * (grp * 2 * d) + h;
                    twiddle(psi_pos, t, j / (2 * t))
                });
                let w_abs = g.twiddle_rows("TWF", w);
                for c in grp * 2 * d..grp * 2 * d + d {
                    g.ct(P + 8 * tile as u8 + c as u8, P + 8 * tile as u8 + (c + d) as u8, w_abs);
                }
            }
        }
    }
    for slot in 0..32u8 {
        let (tile, c) = (slot as usize / 8, slot as usize % 8);
        let w: [i64; 8] = std::array::from_fn(|r| twiddle(psi_pos, 1, (128 * tile + 16 * r + 2 * c) / 2));
        g.level1(slot, w, true);
    }
    // Pointwise product with the prepared pubkey (NTT(h) / N, Montgomery form)
    for s in 0..32u8 {
        g.rows(TW, "H_PREP", &[[0; 8]; 2]);
        g.mont(P + s, P + s, TW, TW_P, QI / 2 + 1);
    }
    // Inverse NTT: level 1, then 2, 4, 8 in the transposed layout, then 16..256
    for slot in 0..32u8 {
        let (tile, c) = (slot as usize / 8, slot as usize % 8);
        let w: [i64; 8] = std::array::from_fn(|r| twiddle(psi_neg, 1, (128 * tile + 16 * r + 2 * c) / 2));
        g.level1(slot, w, false);
    }
    for t in [2usize, 4, 8] {
        let d = t / 2;
        for tile in 0..4 {
            for grp in 0..4 / d {
                let w: [i64; 16] = std::array::from_fn(|word| {
                    let (r, h) = (word / 2, word % 2);
                    let j = 128 * tile + 16 * r + 2 * (grp * 2 * d) + h;
                    twiddle(psi_neg, t, j / (2 * t))
                });
                let w_abs = g.twiddle_rows("TWI", w);
                for c in grp * 2 * d..grp * 2 * d + d {
                    g.gs(P + 8 * tile as u8 + c as u8, P + 8 * tile as u8 + (c + d) as u8, w_abs);
                }
            }
        }
    }
    for tile in 0..4 {
        g.transpose(P + 8 * tile);
    }
    let mut t = 16;
    while t <= 256 {
        let ts = (t / 16) as u8;
        for i in 0..N / (2 * t) {
            let w_abs = g.twiddle_bcast(twiddle(psi_neg, t, i));
            let base = (2 * i * t / 16) as u8;
            for k in 0..ts {
                g.gs(P + base + k, P + base + ts + k, w_abs);
            }
        }
        t *= 2;
    }
    // s1 = c - m: c (u16 < 5q) to [0, 2q), u = c + q - m in [0, 4q) (m reduced below q), u to
    // [0, q), |s1| = min(u, q - u), squares summed in pairs into u32 lanes
    for s in 0..32u8 {
        let (cs, m) = (C + s, P + s);
        g.reduce_if(m, QI);
        assert!(g.bound[m as usize] <= QI);
        g.op(Op::Sub16(T2, cs, Q4V));
        g.op(Op::Minu16(T2, cs, T2)); // [0, 4q) (5q > 4q values wrapped below q)
        g.op(Op::Sub16(T1, T2, Q2V));
        g.op(Op::Minu16(T2, T2, T1)); // [0, 2q)
        g.op(Op::Add16(T2, T2, QV));
        g.op(Op::Sub16(T2, T2, m)); // [0, 4q)
        g.op(Op::Sub16(T1, T2, Q2V));
        g.op(Op::Minu16(T2, T2, T1)); // [0, 2q)
        g.op(Op::Sub16(T1, T2, QV));
        g.op(Op::Minu16(T2, T2, T1)); // [0, q)
        g.op(Op::Sub16(T1, QV, T2));
        g.op(Op::Minu16(T2, T2, T1)); // [0, q / 2]
        g.op(Op::Madd16(T1, T2, T2));
        g.op(Op::Add(ACC1, ACC1, T1));
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


/// NTT(h) / N in Montgomery form, laid out like the forward NTT output: rows (H, H q^-1) per slot
fn prepare(h: &[u32; N], psi_pos: &[u32; N]) -> Vec<[u32; 8]> {
    let mut hn = *h;
    ntt_ref(&mut hn, psi_pos);
    let n_inv = pow_mod(N as u64, Q as u64 - 2, Q as u64) as i64;
    let mut words = [[0i64; 16]; 32];
    for j in 0..N {
        let (slot, word) = transposed_pos(j);
        words[slot][word] = mont_centered(hn[j] as i64 * n_inv);
    }
    let mut rows = Vec::new();
    for w in words {
        rows.push(std::array::from_fn(|i| pair(w[2 * i], w[2 * i + 1])));
        rows.push(std::array::from_fn(|i| pair(mont_prime(w[2 * i]), mont_prime(w[2 * i + 1]))));
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
        let ntt_end = ops.iter().position(|op| matches!(*op, Op::Sub16(_, a, _) if a == C)).unwrap();
        for (k, op) in ops[..ntt_end].iter().enumerate() {
            if let Op::Add16(_, a, b) | Op::Sub16(_, a, b) = *op {
                let sub = matches!(op, Op::Sub16(..));
                for lane in 0..8 {
                    for h in 0..2 {
                        let x = (region[a as usize][lane] >> (16 * h)) as u16 as i16 as i32;
                        let y = (region[b as usize][lane] >> (16 * h)) as u16 as i16 as i32;
                        let r = if sub { x - y } else { x + y };
                        assert!(r == r as i16 as i32, "overflow at op {k} {op:?} round {round}: {x} {y}");
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
        let got1: i64 = region[ACC1 as usize].iter().map(|x| *x as i64).sum();
        let got2: i64 = region[ACC2 as usize].iter().map(|x| *x as i64).sum();
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
// Keccak-f[1600] on the vector intrinsics
//
// Lanes are bit-interleaved: a 64 bit lane is stored as its even bits (E) and its odd bits (O),
// so 64 bit rotations become 32 bit rotations. The state is two 8 x 8 tiles (E and O) of slots,
// 5 x 5 used:
//   R layout: slot y = row y, lane x        (theta, rho)
//   C layout: slot x = column x, lane y     (chi, iota)
// pi maps the R layout to the C layout with one lane permutation per slot, chi runs in place
// (three input logic, d ^ (!a & b)), and an 8 x 8 transpose returns to the R layout for the next
// round. Theta's parity and its application use three input xors, rho one variable rotate per
// half. Lanes and slots 5..8 of a tile only ever hold values that never flow into lanes 0..5.

mod shake {
    #![allow(dead_code)]
    include!("src/shake.rs");
}

const K_TILES: [(u8, u8); 1] = [(0, 8)]; // (E, O) tiles
const KT0: u8 = 32;
const KT1: u8 = 33;
const K_CE: u8 = 36;
const K_CO: u8 = 37;
const K_IN_E: u8 = 40; // absorb buffer rows 0..4
const K_IN_O: u8 = 44;
const K_OUT_L: u8 = 48; // squeezed words rows 0..4
const K_OUT_H: u8 = 52;
// Constants used by three input ops must be below slot 128 (7 bit operand fields)
const K_M16: u8 = 16;
const K_MH16: u8 = 17;
const K_S8: u8 = 18;
const K_S4: u8 = 19;
const K_S2: u8 = 20;
const K_S1: u8 = 21;
const K_RC_E: u8 = 142; // 24 slots
const K_RC_O: u8 = 166; // 24 slots
const K_PM1: u8 = 190;
const K_PP1: u8 = 191;
const K_PI: u8 = 192; // 5 slots
const K_AE: u8 = 197; // rotation amounts of the E and O halves, 5 slots each
const K_AO: u8 = 202;

/// d ^ a ^ b
const XOR3: u8 = 0x96;
/// d ^ (!a & b)
const CHI: u8 = 0xd2;
/// (d ^ a) & b
const XOR_AND: u8 = 0x28;
/// d | (a & b)
const OR_AND: u8 = 0xf8;

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

fn interleave(e: u32, o: u32) -> u64 {
    (0..32).fold(0u64, |x, i| x | ((e as u64 >> i) & 1) << (2 * i) | ((o as u64 >> i) & 1) << (2 * i + 1))
}

/// Rows of the Keccak constant table (loaded into slots K_RC_E..K_AO + 5 by keccak_init)
fn keccak_table() -> Vec<[u32; 8]> {
    let lane0 = |v: u32| -> [u32; 8] { std::array::from_fn(|i| if i == 0 { v } else { 0 }) };
    let idx = |f: &dyn Fn(u32) -> u32| -> [u32; 8] { std::array::from_fn(|i| if i < 5 { f(i as u32) } else { i as u32 }) };
    let mut rows = Vec::new();
    let rc = keccak_round_constants();
    rows.extend(rc.iter().map(|c| lane0(shake::deinterleave(*c).0)));
    rows.extend(rc.iter().map(|c| lane0(shake::deinterleave(*c).1)));
    rows.push(idx(&|x| (x + 4) % 5));
    rows.push(idx(&|x| (x + 1) % 5));
    for big_x in 0..5 {
        rows.push(idx(&|y| (big_x + 3 * y) % 5));
    }
    let r = keccak_rho_offsets();
    let amounts = |f: &dyn Fn(u32) -> u32, y: usize| -> [u32; 8] { std::array::from_fn(|x| if x < 5 { f(r[x][y]) } else { 0 }) };
    let ae = |r: u32| if r % 2 == 0 { r / 2 } else { (r + 1) / 2 };
    let ao = |r: u32| if r % 2 == 0 { r / 2 } else { (r - 1) / 2 };
    for f in [&ae as &dyn Fn(u32) -> u32, &ao] {
        for y in 0..5 {
            rows.push(amounts(f, y));
        }
    }
    rows
}

fn keccak_init_ops() -> Vec<Op> {
    let rows = keccak_table().len();
    let mut ops: Vec<Op> = (0..rows).step_by(32).map(|row| Op::Load(K_RC_E + row as u8, "KC", row, (rows - row).min(32) as u8)).collect();
    for (slot, v) in [(K_M16, 0x0000_ffff), (K_MH16, 0xffff_0000), (K_S8, 0x0000_ff00), (K_S4, 0x00f0_00f0), (K_S2, 0x0c0c_0c0c), (K_S1, 0x2222_2222)] {
        ops.push(Op::Bcast(slot, v));
    }
    ops
}

fn keccak_absorb_ops() -> Vec<Op> {
    let (e, o) = K_TILES[0];
    (0..4).flat_map(|y| [Op::Xor(e + y, e + y, K_IN_E + y), Op::Xor(o + y, o + y, K_IN_O + y)]).collect()
}

fn keccak_f_ops() -> Vec<Op> {
    let r = keccak_rho_offsets();
    let mut ops = Vec::new();
    let (e, o) = K_TILES[0];
    for round in 0..24u8 {
        // theta: column parities C, D[x] = C[x - 1] ^ rotl(C[x + 1], 1)
        for (c, t) in [(K_CE, e), (K_CO, o)] {
            ops.push(Op::Xor(c, t, t + 1));
            ops.push(Op::Tern(c, t + 2, t + 3, XOR3));
            ops.push(Op::Xor(c, c, t + 4));
        }
        ops.push(Op::Permd(KT0, K_CE, K_PM1));
        ops.push(Op::Permd(KT1, K_CO, K_PP1));
        ops.push(Op::Ror(KT1, KT1, 31));
        for y in 0..5 {
            ops.push(Op::Tern(e + y, KT0, KT1, XOR3));
        }
        ops.push(Op::Permd(KT0, K_CO, K_PM1));
        ops.push(Op::Permd(KT1, K_CE, K_PP1));
        for y in 0..5 {
            ops.push(Op::Tern(o + y, KT0, KT1, XOR3));
        }
        // rho: lanes with an odd offset swap halves first
        for y in 0..5u8 {
            let odd = (0..5).fold(0u8, |m, x| m | (((r[x][y as usize] & 1) as u8) << x));
            let (ey, oy) = (e + y, o + y);
            let src = if odd != 0 {
                ops.push(Op::Blend(KT0, ey, oy, odd));
                ops.push(Op::Blend(oy, oy, ey, odd));
                KT0
            } else {
                ey
            };
            ops.push(Op::Rolv(ey, src, K_AE + y));
            ops.push(Op::Rolv(oy, oy, K_AO + y));
        }
        // pi: row X becomes column X
        for t in [e, o] {
            for x in 0..5 {
                ops.push(Op::Permd(t + x, t + x, K_PI + x));
            }
        }
        // chi in place: columns 0 and 1 are needed after they are overwritten
        for t in [e, o] {
            ops.push(Op::Mov(KT0, t));
            ops.push(Op::Mov(KT1, t + 1));
            for x in 0..5u8 {
                let col = |c: u8| if x + c < 5 { t + x + c } else if x + c == 5 { KT0 } else { KT1 };
                ops.push(Op::Tern(t + x, col(1), col(2), CHI));
            }
        }
        // iota
        ops.push(Op::Xor(e, e, K_RC_E + round));
        ops.push(Op::Xor(o, o, K_RC_O + round));
        // back to the R layout
        ops.push(Op::Transpose(e));
        ops.push(Op::Transpose(o));
    }
    ops
}

/// Rate rows 0..4 into standard 32 bit words: (E, O) halves -> perfect shuffle of 16 bit halves
fn keccak_squeeze_ops() -> Vec<Op> {
    let (e, o) = K_TILES[0];
    let mut ops = Vec::new();
    for y in 0..4u8 {
        for (dst, hi) in [(K_OUT_L + y, false), (K_OUT_H + y, true)] {
            if hi {
                ops.push(Op::Srl(dst, e + y, 16));
                ops.push(Op::Tern(dst, o + y, K_MH16, OR_AND));
            } else {
                ops.push(Op::Sll(dst, o + y, 16));
                ops.push(Op::Tern(dst, e + y, K_M16, OR_AND));
            }
            // delta swaps: t = ((x >> s) ^ x) & mask; x ^= t ^ (t << s)
            for (shift, mask) in [(8u8, K_S8), (4, K_S4), (2, K_S2), (1, K_S1)] {
                ops.push(Op::Srl(KT0, dst, shift));
                ops.push(Op::Tern(KT0, dst, mask, XOR_AND));
                ops.push(Op::Sll(KT1, KT0, shift));
                ops.push(Op::Tern(dst, KT0, KT1, XOR3));
            }
        }
    }
    ops
}

/// Region byte offsets of the 68 samples of a squeezed block, in stream order: sample 4L + j is
/// bytes 2j..2j + 2 of rate lane L = 5y + x, i.e. half j % 2 of word j / 2 of that lane
fn sample_offsets() -> Vec<usize> {
    (0..shake::RATE_LANES * 4)
        .map(|i| {
            let (lane, j) = (i / 4, i % 4);
            let slot = if j < 2 { K_OUT_L } else { K_OUT_H } as usize + lane / 5;
            32 * slot + 4 * (lane % 5) + 2 * (j % 2)
        })
        .collect()
}

fn emit_take(s: &mut String) {
    // A cold call keeps the rare rejection out of line, so the common path is a not taken branch
    // instead of a select
    writeln!(s, "#[cold]\n#[inline(never)]\nfn rejected(p: *mut u16) -> *mut u16 {{\n    unsafe {{ p.sub(1) }}\n}}").unwrap();
    writeln!(s, "/// Appends the accepted samples of a squeezed block (`b`: the stored slots K_OUT_L..K_OUT_H + 4)").unwrap();
    writeln!(s, "/// at `p` (see shake::Sponge::take)").unwrap();
    writeln!(s, "#[inline(always)]\npub unsafe fn take_candidates(b: *const u8, mut p: *mut u16) -> *mut u16 {{").unwrap();
    for off in sample_offsets().iter().map(|off| off - 32 * K_OUT_L as usize) {
        writeln!(s, "    let v = u16::from_be((b.add({off}) as *const u16).read_volatile()); p.write_volatile(v); p = p.add(1); if v >= {} {{ p = rejected(p); }}", 5 * Q).unwrap();
    }
    writeln!(s, "    p\n}}").unwrap();
}

struct SimSponge<'a> {
    region: [[u32; 8]; 256],
    f: &'a [Op],
    absorb: &'a [Op],
    squeeze: &'a [Op],
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
    fn set_input(&mut self, lane: usize, even: u32, odd: u32) {
        self.region[(K_IN_E as usize) + lane / 5][lane % 5] = even;
        self.region[(K_IN_O as usize) + lane / 5][lane % 5] = odd;
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
    fn squeeze(&mut self) {
        let s = self.squeeze;
        self.run(s);
    }
    fn take(&mut self, n: usize) -> usize {
        assert_eq!(n, self.out.len());
        let bytes: Vec<u8> = self.region.iter().flat_map(|slot| slot.iter().flat_map(|w| w.to_le_bytes())).collect();
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

fn keccak_self_test(f: &[Op], absorb: &[Op], squeeze: &[Op], init: &[Op]) {
    assert_eq!(&shake256_ref(b"", 8), &[0x46, 0xb9, 0xdd, 0x2b, 0x0b, 0xa8, 0x8d, 0x13], "reference SHAKE256");
    for i in 0..100u64 {
        let x = i.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(i as u32);
        let (e, o) = shake::deinterleave(x);
        assert_eq!(interleave(e, o), x);
    }
    let table = keccak_table();
    // The permutation alone
    let mut region = [[0u32; 8]; 256];
    simulate(init, &mut region, &|_, row| table[row]);
    let mut a: [u64; 25] = std::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0x0123_4567_89ab_cdef).rotate_left(i as u32 * 7));
    for i in 0..25 {
        let (e, o) = shake::deinterleave(a[i]);
        region[(K_TILES[0].0 as usize) + i / 5][i % 5] = e;
        region[(K_TILES[0].1 as usize) + i / 5][i % 5] = o;
    }
    simulate(f, &mut region, &|_, row| table[row]);
    keccak_f_ref(&mut a);
    for i in 0..25 {
        let got = interleave(region[(K_TILES[0].0 as usize) + i / 5][i % 5], region[(K_TILES[0].1 as usize) + i / 5][i % 5]);
        assert_eq!(got, a[i], "keccak lane {i}");
    }
    // hash_to_point through the driver against the reference
    for msg_len in [0usize, 1, 7, 95, 96, 135, 136, 200, 271, 272, 1000] {
        let nonce: Vec<u8> = (0..40u8).map(|i| i.wrapping_mul(37)).collect();
        let msg: Vec<u8> = (0..msg_len).map(|i| (i as u8).wrapping_mul(13) ^ 0x5a).collect();
        let mut sponge = SimSponge { region: [[0; 8]; 256], f, absorb, squeeze, table: table.clone(), out: Vec::new() };
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
    assert_eq!(K_IN_O, K_IN_E + 4);
    assert_eq!(K_OUT_H, K_OUT_L + 4);
    assert_eq!(ACC2, ACC1 + 1);
    let (psi_pos, psi_neg) = twiddles();
    let (ops, mut tables, checks) = generate(&psi_pos, &psi_neg);
    self_test(&ops, &tables, &checks, &psi_pos);
    let h = decode_pubkey(&std::fs::read(&pk_path).expect("pubkey fixture"));
    tables.insert("H_PREP", prepare(&h, &psi_pos));
    let (k_init, k_absorb, k_f, k_squeeze) = (keccak_init_ops(), keccak_absorb_ops(), keccak_f_ops(), keccak_squeeze_ops());
    keccak_self_test(&k_f, &k_absorb, &k_squeeze, &k_init);
    let mut s = String::new();
    for (name, rows) in &tables {
        emit_table(&mut s, name, rows);
    }
    emit_table(&mut s, "KC", &keccak_table());
    writeln!(s, "pub const ACC1: u8 = {ACC1};\npub const C_SLOT: u8 = {C};\npub const P_SLOT: u8 = {P};").unwrap();
    writeln!(s, "pub const K_IN_E: u8 = {K_IN_E};\npub const K_OUT_L: u8 = {K_OUT_L};").unwrap();
    emit_fn(&mut s, "vector_core", &ops);
    emit_fn(&mut s, "keccak_init", &k_init);
    emit_fn(&mut s, "keccak_absorb", &k_absorb);
    emit_fn(&mut s, "keccak_f", &k_f);
    emit_fn(&mut s, "keccak_squeeze", &k_squeeze);
    emit_take(&mut s);
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector_core.rs");
    std::fs::write(out, s).unwrap();
    println!("cargo:rerun-if-changed=src/shake.rs");
}
