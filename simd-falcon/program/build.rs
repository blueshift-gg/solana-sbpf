//! Generates the vectorized part of the Falcon-512 verifier.
//!
//! Firedancer's algorithm (fd_falcon.c, branch drubin/falcon-syscall): a Cooley-Tukey forward NTT
//! and a Gentleman-Sande inverse NTT over Z/12289 with lazy reduction in 32 bit lanes, a pointwise
//! product, and the squared norm. Two adaptations for the vector intrinsics (AVX2, 8 x u32):
//! - Twiddle products use Montgomery reduction (R = 2^16) with 32 bit multiplies only, instead of
//!   the Barrett reduction with 64 bit multiplies of the AVX-512 code.
//! - The last three forward levels (and the first three inverse levels) operate on transposed
//!   8 x 8 tiles, so every butterfly stays between whole vectors.
//!
//! The vector code must address slots with compile time constants for the JIT to lower each
//! intrinsic call inline, so all of it is emitted as straight-line code. The generated op stream is
//! simulated here and checked against schoolbook polynomial arithmetic before the program is built.

use std::fmt::Write;

const Q: u32 = 12289;
const N: usize = 512;
const R_LOG: u32 = 16;

// Slots (8 x u32 each) of the vector region
const P: u8 = 0; // working polynomial (s2, NTT, product, m), 64 slots
const C: u8 = 64; // hash_to_point candidates, 64 slots
const Q1: u8 = 128;
const Q2: u8 = 129;
const Q4: u8 = 130;
const Q8: u8 = 131;
const MASK: u8 = 132;
const QINV: u8 = 133;
const TW: u8 = 134;
const TWV: u8 = 135;
const T0: u8 = 136;
const T1: u8 = 137;
const T2: u8 = 138;
const T3: u8 = 139;
const ACC1: u8 = 140;
const ACC2: u8 = 141;

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

fn to_mont(x: u32) -> u32 {
    (((x as u64) << R_LOG) % Q as u64) as u32
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
    Sub(u8, u8, u8),
    Mullo(u8, u8, u8),
    Minu(u8, u8, u8),
    And(u8, u8, u8),
    Srl(u8, u8, u8),
    Mov(u8, u8),
    Bcast(u8, u32),
    Transpose(u8),
    Xor(u8, u8, u8),
    Or(u8, u8, u8),
    Andn(u8, u8, u8),
    /// (dst, data, indices)
    Permd(u8, u8, u8),
    Ror(u8, u8, u8),
    Sll(u8, u8, u8),
    Sllv(u8, u8, u8),
    Srlv(u8, u8, u8),
    Blend(u8, u8, u8, u8),
    /// Load a generated table row: (slot, table, row)
    Load(u8, &'static str, usize),
}

struct Gen {
    ops: Vec<Op>,
}

impl Gen {
    fn op(&mut self, op: Op) {
        self.ops.push(op);
    }

    fn q_slot(k: u32) -> u8 {
        match k {
            1 => Q1,
            2 => Q2,
            4 => Q4,
            8 => Q8,
            _ => unreachable!(),
        }
    }

    /// dst = src reduced from [0, bound) to [0, Q) with conditional subtractions
    /// (`x = min(x, x - kQ)`, the wrap-around makes the subtraction a no-op below kQ)
    fn reduce(&mut self, dst: u8, src: u8, mut bound: u64) {
        let mut cur = src;
        for k in [8u32, 4, 2, 1] {
            let kq = k as u64 * Q as u64;
            if bound > kq {
                self.op(Op::Sub(T1, cur, Self::q_slot(k)));
                self.op(Op::Minu(dst, cur, T1));
                cur = dst;
                bound = kq.max(bound - kq);
            }
        }
        assert!(bound <= Q as u64);
        if cur != dst {
            self.op(Op::Mov(dst, cur));
        }
    }

    /// dst = a * b * R^-1 mod Q in [0, Q), for a < a_bound and b < Q (Montgomery, R = 2^16)
    fn mont(&mut self, dst: u8, a: u8, b: u8, a_bound: u64) {
        let t_max = (a_bound - 1) * (Q as u64 - 1);
        assert!(t_max + 0xffff * Q as u64 <= u32::MAX as u64, "Montgomery input overflows 32 bits");
        self.op(Op::Mullo(T0, a, b));
        self.op(Op::And(T1, T0, MASK));
        self.op(Op::Mullo(T1, T1, QINV));
        self.op(Op::And(T1, T1, MASK));
        self.op(Op::Mullo(T1, T1, Q1));
        self.op(Op::Add(T0, T0, T1));
        self.op(Op::Srl(T0, T0, R_LOG as u8));
        let u_bound = (t_max + 0xffff * Q as u64) / (1 << R_LOG) + 1;
        self.reduce(dst, T0, u_bound);
    }

    /// Cooley-Tukey butterfly: (A, B) = (A + B*w, A - B*w + Q), inputs < bound
    fn ct(&mut self, a: u8, b: u8, w: u8, bound: u64) {
        self.mont(T2, b, w, bound);
        self.op(Op::Add(b, a, Q1));
        self.op(Op::Sub(b, b, T2));
        self.op(Op::Add(a, a, T2));
    }

    /// Gentleman-Sande butterfly: (A, B) = (A + B, (A - B + off) * w), inputs < bound
    fn gs(&mut self, a: u8, b: u8, w: u8, bound: u64, off_q: u32, reduce_sum: bool) {
        assert!(off_q as u64 * Q as u64 >= bound);
        self.op(Op::Add(T3, a, Self::q_slot(off_q)));
        self.op(Op::Sub(T3, T3, b));
        self.op(Op::Add(a, a, b));
        if reduce_sum {
            self.reduce(a, a, 2 * bound);
        }
        self.mont(b, T3, w, bound + off_q as u64 * Q as u64);
    }
}

/// Rows of the twiddle tables for the transposed levels (7 per 8 x 8 tile):
/// row 0: t = 4, rows 1..3: t = 2 (g = 0, 1), rows 3..7: t = 1 (g = 0..4).
/// Lane l of tile T belongs to block b = 8T + l (8 consecutive coefficients).
fn tile_table(psi: &[u32; N]) -> Vec<[u32; 8]> {
    let mut rows = Vec::new();
    for t in 0..8 {
        let row = |f: &dyn Fn(usize) -> usize| -> [u32; 8] { std::array::from_fn(|l| to_mont(psi[f(8 * t + l)])) };
        rows.push(row(&|b| 64 + b));
        for g in 0..2 {
            rows.push(row(&|b| 128 + 2 * b + g));
        }
        for g in 0..4 {
            rows.push(row(&|b| 256 + 4 * b + g));
        }
    }
    rows
}

fn tile_row(tile: usize, t: usize, g: usize) -> usize {
    tile * 7 + match t {
        4 => 0,
        2 => 1 + g,
        _ => 3 + g,
    }
}

fn generate(psi_pos: &[u32; N], psi_neg: &[u32; N]) -> Vec<Op> {
    let mut g = Gen { ops: Vec::new() };
    let q = Q as u64;
    // Constants
    let qinv = (1u32 << R_LOG).wrapping_sub((1..1u32 << R_LOG).find(|x| (x * Q) & 0xffff == 1).unwrap());
    for (slot, value) in [(Q1, Q), (Q2, 2 * Q), (Q4, 4 * Q), (Q8, 8 * Q), (MASK, 0xffff), (QINV, qinv)] {
        g.op(Op::Bcast(slot, value));
    }
    // ||s2||^2 (s2 in [0, Q)): centered |x| = min(x, Q - x)
    for s in 0..64 {
        g.op(Op::Sub(T2, Q1, P + s));
        g.op(Op::Minu(T3, P + s, T2));
        g.op(Op::Mullo(T3, T3, T3));
        g.op(Op::Add(ACC2, ACC2, T3));
    }
    // Forward NTT of s2
    let (mut t, mut m, mut bound) = (N, 1, q);
    while m < N {
        t >>= 1;
        if t >= 8 {
            let tv = (t / 8) as u8;
            for i in 0..m {
                g.op(Op::Bcast(TW, to_mont(psi_pos[m + i])));
                let base = (2 * i * t / 8) as u8;
                for k in 0..tv {
                    g.ct(P + base + k, P + base + tv + k, TW, bound);
                }
            }
        } else {
            if t == 4 {
                for tile in 0..8 {
                    g.op(Op::Transpose(P + 8 * tile));
                }
            }
            for tile in 0..8u8 {
                let groups = 4 / t;
                for grp in 0..groups {
                    g.op(Op::Load(TWV, "TWF", tile_row(tile as usize, t, grp)));
                    // Element pairs (e, e + t) of the group, e = grp * 2t + 0..t
                    for e in grp * 2 * t..grp * 2 * t + t {
                        g.ct(P + 8 * tile + e as u8, P + 8 * tile + (e + t) as u8, TWV, bound);
                    }
                }
            }
        }
        bound += q;
        m <<= 1;
    }
    // Pointwise product with the prepared pubkey (NTT(h) * N^-1, Montgomery form)
    for s in 0..64u8 {
        g.op(Op::Load(TWV, "H_PREP", s as usize));
        g.mont(P + s, P + s, TWV, bound);
    }
    // Inverse NTT
    let (mut t, mut m, mut off_q, mut bound) = (1usize, N, 1u32, q);
    while m > 1 {
        let h = m / 2;
        let reduce_sum = off_q >= 8;
        if t < 8 {
            for tile in 0..8u8 {
                for grp in 0..4 / t {
                    g.op(Op::Load(TWV, "TWI", tile_row(tile as usize, t, grp)));
                    for e in grp * 2 * t..grp * 2 * t + t {
                        g.gs(P + 8 * tile + e as u8, P + 8 * tile + (e + t) as u8, TWV, bound, off_q, reduce_sum);
                    }
                }
            }
            if t == 4 {
                for tile in 0..8 {
                    g.op(Op::Transpose(P + 8 * tile));
                }
            }
        } else {
            let tv = (t / 8) as u8;
            for i in 0..h {
                g.op(Op::Bcast(TW, to_mont(psi_neg[h + i])));
                let base = (2 * i * t / 8) as u8;
                for k in 0..tv {
                    g.gs(P + base + k, P + base + tv + k, TW, bound, off_q, reduce_sum);
                }
            }
        }
        if reduce_sum {
            bound = q;
            off_q = 1;
        } else {
            bound *= 2;
            off_q *= 2;
        }
        t <<= 1;
        m >>= 1;
    }
    // s1 = c - s2 * h (c < 5Q unreduced candidates, m < bound) and ||s1||^2
    assert!(bound <= 2 * q);
    for s in 0..64u8 {
        g.op(Op::Add(T3, C + s, Q2));
        g.op(Op::Sub(T3, T3, P + s));
        g.reduce(T3, T3, 7 * q);
        g.op(Op::Sub(T2, Q1, T3));
        g.op(Op::Minu(T3, T3, T2));
        g.op(Op::Mullo(T3, T3, T3));
        g.op(Op::Add(ACC1, ACC1, T3));
    }
    g.ops
}

/// Executes the op stream on a model of the vector region
fn simulate(ops: &[Op], region: &mut [[u32; 8]; 256], tables: &dyn Fn(&str, usize) -> [u32; 8]) {
    for op in ops {
        let lanes = |r: &[[u32; 8]; 256], a: u8, b: u8, f: &dyn Fn(u32, u32) -> u32| -> [u32; 8] {
            std::array::from_fn(|i| f(r[a as usize][i], r[b as usize][i]))
        };
        match *op {
            Op::Add(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x.wrapping_add(y)),
            Op::Sub(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x.wrapping_sub(y)),
            Op::Mullo(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x.wrapping_mul(y)),
            Op::Minu(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x.min(y)),
            Op::And(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x & y),
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
            Op::Load(d, table, row) => region[d as usize] = tables(table, row),
            Op::Xor(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x ^ y),
            Op::Or(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| x | y),
            Op::Andn(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| !x & y),
            Op::Permd(d, a, idx) => {
                let (data, idx) = (region[a as usize], region[idx as usize]);
                region[d as usize] = std::array::from_fn(|i| data[(idx[i] & 7) as usize]);
            }
            Op::Ror(d, a, n) => region[d as usize] = region[a as usize].map(|x| x.rotate_right(n as u32)),
            Op::Sll(d, a, n) => region[d as usize] = region[a as usize].map(|x| x << n),
            Op::Sllv(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| if y > 31 { 0 } else { x << y }),
            Op::Srlv(d, a, b) => region[d as usize] = lanes(region, a, b, &|x, y| if y > 31 { 0 } else { x >> y }),
            Op::Blend(d, a, b, m) => {
                let (x, y) = (region[a as usize], region[b as usize]);
                region[d as usize] = std::array::from_fn(|i| if (m >> i) & 1 != 0 { y[i] } else { x[i] });
            }
        }
    }
}

/// Splits the output of the forward NTT into the tile layout of the vector code:
/// coefficient 64T + 8k + e lives in slot 8T + e, lane k
fn to_tile_layout(x: &[u32; N]) -> Vec<[u32; 8]> {
    (0..64).map(|slot| std::array::from_fn(|k| x[64 * (slot / 8) + 8 * k + slot % 8])).collect()
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

fn prepare(h: &[u32; N], psi_pos: &[u32; N]) -> Vec<[u32; 8]> {
    let mut hn = *h;
    ntt_ref(&mut hn, psi_pos);
    let n_inv = pow_mod(N as u64, Q as u64 - 2, Q as u64);
    let folded: [u32; N] = std::array::from_fn(|i| to_mont((hn[i] as u64 * n_inv % Q as u64) as u32));
    to_tile_layout(&folded)
}

fn self_test(ops: &[Op], psi_pos: &[u32; N], twf: &[[u32; 8]], twi: &[[u32; 8]]) {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rand = |m: u32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as u32
    };
    for round in 0..4 {
        let h: [u32; N] = std::array::from_fn(|_| rand(Q));
        let s2: [u32; N] = std::array::from_fn(|i| if round == 0 && i > 3 { 0 } else { rand(Q) });
        let c: [u32; N] = std::array::from_fn(|_| rand(5 * Q));
        let h_prep = prepare(&h, psi_pos);
        let mut region = [[0u32; 8]; 256];
        for i in 0..N {
            region[P as usize + i / 8][i % 8] = s2[i];
            region[C as usize + i / 8][i % 8] = c[i];
        }
        let tables = |name: &str, row: usize| match name {
            "TWF" => twf[row],
            "TWI" => twi[row],
            _ => h_prep[row],
        };
        simulate(ops, &mut region, &tables);
        // Schoolbook s2 * h mod (x^N + 1, Q)
        let mut prod = [0i64; N];
        for i in 0..N {
            for j in 0..N {
                let v = s2[i] as i64 * h[j] as i64;
                if i + j < N { prod[i + j] += v } else { prod[i + j - N] -= v }
            }
        }
        let centered = |x: i64| {
            let x = x.rem_euclid(Q as i64);
            if x > Q as i64 / 2 { x - Q as i64 } else { x }
        };
        let norm1: i64 = (0..N).map(|i| centered(c[i] as i64 - prod[i]).pow(2)).sum();
        let norm2: i64 = (0..N).map(|i| centered(s2[i] as i64).pow(2)).sum();
        let got1: i64 = region[ACC1 as usize].iter().map(|x| *x as i64).sum();
        let got2: i64 = region[ACC2 as usize].iter().map(|x| *x as i64).sum();
        assert_eq!(got2, norm2, "s2 norm mismatch (round {round})");
        assert_eq!(got1, norm1, "s1 norm mismatch (round {round})");
        // The working polynomial holds m = s2 * h (mod Q, < 2Q)
        for i in 0..N {
            assert_eq!(region[P as usize + i / 8][i % 8] as i64 % Q as i64, prod[i].rem_euclid(Q as i64), "m[{i}]");
        }
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
            Op::Sub(d, a, b) => writeln!(s, "    sub({d}, {a}, {b});"),
            Op::Mullo(d, a, b) => writeln!(s, "    mullo({d}, {a}, {b});"),
            Op::Minu(d, a, b) => writeln!(s, "    minu({d}, {a}, {b});"),
            Op::And(d, a, b) => writeln!(s, "    and({d}, {a}, {b});"),
            Op::Srl(d, a, n) => writeln!(s, "    srl({d}, {a}, {n});"),
            Op::Mov(d, a) => writeln!(s, "    mov({d}, {a});"),
            Op::Bcast(d, v) => writeln!(s, "    bcast({d}, {v});"),
            Op::Transpose(d) => writeln!(s, "    transpose({d});"),
            Op::Load(d, t, row) => writeln!(s, "    load({d}, {t}.0[{row}].as_ptr());"),
            Op::Xor(d, a, b) => writeln!(s, "    xor({d}, {a}, {b});"),
            Op::Or(d, a, b) => writeln!(s, "    or({d}, {a}, {b});"),
            Op::Andn(d, a, b) => writeln!(s, "    andn({d}, {a}, {b});"),
            Op::Permd(d, a, i) => writeln!(s, "    permd({d}, {a}, {i});"),
            Op::Ror(d, a, n) => writeln!(s, "    ror({d}, {a}, {n});"),
            Op::Sll(d, a, n) => writeln!(s, "    sll({d}, {a}, {n});"),
            Op::Sllv(d, a, b) => writeln!(s, "    sllv({d}, {a}, {b});"),
            Op::Srlv(d, a, b) => writeln!(s, "    srlv({d}, {a}, {b});"),
            Op::Blend(d, a, b, m) => writeln!(s, "    blend({d}, {a}, {b}, {m});"),
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
// pi maps the R layout to the C layout with one lane permutation per slot, chi writes into the
// other tile pair, and an 8 x 8 transpose returns to the R layout for the next round. Lanes and
// slots 5..8 of a tile only ever hold values that never flow into lanes 0..5.

mod shake {
    #![allow(dead_code)]
    include!("src/shake.rs");
}

const K_TILES: [(u8, u8); 2] = [(0, 8), (16, 24)]; // (E, O) tile pairs
const KT0: u8 = 32;
const KT1: u8 = 33;
const K_CE: u8 = 36;
const K_CO: u8 = 37;
const K_DE: u8 = 38;
const K_DO: u8 = 39;
const K_IN_E: u8 = 40; // absorb buffer rows 0..4
const K_IN_O: u8 = 44;
const K_OUT_L: u8 = 48; // squeezed words rows 0..4
const K_OUT_H: u8 = 52;
const K_RC_E: u8 = 142; // 24 slots
const K_RC_O: u8 = 166; // 24 slots
const K_PM1: u8 = 190;
const K_PP1: u8 = 191;
const K_PI: u8 = 192; // 5 slots
const K_AE: u8 = 197; // rotation amounts, 5 slots each
const K_AEC: u8 = 202;
const K_AO: u8 = 207;
const K_AOC: u8 = 212;
const K_M16: u8 = 217;
const K_MH16: u8 = 218;
const K_S8: u8 = 219;
const K_S4: u8 = 220;
const K_S2: u8 = 221;
const K_S1: u8 = 222;

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

/// Rows of the Keccak constant table (loaded into slots K_RC_E..K_AOC by keccak_init)
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
    let amounts = |f: &dyn Fn(u32) -> u32, y: usize, complement: bool| -> [u32; 8] {
        std::array::from_fn(|x| {
            let a = if x < 5 { f(r[x][y]) } else { 0 };
            if complement { 32 - a } else { a }
        })
    };
    let ae = |r: u32| if r % 2 == 0 { r / 2 } else { (r + 1) / 2 };
    let ao = |r: u32| if r % 2 == 0 { r / 2 } else { (r - 1) / 2 };
    for (f, complement) in [(&ae as &dyn Fn(u32) -> u32, false), (&ae, true), (&ao, false), (&ao, true)] {
        for y in 0..5 {
            rows.push(amounts(f, y, complement));
        }
    }
    rows
}

fn keccak_init_ops() -> Vec<Op> {
    let mut ops: Vec<Op> = (0..keccak_table().len()).map(|row| Op::Load(K_RC_E + row as u8, "KC", row)).collect();
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
    let (mut cur, mut alt) = (K_TILES[0], K_TILES[1]);
    for round in 0..24u8 {
        let (e, o) = cur;
        // theta
        for (c, t) in [(K_CE, e), (K_CO, o)] {
            ops.push(Op::Xor(c, t, t + 1));
            for y in 2..5 {
                ops.push(Op::Xor(c, c, t + y));
            }
        }
        ops.push(Op::Permd(KT0, K_CE, K_PM1));
        ops.push(Op::Permd(KT1, K_CO, K_PP1));
        ops.push(Op::Ror(KT1, KT1, 31));
        ops.push(Op::Xor(K_DE, KT0, KT1));
        ops.push(Op::Permd(KT0, K_CO, K_PM1));
        ops.push(Op::Permd(KT1, K_CE, K_PP1));
        ops.push(Op::Xor(K_DO, KT0, KT1));
        for y in 0..5 {
            ops.push(Op::Xor(e + y, e + y, K_DE));
            ops.push(Op::Xor(o + y, o + y, K_DO));
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
            // C is free after theta
            ops.push(Op::Sllv(KT1, src, K_AE + y));
            ops.push(Op::Srlv(K_CE, src, K_AEC + y));
            ops.push(Op::Or(ey, KT1, K_CE));
            ops.push(Op::Sllv(KT1, oy, K_AO + y));
            ops.push(Op::Srlv(K_CE, oy, K_AOC + y));
            ops.push(Op::Or(oy, KT1, K_CE));
        }
        // pi: row X becomes column X
        for t in [e, o] {
            for x in 0..5 {
                ops.push(Op::Permd(t + x, t + x, K_PI + x));
            }
        }
        // chi into the other tiles
        for (src, dst) in [(e, alt.0), (o, alt.1)] {
            for x in 0..5u8 {
                ops.push(Op::Andn(KT0, src + (x + 1) % 5, src + (x + 2) % 5));
                ops.push(Op::Xor(dst + x, src + x, KT0));
            }
        }
        // iota
        ops.push(Op::Xor(alt.0, alt.0, K_RC_E + round));
        ops.push(Op::Xor(alt.1, alt.1, K_RC_O + round));
        // back to the R layout
        ops.push(Op::Transpose(alt.0));
        ops.push(Op::Transpose(alt.1));
        (cur, alt) = (alt, cur);
    }
    assert_eq!(cur, K_TILES[0]);
    ops
}

/// Rate rows 0..4 into standard 32 bit words: (E, O) halves -> perfect shuffle of 16 bit halves
fn keccak_squeeze_ops() -> Vec<Op> {
    let (e, o) = K_TILES[0];
    let mut ops = Vec::new();
    for y in 0..4u8 {
        for (dst, hi) in [(K_OUT_L + y, false), (K_OUT_H + y, true)] {
            if hi {
                ops.push(Op::Srl(KT0, e + y, 16));
                ops.push(Op::And(KT1, o + y, K_MH16));
            } else {
                ops.push(Op::And(KT0, e + y, K_M16));
                ops.push(Op::Sll(KT1, o + y, 16));
            }
            ops.push(Op::Or(dst, KT0, KT1));
            for (shift, mask) in [(8u8, K_S8), (4, K_S4), (2, K_S2), (1, K_S1)] {
                ops.push(Op::Srl(KT0, dst, shift));
                ops.push(Op::Xor(KT0, KT0, dst));
                ops.push(Op::And(KT0, KT0, mask));
                ops.push(Op::Xor(dst, dst, KT0));
                ops.push(Op::Sll(KT0, KT0, shift));
                ops.push(Op::Xor(dst, dst, KT0));
            }
        }
    }
    ops
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
    fn out_word(&self, lane: usize, hi: bool) -> u32 {
        self.region[(if hi { K_OUT_H } else { K_OUT_L }) as usize + lane / 5][lane % 5]
    }
    fn put_candidate(&mut self, i: usize, value: u32) {
        assert_eq!(i, self.out.len());
        self.out.push(value);
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
        assert_eq!(sponge.out, expected, "hash_to_point, message length {msg_len}");
    }
}

fn main() {
    let pk_path = std::env::var("FALCON_PK").unwrap_or_else(|_| "../fixtures/falcon.pk".to_string());
    println!("cargo:rerun-if-changed={pk_path}");
    println!("cargo:rerun-if-changed=build.rs");
    let (psi_pos, psi_neg) = twiddles();
    let twf = tile_table(&psi_pos);
    let twi = tile_table(&psi_neg);
    let ops = generate(&psi_pos, &psi_neg);
    self_test(&ops, &psi_pos, &twf, &twi);
    let h = decode_pubkey(&std::fs::read(&pk_path).expect("pubkey fixture"));
    let h_prep = prepare(&h, &psi_pos);
    let (k_init, k_absorb, k_f, k_squeeze) = (keccak_init_ops(), keccak_absorb_ops(), keccak_f_ops(), keccak_squeeze_ops());
    keccak_self_test(&k_f, &k_absorb, &k_squeeze, &k_init);
    let mut s = String::new();
    emit_table(&mut s, "H_PREP", &h_prep);
    emit_table(&mut s, "TWF", &twf);
    emit_table(&mut s, "TWI", &twi);
    emit_table(&mut s, "KC", &keccak_table());
    writeln!(s, "pub const ACC1: usize = {ACC1};\npub const ACC2: usize = {ACC2};\npub const C_SLOT: usize = {C};\npub const P_SLOT: usize = {P};").unwrap();
    writeln!(s, "pub const K_IN_E: usize = {K_IN_E};\npub const K_IN_O: usize = {K_IN_O};\npub const K_OUT_L: usize = {K_OUT_L};\npub const K_OUT_H: usize = {K_OUT_H};").unwrap();
    emit_fn(&mut s, "vector_core", &ops);
    emit_fn(&mut s, "keccak_init", &k_init);
    emit_fn(&mut s, "keccak_absorb", &k_absorb);
    emit_fn(&mut s, "keccak_f", &k_f);
    emit_fn(&mut s, "keccak_squeeze", &k_squeeze);
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector_core.rs");
    std::fs::write(out, s).unwrap();
    println!("cargo:rerun-if-changed=src/shake.rs");
}
