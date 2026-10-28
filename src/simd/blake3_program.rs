#![allow(clippy::needless_range_loop)] // index loops mirror the reference algorithm
//! Generators for the BLAKE3 sBPF programs of the SIMD experiment.
//!
//! All programs are SBPFv4 and take the input region (`r1`) in the layout produced by
//! [build_input]. The 32 byte digest is written to the heap at [HEAP_OUT].
//!
//! Register conventions: `r9` is never used by the generated scalar code, because in memory mode it
//! holds the base address of the vector file in the heap. This keeps the slot and memory mode
//! programs identical except for the encoding of the vector instructions.

#![allow(clippy::arithmetic_side_effects, missing_docs)]

use super::asm::{Asm, Label, VecMode};
use crate::ebpf;

/// BLAKE3 initialization vector
pub const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
/// BLAKE3 message word permutation
pub const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
pub const CHUNK_START: i64 = 1;
pub const CHUNK_END: i64 = 2;
pub const PARENT: i64 = 4;
pub const ROOT: i64 = 8;
pub const CHUNK_LEN: usize = 1024;
pub const BLOCK_LEN: usize = 64;

/// Input layout: `[len: u64][pad][constants table][...][data @ INPUT_DATA]`
pub const INPUT_CONSTANTS: usize = 0x20;
pub const INPUT_DATA: usize = 0x400;

// Constants table entries (32 bytes each), the moral equivalent of .rodata
const C_EVEN: usize = 0;
const C_ODD: usize = 1;
const C_PLO: usize = 2;
const C_PHI: usize = 3;
const C_HI4: usize = 4;
const C_DUP_LO: usize = 5;
const C_IV_LO: usize = 6;
const C_LANE: usize = 7;
const C_IV: usize = 8;
const C_COUNT: usize = 9;

/// Heap layout
pub const HEAP_VECTOR_FILE: usize = 0x0000;
const HEAP_ARGS: usize = 0x0800;
const HEAP_PAD: usize = 0x0880;
const HEAP_STATE: usize = 0x08c0;
pub const HEAP_OUT: usize = 0x0900;
const HEAP_IV: usize = 0x0940;
pub const HEAP_CVS: usize = 0x1000;
/// Minimal heap size (enough for 1 MiB of input)
pub const HEAP_SIZE: usize = HEAP_CVS + 32 * 1024 + 0x1000;

// Arguments / locals in the heap
const A_OUT: i16 = HEAP_ARGS as i16; // compress8 output, hash_chunk output
const A_CVIN: i16 = HEAP_ARGS as i16 + 0x08;
const A_CNT: i16 = HEAP_ARGS as i16 + 0x10; // number of chunk CVs / tree level size
const A_I: i16 = HEAP_ARGS as i16 + 0x18;
const A_PAIRS: i16 = HEAP_ARGS as i16 + 0x20;
const A_NFULL: i16 = HEAP_ARGS as i16 + 0x28;
const A_NBLOCKS: i16 = HEAP_ARGS as i16 + 0x30;
const A_B: i16 = HEAP_ARGS as i16 + 0x38;
const A_CTR: i16 = HEAP_ARGS as i16 + 0x40;
const A_FLAGS: i16 = HEAP_ARGS as i16 + 0x48;
const A_STRIDE: i16 = HEAP_ARGS as i16 + 0x50;
const A_PARENT: i16 = HEAP_ARGS as i16 + 0x58;
const A_BASE: i16 = HEAP_ARGS as i16 + 0x60;
const A_ITER: i16 = HEAP_ARGS as i16 + 0x70;

// Vector slots
const S_STATE: u8 = 0; // 0..16, 8-way state
const S_MSG: u8 = 16; // 16..32, 8-way message (transposed)
const S_CV8: u8 = 32; // 32..40, 8-way chaining values
const S_CTR_LO: u8 = 40;
const S_CTR_HI: u8 = 41;
const S_IV: u8 = 42;
const S_IV_LO: u8 = 43;
const S_DUP_LO: u8 = 46;
const S_LANE: u8 = 47;
const S_CV: u8 = 48; // single-state chaining value
const S_ML: u8 = 49;
const S_MH: u8 = 50;
const S_ROW: u8 = 51; // 51..55
const S_T: u8 = 55; // 55..59
const S_EVEN: u8 = 59;
const S_ODD: u8 = 60;
const S_PLO: u8 = 61;
const S_PHI: u8 = 62;
const S_HI4: u8 = 63;

const VECTOR_FILE_REG: u8 = 9;

/// Compression kernel used by the generated program
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Plain scalar sBPF (32 bit ALU ops, state spilled to the heap)
    Scalar,
    /// Single-state SIMD compression (one 4 word row per vector)
    Simd1,
    /// Eight-way parallel compression of independent chunks / parents, plus Simd1 for the rest
    Simd8,
}

/// Message schedule of each round
pub fn msg_schedule() -> [[usize; 16]; 7] {
    let mut schedule = [[0usize; 16]; 7];
    let mut current: [usize; 16] = std::array::from_fn(|i| i);
    for round in schedule.iter_mut() {
        *round = current;
        current = std::array::from_fn(|i| current[MSG_PERMUTATION[i]]);
    }
    schedule
}

fn lanes_bytes(lanes: [u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, lane) in lanes.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&lane.to_le_bytes());
    }
    out
}

/// Blend mask selecting the message half for the in-vector message permutation
fn permutation_masks() -> (i16, i16) {
    let mut lo = 0;
    let mut hi = 0;
    for i in 0..8 {
        if MSG_PERMUTATION[i] >= 8 {
            lo |= 1 << i;
        }
        if MSG_PERMUTATION[8 + i] >= 8 {
            hi |= 1 << i;
        }
    }
    (lo, hi)
}

/// Builds the input region contents for a message
pub fn build_input(data: &[u8]) -> Vec<u8> {
    let mut input = vec![0u8; INPUT_DATA + data.len()];
    input[0..8].copy_from_slice(&(data.len() as u64).to_le_bytes());
    let p = MSG_PERMUTATION;
    let table: [[u32; 8]; C_COUNT] = [
        [0, 2, 4, 6, 0, 2, 4, 6],
        [1, 3, 5, 7, 1, 3, 5, 7],
        std::array::from_fn(|i| (p[i] & 7) as u32),
        std::array::from_fn(|i| (p[8 + i] & 7) as u32),
        [4, 5, 6, 7, 4, 5, 6, 7],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [IV[0], IV[1], IV[2], IV[3], IV[0], IV[1], IV[2], IV[3]],
        [0, 1, 2, 3, 4, 5, 6, 7],
        IV,
    ];
    for (i, entry) in table.iter().enumerate() {
        let at = INPUT_CONSTANTS + 32 * i;
        input[at..at + 32].copy_from_slice(&lanes_bytes(*entry));
    }
    input[INPUT_DATA..].copy_from_slice(data);
    input
}

struct Gen {
    a: Asm,
    kernel: Kernel,
    compress1: Label,
    compress8: Label,
    hash_chunk: Label,
    parent: Label,
}

impl Gen {
    fn new(kernel: Kernel, mode: VecMode) -> Self {
        let mut a = Asm::new(mode);
        let compress1 = a.new_label();
        let compress8 = a.new_label();
        let hash_chunk = a.new_label();
        let parent = a.new_label();
        Self {
            a,
            kernel,
            compress1,
            compress8,
            hash_chunk,
            parent,
        }
    }

    fn simd(&self) -> bool {
        self.kernel != Kernel::Scalar
    }

    /// Program prologue: vector file base, constants
    fn prologue(&mut self) {
        let a = &mut self.a;
        // r6 = input (r1 doesn't survive vector intrinsic calls)
        a.mov64(6, 1);
        a.lddw(
            VECTOR_FILE_REG,
            ebpf::MM_HEAP_START + HEAP_VECTOR_FILE as u64,
        );
        if self.kernel != Kernel::Scalar {
            for (slot, entry) in [
                (S_EVEN, C_EVEN),
                (S_ODD, C_ODD),
                (S_PLO, C_PLO),
                (S_PHI, C_PHI),
                (S_HI4, C_HI4),
                (S_DUP_LO, C_DUP_LO),
                (S_IV_LO, C_IV_LO),
                (S_LANE, C_LANE),
                (S_IV, C_IV),
            ] {
                a.vload(slot, 6, (INPUT_CONSTANTS + 32 * entry) as i16);
            }
        } else {
            a.lddw(8, ebpf::MM_HEAP_START);
            for (i, word) in IV.iter().enumerate() {
                a.mov32_imm(0, *word as i32 as i64);
                a.stxw(8, (HEAP_IV + 4 * i) as i16, 0);
            }
        }
    }

    /// Full BLAKE3 hash of the input
    fn main(&mut self) {
        self.prologue();
        let a = &mut self.a;
        a.ldxdw(7, 6, 0); // len
        a.add64_imm(6, INPUT_DATA as i64); // data
        a.lddw(8, ebpf::MM_HEAP_START);
        let multi = a.new_label();
        a.jgt_imm(7, CHUNK_LEN as i64, multi);
        // Single chunk: the chunk is the root
        a.mov64(1, 6);
        a.mov64(2, 7);
        a.mov64_imm(3, 0);
        a.mov64_imm(4, ROOT);
        a.mov64(5, 8);
        a.add64_imm(5, HEAP_OUT as i64);
        a.call(self.hash_chunk);
        a.mov64_imm(0, 0);
        a.exit();

        a.bind(multi);
        a.mov64(0, 7);
        a.rsh64_imm(0, 10);
        a.stxdw(8, A_NFULL, 0);
        a.mov64_imm(0, 0);
        a.stxdw(8, A_I, 0);
        if self.kernel == Kernel::Simd8 {
            // Groups of eight full chunks
            let top = a.new_label();
            let done = a.new_label();
            a.bind(top);
            a.ldxdw(4, 8, A_I);
            a.ldxdw(0, 8, A_NFULL);
            a.mov64(1, 4);
            a.add64_imm(1, 8);
            a.jgt(1, 0, done);
            a.mov64(1, 4);
            a.lsh64_imm(1, 10);
            a.add64(1, 6); // base = data + 1024 * i
            a.mov64(0, 4);
            a.lsh64_imm(0, 5);
            a.add64(0, 8);
            a.add64_imm(0, HEAP_CVS as i64);
            a.stxdw(8, A_OUT, 0); // out = cvs + 32 * i
            a.mov64_imm(2, CHUNK_LEN as i64);
            a.mov64_imm(3, 16);
            a.mov64_imm(5, 0);
            a.call(self.compress8);
            a.ldxdw(4, 8, A_I);
            a.add64_imm(4, 8);
            a.stxdw(8, A_I, 4);
            a.ja(top);
            a.bind(done);
        }
        // Remaining full chunks, one at a time
        {
            let top = a.new_label();
            let done = a.new_label();
            a.bind(top);
            a.ldxdw(3, 8, A_I);
            a.ldxdw(0, 8, A_NFULL);
            a.jge(3, 0, done);
            a.mov64(1, 3);
            a.lsh64_imm(1, 10);
            a.add64(1, 6);
            a.mov64_imm(2, CHUNK_LEN as i64);
            a.mov64_imm(4, 0);
            a.mov64(5, 3);
            a.lsh64_imm(5, 5);
            a.add64(5, 8);
            a.add64_imm(5, HEAP_CVS as i64);
            a.call(self.hash_chunk);
            a.ldxdw(3, 8, A_I);
            a.add64_imm(3, 1);
            a.stxdw(8, A_I, 3);
            a.ja(top);
            a.bind(done);
        }
        // Partial last chunk
        {
            let done = a.new_label();
            a.mov64(2, 7);
            a.and64_imm(2, CHUNK_LEN as i64 - 1);
            a.jeq_imm(2, 0, done);
            a.ldxdw(3, 8, A_I);
            a.mov64(1, 3);
            a.lsh64_imm(1, 10);
            a.add64(1, 6);
            a.mov64_imm(4, 0);
            a.mov64(5, 3);
            a.lsh64_imm(5, 5);
            a.add64(5, 8);
            a.add64_imm(5, HEAP_CVS as i64);
            a.call(self.hash_chunk);
            a.ldxdw(3, 8, A_I);
            a.add64_imm(3, 1);
            a.stxdw(8, A_I, 3);
            a.bind(done);
        }
        // Tree reduction: pair adjacent nodes left to right, an odd node is carried up
        a.ldxdw(0, 8, A_I);
        a.stxdw(8, A_CNT, 0);
        let level = a.new_label();
        let root = a.new_label();
        a.bind(level);
        a.ldxdw(0, 8, A_CNT);
        a.jle_imm(0, 2, root);
        a.rsh64_imm(0, 1);
        a.stxdw(8, A_PAIRS, 0);
        a.mov64_imm(0, 0);
        a.stxdw(8, A_I, 0);
        if self.kernel == Kernel::Simd8 {
            let top = a.new_label();
            let done = a.new_label();
            a.bind(top);
            a.ldxdw(4, 8, A_I);
            a.ldxdw(0, 8, A_PAIRS);
            a.mov64(1, 4);
            a.add64_imm(1, 8);
            a.jgt(1, 0, done);
            a.mov64(1, 4);
            a.lsh64_imm(1, 6);
            a.add64(1, 8);
            a.add64_imm(1, HEAP_CVS as i64); // base = cvs + 64 * p
            a.mov64(0, 4);
            a.lsh64_imm(0, 5);
            a.add64(0, 8);
            a.add64_imm(0, HEAP_CVS as i64);
            a.stxdw(8, A_OUT, 0); // out = cvs + 32 * p
            a.mov64_imm(2, BLOCK_LEN as i64);
            a.mov64_imm(3, 1);
            a.mov64_imm(4, 0);
            a.mov64_imm(5, 1);
            a.call(self.compress8);
            a.ldxdw(4, 8, A_I);
            a.add64_imm(4, 8);
            a.stxdw(8, A_I, 4);
            a.ja(top);
            a.bind(done);
        }
        {
            let top = a.new_label();
            let done = a.new_label();
            a.bind(top);
            a.ldxdw(3, 8, A_I);
            a.ldxdw(0, 8, A_PAIRS);
            a.jge(3, 0, done);
            a.mov64(1, 3);
            a.lsh64_imm(1, 6);
            a.add64(1, 8);
            a.add64_imm(1, HEAP_CVS as i64);
            a.mov64(2, 3);
            a.lsh64_imm(2, 5);
            a.add64(2, 8);
            a.add64_imm(2, HEAP_CVS as i64);
            a.mov64_imm(3, PARENT);
            a.call(self.parent);
            a.ldxdw(3, 8, A_I);
            a.add64_imm(3, 1);
            a.stxdw(8, A_I, 3);
            a.ja(top);
            a.bind(done);
        }
        {
            // Carry the odd node: cvs[pairs] = cvs[cnt - 1]
            let even = a.new_label();
            a.ldxdw(0, 8, A_CNT);
            a.mov64(1, 0);
            a.and64_imm(1, 1);
            a.jeq_imm(1, 0, even);
            a.sub64_imm(0, 1);
            a.lsh64_imm(0, 5);
            a.add64(0, 8);
            a.ldxdw(2, 8, A_PAIRS);
            a.lsh64_imm(2, 5);
            a.add64(2, 8);
            for i in 0..4 {
                a.ldxdw(3, 0, HEAP_CVS as i16 + 8 * i);
                a.stxdw(2, HEAP_CVS as i16 + 8 * i, 3);
            }
            a.bind(even);
            a.ldxdw(0, 8, A_CNT);
            a.mov64(1, 0);
            a.and64_imm(1, 1);
            a.rsh64_imm(0, 1);
            a.add64(0, 1);
            a.stxdw(8, A_CNT, 0);
            a.ja(level);
        }
        a.bind(root);
        a.mov64(1, 8);
        a.add64_imm(1, HEAP_CVS as i64);
        a.mov64(2, 8);
        a.add64_imm(2, HEAP_OUT as i64);
        a.mov64_imm(3, PARENT | ROOT);
        a.call(self.parent);
        a.mov64_imm(0, 0);
        a.exit();
    }

    /// hash_chunk(r1 = ptr, r2 = len, r3 = counter, r4 = extra flags, r5 = out)
    fn hash_chunk(&mut self) {
        let simd = self.simd();
        let a = &mut self.a;
        a.bind(self.hash_chunk);
        a.lddw(8, ebpf::MM_HEAP_START);
        a.mov64(6, 1);
        a.mov64(7, 2);
        a.stxdw(8, A_CTR, 3);
        a.stxdw(8, A_FLAGS, 4);
        a.stxdw(8, A_OUT, 5);
        if simd {
            a.vmov(S_CV, S_IV);
        } else {
            for i in 0..4 {
                a.ldxdw(0, 8, (HEAP_IV + 8 * i) as i16);
                a.stxdw(5, 8 * i as i16, 0);
            }
            a.stxdw(8, A_CVIN, 5);
        }
        // nblocks = max(1, (len + 63) / 64)
        a.mov64(0, 7);
        a.add64_imm(0, 63);
        a.rsh64_imm(0, 6);
        let nonzero = a.new_label();
        a.jne_imm(0, 0, nonzero);
        a.mov64_imm(0, 1);
        a.bind(nonzero);
        a.stxdw(8, A_NBLOCKS, 0);
        a.mov64_imm(0, 0);
        a.stxdw(8, A_B, 0);
        let top = a.new_label();
        let full = a.new_label();
        let compress = a.new_label();
        a.bind(top);
        // flags
        a.ldxdw(5, 8, A_B);
        a.mov64_imm(4, 0);
        let not_first = a.new_label();
        a.jne_imm(5, 0, not_first);
        a.or64_imm(4, CHUNK_START);
        a.bind(not_first);
        a.ldxdw(0, 8, A_NBLOCKS);
        a.sub64_imm(0, 1);
        let not_last = a.new_label();
        a.jne(5, 0, not_last);
        a.ldxdw(0, 8, A_FLAGS);
        a.or64(4, 0);
        a.or64_imm(4, CHUNK_END);
        a.bind(not_last);
        // block_len = min(64, len - 64 * b), block pointer
        a.lsh64_imm(5, 6);
        a.mov64(3, 7);
        a.sub64(3, 5);
        a.mov64(1, 6);
        a.add64(1, 5);
        a.jge_imm(3, BLOCK_LEN as i64, full);
        // Partial block: copy into a zeroed buffer
        for i in 0..8 {
            a.stdw(8, (HEAP_PAD + 8 * i) as i16, 0);
        }
        {
            let copy = a.new_label();
            let copied = a.new_label();
            a.mov64_imm(2, 0);
            a.bind(copy);
            a.jge(2, 3, copied);
            a.mov64(5, 1);
            a.add64(5, 2);
            a.ldxb(0, 5, 0);
            a.mov64(5, 8);
            a.add64(5, 2);
            a.stxb(5, HEAP_PAD as i16, 0);
            a.add64_imm(2, 1);
            a.ja(copy);
            a.bind(copied);
        }
        a.mov64(1, 8);
        a.add64_imm(1, HEAP_PAD as i64);
        a.ja(compress);
        a.bind(full);
        a.mov64_imm(3, BLOCK_LEN as i64);
        a.bind(compress);
        a.ldxdw(2, 8, A_CTR);
        a.ldxdw(5, 8, A_CVIN);
        a.call(self.compress1);
        a.lddw(8, ebpf::MM_HEAP_START);
        a.ldxdw(5, 8, A_B);
        a.add64_imm(5, 1);
        a.stxdw(8, A_B, 5);
        a.ldxdw(0, 8, A_NBLOCKS);
        a.jlt(5, 0, top);
        if simd {
            a.ldxdw(4, 8, A_OUT);
            a.vstore(4, 0, S_CV);
        }
        a.exit();
    }

    /// parent(r1 = block, r2 = out, r3 = flags)
    fn parent(&mut self) {
        let simd = self.simd();
        let a = &mut self.a;
        a.bind(self.parent);
        a.lddw(8, ebpf::MM_HEAP_START);
        a.stxdw(8, A_OUT, 2);
        a.mov64(4, 3);
        a.mov64_imm(2, 0);
        a.mov64_imm(3, BLOCK_LEN as i64);
        if simd {
            // r1 doesn't survive vector intrinsic calls (r7 is callee saved)
            a.mov64(7, 1);
            a.vmov(S_CV, S_IV);
            a.mov64(1, 7);
            a.mov64(6, 8);
            a.call(self.compress1);
            a.ldxdw(4, 6, A_OUT);
            a.vstore(4, 0, S_CV);
        } else {
            a.mov64(5, 8);
            a.add64_imm(5, HEAP_IV as i64);
            a.call(self.compress1);
        }
        a.exit();
    }

    /// compress1(r1 = block, r2 = counter, r3 = block_len, r4 = flags, r5 = cv_in)
    ///
    /// Scalar: writes the new chaining value to `*A_OUT`. SIMD: updates slot `S_CV` in place.
    fn compress1(&mut self) {
        self.a.bind(self.compress1);
        if self.simd() {
            self.compress1_simd();
        } else {
            self.compress1_scalar();
        }
        self.a.exit();
    }

    fn rot32(a: &mut Asm, reg: u8, n: i64) {
        a.mov32(0, reg);
        a.rsh32_imm(reg, n);
        a.lsh32_imm(0, 32 - n);
        a.or32(reg, 0);
    }

    fn compress1_scalar(&mut self) {
        let a = &mut self.a;
        a.lddw(7, ebpf::MM_HEAP_START + HEAP_STATE as u64);
        a.mov64(6, 1);
        for i in 0..8 {
            a.ldxw(0, 5, 4 * i);
            a.stxw(7, 4 * i, 0);
        }
        for i in 0..4 {
            a.mov32_imm(0, IV[i] as i32 as i64);
            a.stxw(7, 32 + 4 * i as i16, 0);
        }
        a.stxw(7, 48, 2);
        a.rsh64_imm(2, 32);
        a.stxw(7, 52, 2);
        a.stxw(7, 56, 3);
        a.stxw(7, 60, 4);
        let schedule = msg_schedule();
        const G: [[usize; 4]; 8] = [
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
            [0, 5, 10, 15],
            [1, 6, 11, 12],
            [2, 7, 8, 13],
            [3, 4, 9, 14],
        ];
        for s in schedule.iter() {
            for (k, [ia, ib, ic, id]) in G.iter().enumerate() {
                let (ia, ib, ic, id) = (*ia as i16, *ib as i16, *ic as i16, *id as i16);
                a.ldxw(1, 7, 4 * ia);
                a.ldxw(2, 7, 4 * ib);
                a.ldxw(3, 7, 4 * ic);
                a.ldxw(4, 7, 4 * id);
                a.ldxw(5, 6, 4 * s[2 * k] as i16);
                a.add32(1, 2);
                a.add32(1, 5);
                a.xor32(4, 1);
                Self::rot32(a, 4, 16);
                a.add32(3, 4);
                a.xor32(2, 3);
                Self::rot32(a, 2, 12);
                a.ldxw(5, 6, 4 * s[2 * k + 1] as i16);
                a.add32(1, 2);
                a.add32(1, 5);
                a.xor32(4, 1);
                Self::rot32(a, 4, 8);
                a.add32(3, 4);
                a.xor32(2, 3);
                Self::rot32(a, 2, 7);
                a.stxw(7, 4 * ia, 1);
                a.stxw(7, 4 * ib, 2);
                a.stxw(7, 4 * ic, 3);
                a.stxw(7, 4 * id, 4);
            }
        }
        a.lddw(8, ebpf::MM_HEAP_START);
        a.ldxdw(5, 8, A_OUT);
        for i in 0..8 {
            a.ldxw(1, 7, 4 * i);
            a.ldxw(2, 7, 32 + 4 * i);
            a.xor32(1, 2);
            a.stxw(5, 4 * i, 1);
        }
    }

    fn g4(a: &mut Asm, va: u8, vb: u8, vc: u8, vd: u8, mx: u8, my: u8) {
        a.vadd(va, va, vb);
        a.vadd(va, va, mx);
        a.vxor(vd, vd, va);
        a.vror(vd, vd, 16);
        a.vadd(vc, vc, vd);
        a.vxor(vb, vb, vc);
        a.vror(vb, vb, 12);
        a.vadd(va, va, vb);
        a.vadd(va, va, my);
        a.vxor(vd, vd, va);
        a.vror(vd, vd, 8);
        a.vadd(vc, vc, vd);
        a.vxor(vb, vb, vc);
        a.vror(vb, vb, 7);
    }

    fn compress1_simd(&mut self) {
        let a = &mut self.a;
        let [r0, r1, r2, r3] = [S_ROW, S_ROW + 1, S_ROW + 2, S_ROW + 3];
        let [t1, t2, t3, t4] = [S_T, S_T + 1, S_T + 2, S_T + 3];
        // r1 (block) and r2 (counter) don't survive vector intrinsic calls with a scalar operand
        a.mov64(5, 1);
        a.vmov(r0, S_CV);
        a.vpermd(r1, S_CV, S_HI4);
        a.vmov(r2, S_IV_LO);
        a.vbcast(r3, 2);
        a.rsh64_imm(2, 32);
        a.vbcast(t1, 2);
        a.vblendd(r3, r3, t1, 0x22);
        a.vbcast(t1, 3);
        a.vblendd(r3, r3, t1, 0x44);
        a.vbcast(t1, 4);
        a.vblendd(r3, r3, t1, 0x88);
        a.vload(S_ML, 5, 0);
        a.vload(S_MH, 5, 32);
        let (mask_lo, mask_hi) = permutation_masks();
        for round in 0..7 {
            a.vpermd(t1, S_ML, S_EVEN);
            a.vpermd(t2, S_ML, S_ODD);
            Self::g4(a, r0, r1, r2, r3, t1, t2);
            // Diagonalize: lane i then holds (a[i], b[i + 1], c[i + 2], d[i + 3])
            a.vshufd(r1, r1, 0x39);
            a.vshufd(r2, r2, 0x4e);
            a.vshufd(r3, r3, 0x93);
            a.vpermd(t1, S_MH, S_EVEN);
            a.vpermd(t2, S_MH, S_ODD);
            Self::g4(a, r0, r1, r2, r3, t1, t2);
            a.vshufd(r1, r1, 0x93);
            a.vshufd(r2, r2, 0x4e);
            a.vshufd(r3, r3, 0x39);
            if round < 6 {
                a.vpermd(t1, S_ML, S_PLO);
                a.vpermd(t2, S_MH, S_PLO);
                a.vpermd(t3, S_ML, S_PHI);
                a.vpermd(t4, S_MH, S_PHI);
                a.vblendd(S_ML, t1, t2, mask_lo);
                a.vblendd(S_MH, t3, t4, mask_hi);
            }
        }
        a.vxor(t1, r0, r2);
        a.vxor(t2, r1, r3);
        a.vpermd(t2, t2, S_DUP_LO);
        a.vblendd(S_CV, t1, t2, 0xf0);
    }

    /// compress8(r1 = base, r2 = stride, r3 = nblocks, r4 = counter, r5 = parent), out at A_OUT
    fn compress8(&mut self) {
        let a = &mut self.a;
        a.bind(self.compress8);
        a.lddw(8, ebpf::MM_HEAP_START);
        a.stxdw(8, A_BASE, 1);
        a.stxdw(8, A_STRIDE, 2);
        a.stxdw(8, A_NBLOCKS, 3);
        a.stxdw(8, A_PARENT, 5);
        for (i, word) in IV.iter().enumerate() {
            a.mov32_imm(2, *word as i32 as i64);
            a.vbcast(S_CV8 + i as u8, 2);
        }
        a.vbcast(S_CTR_LO, 4);
        let is_parent = a.new_label();
        // r0 and r5 don't survive vector intrinsic calls, reload the argument
        a.ldxdw(0, 8, A_PARENT);
        a.jne_imm(0, 0, is_parent);
        a.vadd(S_CTR_LO, S_CTR_LO, S_LANE);
        a.bind(is_parent);
        a.rsh64_imm(4, 32);
        a.vbcast(S_CTR_HI, 4);
        a.mov64_imm(0, 0);
        a.stxdw(8, A_B, 0);
        let top = a.new_label();
        a.bind(top);
        // flags
        let flags_done = a.new_label();
        a.ldxdw(5, 8, A_B);
        a.ldxdw(0, 8, A_PARENT);
        a.mov64_imm(4, PARENT);
        a.jne_imm(0, 0, flags_done);
        a.mov64_imm(4, 0);
        let not_first = a.new_label();
        a.jne_imm(5, 0, not_first);
        a.or64_imm(4, CHUNK_START);
        a.bind(not_first);
        a.ldxdw(0, 8, A_NBLOCKS);
        a.sub64_imm(0, 1);
        a.jne(5, 0, flags_done);
        a.or64_imm(4, CHUNK_END);
        a.bind(flags_done);
        // Load and transpose the message blocks of the eight lanes
        a.ldxdw(2, 8, A_BASE);
        a.lsh64_imm(5, 6);
        a.add64(2, 5);
        a.ldxdw(5, 8, A_STRIDE);
        for lane in 0..8u8 {
            a.vload(S_MSG + lane, 2, 0);
            a.vload(S_MSG + 8 + lane, 2, 32);
            if lane < 7 {
                a.add64(2, 5);
            }
        }
        a.vtranspose(S_MSG);
        a.vtranspose(S_MSG + 8);
        // Initialize the state
        for i in 0..8u8 {
            a.vmov(S_STATE + i, S_CV8 + i);
        }
        for i in 0..4u8 {
            a.mov32_imm(2, IV[i as usize] as i32 as i64);
            a.vbcast(S_STATE + 8 + i, 2);
        }
        a.vmov(S_STATE + 12, S_CTR_LO);
        a.vmov(S_STATE + 13, S_CTR_HI);
        a.mov64_imm(2, BLOCK_LEN as i64);
        a.vbcast(S_STATE + 14, 2);
        a.vbcast(S_STATE + 15, 4);
        // Seven rounds, the four G functions of each half round interleaved like the official AVX2
        // implementation
        let schedule = msg_schedule();
        let v = |i: usize| S_STATE + i as u8;
        let m = |s: &[usize; 16], i: usize| S_MSG + s[i] as u8;
        for s in schedule.iter() {
            for half in 0..2 {
                let cols: [[usize; 4]; 4] = if half == 0 {
                    [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
                } else {
                    [[0, 5, 10, 15], [1, 6, 11, 12], [2, 7, 8, 13], [3, 4, 9, 14]]
                };
                let mi = |k: usize, second: usize| m(s, 8 * half + 2 * k + second);
                for k in 0..4 {
                    a.vadd(v(cols[k][0]), v(cols[k][0]), mi(k, 0));
                }
                for k in 0..4 {
                    a.vadd(v(cols[k][0]), v(cols[k][0]), v(cols[k][1]));
                }
                for k in 0..4 {
                    a.vxor(v(cols[k][3]), v(cols[k][3]), v(cols[k][0]));
                }
                for k in 0..4 {
                    a.vror(v(cols[k][3]), v(cols[k][3]), 16);
                }
                for k in 0..4 {
                    a.vadd(v(cols[k][2]), v(cols[k][2]), v(cols[k][3]));
                }
                for k in 0..4 {
                    a.vxor(v(cols[k][1]), v(cols[k][1]), v(cols[k][2]));
                }
                for k in 0..4 {
                    a.vror(v(cols[k][1]), v(cols[k][1]), 12);
                }
                for k in 0..4 {
                    a.vadd(v(cols[k][0]), v(cols[k][0]), mi(k, 1));
                }
                for k in 0..4 {
                    a.vadd(v(cols[k][0]), v(cols[k][0]), v(cols[k][1]));
                }
                for k in 0..4 {
                    a.vxor(v(cols[k][3]), v(cols[k][3]), v(cols[k][0]));
                }
                for k in 0..4 {
                    a.vror(v(cols[k][3]), v(cols[k][3]), 8);
                }
                for k in 0..4 {
                    a.vadd(v(cols[k][2]), v(cols[k][2]), v(cols[k][3]));
                }
                for k in 0..4 {
                    a.vxor(v(cols[k][1]), v(cols[k][1]), v(cols[k][2]));
                }
                for k in 0..4 {
                    a.vror(v(cols[k][1]), v(cols[k][1]), 7);
                }
            }
        }
        for i in 0..8u8 {
            a.vxor(S_CV8 + i, S_STATE + i, S_STATE + 8 + i);
        }
        a.ldxdw(5, 8, A_B);
        a.add64_imm(5, 1);
        a.stxdw(8, A_B, 5);
        a.ldxdw(0, 8, A_NBLOCKS);
        a.jlt(5, 0, top);
        // Transpose back into one chaining value per lane and store them
        a.vtranspose(S_CV8);
        a.ldxdw(2, 8, A_OUT);
        for lane in 0..8u8 {
            a.vstore(2, 32 * lane as i16, S_CV8 + lane);
        }
        a.exit();
    }

    fn finish(mut self) -> Vec<u8> {
        self.hash_chunk();
        self.parent();
        self.compress1();
        if self.kernel == Kernel::Simd8 {
            self.compress8();
        }
        self.a.finish()
    }
}

/// Generates the full BLAKE3 hash program
pub fn hash_program(kernel: Kernel, mode: VecMode) -> Vec<u8> {
    let mut g = Gen::new(kernel, mode);
    g.main();
    g.finish()
}

/// Generates a program which runs the compression kernel on the first block(s) of the data
/// `iterations` times (read from `A_ITER`-style input field at offset 8 of the input).
///
/// Simd8 compresses eight independent 64 byte blocks (data must be at least 512 bytes), the other
/// kernels compress one 64 byte block per iteration. The final chaining value(s) are written to
/// [HEAP_CVS].
pub fn compress_program(kernel: Kernel, mode: VecMode) -> Vec<u8> {
    let mut g = Gen::new(kernel, mode);
    g.prologue();
    {
        let simd = g.simd();
        let a = &mut g.a;
        a.lddw(8, ebpf::MM_HEAP_START);
        a.ldxdw(0, 6, 8);
        a.stxdw(8, A_ITER, 0);
        a.add64_imm(6, INPUT_DATA as i64);
        if simd {
            a.vmov(S_CV, S_IV);
        } else {
            for i in 0..4 {
                a.ldxdw(0, 8, (HEAP_IV + 8 * i) as i16);
                a.stxdw(8, (HEAP_CVS + 8 * i) as i16, 0);
            }
        }
        let top = a.new_label();
        a.bind(top);
        match g.kernel {
            Kernel::Simd8 => {
                a.mov64(1, 6);
                a.mov64_imm(2, BLOCK_LEN as i64);
                a.mov64_imm(3, 1);
                a.mov64_imm(4, 0);
                a.mov64_imm(5, 1);
                a.mov64(0, 8);
                a.add64_imm(0, HEAP_CVS as i64);
                a.stxdw(8, A_OUT, 0);
                a.call(g.compress8);
            }
            _ => {
                a.mov64(1, 6);
                a.mov64_imm(2, 0);
                a.mov64_imm(3, BLOCK_LEN as i64);
                a.mov64_imm(4, CHUNK_START | CHUNK_END);
                a.mov64(5, 8);
                a.add64_imm(5, HEAP_CVS as i64);
                a.stxdw(8, A_OUT, 5);
                a.call(g.compress1);
            }
        }
        a.lddw(8, ebpf::MM_HEAP_START);
        a.ldxdw(0, 8, A_ITER);
        a.sub64_imm(0, 1);
        a.stxdw(8, A_ITER, 0);
        a.jne_imm(0, 0, top);
        if simd && g.kernel != Kernel::Simd8 {
            a.mov64(1, 8);
            a.add64_imm(1, HEAP_CVS as i64);
            a.vstore(1, 0, S_CV);
        }
        a.mov64_imm(0, 0);
        a.exit();
    }
    g.compress1();
    if g.kernel == Kernel::Simd8 {
        g.compress8();
    }
    g.a.finish()
}

/// Generates the register caching microbenchmark:
/// `v0 += v1; v0 ^= v2; v0 += v3; v0 ^= v4` repeated `repeat` times in one basic block, executed
/// `iterations` times (input offset 8). v0..v4 are loaded from the input constants area, the result
/// is stored to [HEAP_OUT].
pub fn microbench_program(repeat: usize, mode: VecMode) -> Vec<u8> {
    let mut a = Asm::new(mode);
    a.lddw(
        VECTOR_FILE_REG,
        ebpf::MM_HEAP_START + HEAP_VECTOR_FILE as u64,
    );
    a.mov64(3, 1);
    for i in 0..5u8 {
        a.vload(i, 3, (INPUT_CONSTANTS + 32 * i as usize) as i16);
    }
    a.ldxdw(2, 3, 8);
    let top = a.new_label();
    a.bind(top);
    for _ in 0..repeat {
        a.vadd(0, 0, 1);
        a.vxor(0, 0, 2);
        a.vadd(0, 0, 3);
        a.vxor(0, 0, 4);
    }
    a.sub64_imm(2, 1);
    a.jne_imm(2, 0, top);
    a.lddw(1, ebpf::MM_HEAP_START + HEAP_OUT as u64);
    a.vstore(1, 0, 0);
    a.mov64_imm(0, 0);
    a.exit();
    a.finish()
}

/// Reference result of [microbench_program]
pub fn microbench_reference(input: &[u8], repeat: usize, iterations: u64) -> [u32; 8] {
    let lanes = |i: usize| super::read_lanes(&input[INPUT_CONSTANTS + 32 * i..]);
    let mut v = [lanes(0), lanes(1), lanes(2), lanes(3), lanes(4)];
    for _ in 0..iterations * repeat as u64 {
        for i in 0..8 {
            v[0][i] = v[0][i].wrapping_add(v[1][i]);
            v[0][i] ^= v[2][i];
            v[0][i] = v[0][i].wrapping_add(v[3][i]);
            v[0][i] ^= v[4][i];
        }
    }
    v[0]
}
