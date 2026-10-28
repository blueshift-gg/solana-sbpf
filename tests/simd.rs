#![cfg(feature = "simd-experiment")]
//! Tests of the experimental SIMD extension

use std::convert::TryInto;
use solana_sbpf::{
    simd::{
        asm::VecMode,
        blake3_program::{self, Kernel, HEAP_OUT, HEAP_SIZE},
        harness::{config, Engine, Runner},
        VADD16, VMADD16, VMINU16, VMULHI16, VMULLO16, VROLV32, VSUB16,
    },
    vm::SimdJitStrategy,
};

fn test_data(len: usize, seed: u64) -> Vec<u8> {
    // Official BLAKE3 test vector input pattern for seed 0, deterministic pseudo random otherwise
    let mut state = seed;
    (0..len)
        .map(|i| {
            if seed == 0 {
                (i % 251) as u8
            } else {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 56) as u8
            }
        })
        .collect()
}

const SIZES: &[usize] = &[
    0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 4096, 8192, 8193, 16384, 65536,
];

fn hash_with(kernel: Kernel, mode: VecMode, engine: Engine, strategy: SimdJitStrategy, data: &[u8]) -> ([u8; 32], u64) {
    let program = blake3_program::hash_program(kernel, mode);
    let mut runner = Runner::new(&program, &blake3_program::build_input(data), HEAP_SIZE, config(strategy)).unwrap();
    if engine == Engine::Jit {
        runner.jit_compile().unwrap();
    }
    let (cus, result) = runner.run(engine);
    assert!(result.is_ok(), "{:?} {:?} {:?} len={} {:?}", kernel, mode, engine, data.len(), result);
    (runner.heap()[HEAP_OUT..HEAP_OUT + 32].try_into().unwrap(), cus)
}

#[test]
fn blake3_interpreter_all_kernels() {
    for &len in SIZES {
        for seed in [0, 7] {
            let data = test_data(len, seed);
            let expected = *blake3::hash(&data).as_bytes();
            for kernel in [Kernel::Scalar, Kernel::Simd1, Kernel::Simd8] {
                for mode in [VecMode::Slot, VecMode::Call] {
                    if kernel == Kernel::Scalar && mode != VecMode::Slot {
                        continue;
                    }
                    let (digest, _) = hash_with(kernel, mode, Engine::Interpreter, SimdJitStrategy::Direct, &data);
                    assert_eq!(digest, expected, "{kernel:?} {mode:?} len={len} seed={seed}");
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn blake3_jit_matches_interpreter() {
    for &len in SIZES {
        let data = test_data(len, 3);
        let expected = *blake3::hash(&data).as_bytes();
        for kernel in [Kernel::Scalar, Kernel::Simd1, Kernel::Simd8] {
            for mode in [VecMode::Slot, VecMode::Call] {
                if kernel == Kernel::Scalar && mode != VecMode::Slot {
                    continue;
                }
                let (_, interp_cus) = hash_with(kernel, mode, Engine::Interpreter, SimdJitStrategy::Direct, &data);
                for strategy in [SimdJitStrategy::Direct, SimdJitStrategy::RegisterCached] {
                    let (digest, cus) = hash_with(kernel, mode, Engine::Jit, strategy, &data);
                    assert_eq!(digest, expected, "{kernel:?} {mode:?} {strategy:?} len={len}");
                    assert_eq!(cus, interp_cus, "CU mismatch {kernel:?} {mode:?} {strategy:?} len={len}");
                }
            }
        }
    }
}

/// Loads 16 random slots from the input, runs `body`, stores all 16 slots to the heap
fn op_program(mode: VecMode, body: &dyn Fn(&mut solana_sbpf::simd::asm::Asm)) -> Vec<u8> {
    let mut a = solana_sbpf::simd::asm::Asm::new(mode);
    a.lddw(9, solana_sbpf::ebpf::MM_HEAP_START + 0x1000);
    a.mov64(3, 1);
    for i in 0..16u8 {
        a.vload(i, 3, 32 * i as i16);
    }
    body(&mut a);
    a.lddw(2, solana_sbpf::ebpf::MM_HEAP_START);
    for i in 0..16u8 {
        a.vstore(2, 32 * i as i16, i);
    }
    a.mov64_imm(0, 0);
    a.exit();
    a.finish()
}

fn random_input(seed: u64) -> Vec<u8> {
    let mut data = test_data(16 * 32, seed + 1);
    // Some edge values for wrapping / rotation
    data[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
    data[32..36].copy_from_slice(&1u32.to_le_bytes());
    data[4..8].copy_from_slice(&0x8000_0000u32.to_le_bytes());
    data
}

fn run_op(program: &[u8], input: &[u8], engine: Engine, strategy: SimdJitStrategy) -> (Vec<u8>, u64) {
    let mut runner = Runner::new(program, input, 0x2000, config(strategy)).unwrap();
    if engine == Engine::Jit {
        runner.jit_compile().unwrap();
    }
    let (cus, result) = runner.run(engine);
    assert!(result.is_ok(), "{:?} {:?} {:?}", engine, strategy, result);
    (runner.heap()[..16 * 32].to_vec(), cus)
}

type Body = Box<dyn Fn(&mut solana_sbpf::simd::asm::Asm)>;

fn op_cases() -> Vec<(&'static str, Body)> {
    let mut cases: Vec<(&'static str, Body)> = vec![
        ("nop", Box::new(|_a| {})),
        ("vadd", Box::new(|a| a.vadd(0, 0, 1))),
        ("vadd3", Box::new(|a| a.vadd(2, 3, 4))),
        ("vadd_same", Box::new(|a| a.vadd(5, 5, 5))),
        ("vxor", Box::new(|a| a.vxor(0, 1, 2))),
        ("vand", Box::new(|a| a.vand(3, 1, 2))),
        ("vor", Box::new(|a| a.vor(3, 3, 2))),
        ("vshufd", Box::new(|a| a.vshufd(1, 2, 0x39))),
        ("vshufd_inplace", Box::new(|a| a.vshufd(2, 2, 0x93))),
        ("vpermd", Box::new(|a| a.vpermd(1, 2, 3))),
        ("vpermd_alias", Box::new(|a| a.vpermd(3, 2, 3))),
        ("vblendd", Box::new(|a| a.vblendd(4, 5, 6, 0xa5))),
        ("vmov", Box::new(|a| a.vmov(7, 8))),
        ("vtranspose", Box::new(|a| a.vtranspose(3))),
        ("vsub", Box::new(|a| a.vsub(0, 1, 2))),
        ("vsub_same", Box::new(|a| a.vsub(3, 3, 3))),
        ("vmullo", Box::new(|a| a.vmullo(4, 5, 6))),
        ("vmullo_square", Box::new(|a| a.vmullo(7, 7, 7))),
        ("vminu", Box::new(|a| a.vminu(8, 0, 9))),
        ("vsrl", Box::new(|a| a.vsrl(1, 2, 13))),
        ("vsll_inplace", Box::new(|a| a.vsll(2, 2, 31))),
        ("vsllv", Box::new(|a| a.vsllv(1, 2, 3))),
        ("vsrlv", Box::new(|a| a.vsrlv(4, 5, 6))),
        ("vandn", Box::new(|a| a.vandn(7, 8, 9))),
        ("vandn_alias", Box::new(|a| a.vandn(9, 9, 10))),
        ("vbcast", Box::new(|a| {
            a.mov64_imm(3, 0x1234_5678);
            a.vbcast(9, 3);
        })),
        ("chain", Box::new(|a| {
            for i in 0..40u8 {
                a.vadd(i % 16, (i + 1) % 16, (i + 5) % 16);
                a.vxor((i + 3) % 16, i % 16, (i + 7) % 16);
                a.vror((i + 2) % 16, (i + 3) % 16, (i % 31) as i16 + 1);
            }
        })),
    ];
    cases.push(("vrolv", Box::new(|a| a.vrolv(1, 2, 3))));
    cases.push(("vrolv_small", Box::new(|a| {
        a.vsrl(3, 3, 26); // counts 0..=63, covering 32
        a.vrolv(1, 2, 3);
        a.vrolv(4, 4, 3);
        a.vrolv(5, 6, 5);
    })));
    for (name, op) in [
        ("vadd16", VADD16), ("vsub16", VSUB16), ("vmullo16", VMULLO16), ("vmulhi16", VMULHI16),
        ("vminu16", VMINU16), ("vmadd16", VMADD16),
    ] {
        cases.push((name, Box::new(move |a| {
            let f = |a: &mut solana_sbpf::simd::asm::Asm, d, x, y| match op {
                VADD16 => a.vadd16(d, x, y),
                VSUB16 => a.vsub16(d, x, y),
                VMULLO16 => a.vmullo16(d, x, y),
                VMULHI16 => a.vmulhi16(d, x, y),
                VMINU16 => a.vminu16(d, x, y),
                _ => a.vmadd16(d, x, y),
            };
            f(a, 1, 2, 3);
            f(a, 4, 4, 5);
            f(a, 6, 7, 6);
            f(a, 8, 8, 8);
        })));
    }
    cases.push(("vblendw", Box::new(|a| {
        a.vblendw(1, 2, 3, 0xa6);
        a.vblendw(4, 4, 5, 0x0f);
    })));
    for imm in [0x00i16, 0xff, 0x96, 0xd2, 0x78, 0xe8, 0xca, 0x01, 0x5a, 0x3c, 0xf0, 0xaa, 0xcc, 0x1e, 0xb4, 0x6b, 0x0f, 0x55, 0x33] {
        cases.push(("vternlog", Box::new(move |a| {
            a.vternlog(1, 2, 3, imm);
            a.vternlog(4, 4, 5, imm);
            a.vternlog(6, 7, 6, imm);
            a.vternlog(8, 9, 9, imm);
            a.vternlog(10, 10, 10, imm);
        })));
    }
    for amount in [0i16, 1, 7, 8, 12, 16, 31] {
        cases.push(("vror", Box::new(move |a| a.vror(2, 3, amount))));
        cases.push(("vror_inplace", Box::new(move |a| a.vror(3, 3, amount))));
    }
    cases.push(("lane_moves", Box::new(|a| {
        for lane in 0..4 {
            a.vext64(4, 1 + lane as u8, lane);
            a.vins64(5 + lane as u8, 4, 3 - lane);
        }
        a.vadd(9, 9, 10); // cached source for vext64
        a.vext64(6, 9, 2);
        a.vins64(9, 6, 0);
        a.vbcast64(10, 6);
        a.vmovmsk8(7, 11, false);
        a.vmovmsk8(8, 12, true);
        a.vbcast64(13, 7);
        a.vbcast64(14, 8);
    })));
    cases.push(("vpermi2d", Box::new(|a| {
        a.vpermi2d(0, 1, 2, false);
        a.vpermi2d(3, 3, 4, false); // table aliases the index
        a.vpermi2d(5, 6, 6, false);
    })));
    cases.push(("wide_cross", Box::new(|a| {
        a.wide(solana_sbpf::simd::VPERMD, 0, 2, 4, 0, 0);
        a.wide(solana_sbpf::simd::VPERMD, 6, 6, 8, 0, 0); // data aliases the destination
        a.vpermi2d(10, 12, 14, true);
        a.vpermi2d(2, 2, 4, true); // first table aliases the index
        a.vpermi2d(4, 6, 4, true); // second table aliases the index
    })));
    for subop in [
        solana_sbpf::simd::VADD32, solana_sbpf::simd::VXOR, solana_sbpf::simd::VROR32, solana_sbpf::simd::VSHUFD,
        solana_sbpf::simd::VBLENDD, solana_sbpf::simd::VROLV32, solana_sbpf::simd::VTERNLOG, solana_sbpf::simd::VMULHI16,
        solana_sbpf::simd::VBLENDW, solana_sbpf::simd::VMOV, solana_sbpf::simd::VMADD16,
    ] {
        cases.push(("wide", Box::new(move |a| {
            let off = match subop {
                solana_sbpf::simd::VROR32 => 7,
                solana_sbpf::simd::VSHUFD | solana_sbpf::simd::VBLENDD | solana_sbpf::simd::VBLENDW => 0x93,
                solana_sbpf::simd::VTERNLOG => 0xd2,
                _ => 0,
            };
            a.wide(subop, 0, 2, 4, 0, off);
            a.wide(subop, 6, 6, 8, 0, off);
            a.wide(subop, 10, 12, 10, 0, off);
            a.wide(subop, 14, 14, 14, 0, off);
        })));
    }
    cases.push(("rolv64", Box::new(|a| {
        use solana_sbpf::simd::VROLV64;
        a.vop_full(VROLV64, false, 1, 2, 3, 0, 0, 0);
        a.vop_full(VROLV64, false, 4, 4, 5, 0, 0, 0);
        a.vsrl(6, 6, 25); // small counts, 0..=127 covering 64
        a.vop_full(VROLV64, false, 7, 8, 6, 0, 0, 0);
        a.wide(VROLV64, 8, 10, 6, 0, 0);
        a.wide(VROLV64, 12, 12, 12, 0, 0);
    })));
    cases.push(("transpose64", Box::new(|a| {
        use solana_sbpf::simd::VTRANSPOSE64;
        a.vadd(0, 0, 1); // cached, dirty
        a.vop_full(VTRANSPOSE64, false, 1, 0, 0, 0, 0, 0);
        a.wide(VTRANSPOSE64, 0, 0, 0, 0, 0);
        a.vxor(3, 3, 4);
    })));
    cases.push(("keccak_like", Box::new(|a| {
        use solana_sbpf::simd::{VPERMD, VROLV64, VTERNLOG, VTRANSPOSE64, VXOR};
        // The op mix of the 64 bit Keccak: 512 bit xor / ternlog / permd / rolv64 and the transpose
        a.vsrl(14, 14, 26);
        for round in 0..3 {
            a.wide(VXOR, 8, 0, 2, 0, 0);
            a.wide(VTERNLOG, 8, 4, 6, 0, 0x96);
            a.wide(VPERMD, 10, 8, 14, 0, 0);
            a.wide(VROLV64, 10, 10, 14, 0, 0);
            for y in 0..4u8 {
                a.wide(VTERNLOG, 2 * y, 8, 10, 0, 0x96);
                a.wide(VROLV64, 2 * y, 2 * y, 14, 0, 0);
            }
            a.wide(VTERNLOG, 0, 2, 4, 0, 0xd2);
            if round != 1 {
                a.wide(VTRANSPOSE64, 0, 0, 0, 0, 0);
            }
            a.vadd(3, 3, 5); // narrow op on a slot of a pair
        }
    })));
    cases.push(("shufb_permt2d", Box::new(|a| {
        use solana_sbpf::simd::{VPERMT2D, VSHUFB};
        a.vop_full(VSHUFB, false, 1, 2, 3, 0, 0, 0);
        a.vop_full(VSHUFB, false, 4, 4, 4, 0, 0, 0);
        a.wide(VSHUFB, 6, 8, 10, 0, 0);
        a.vop_full(VPERMT2D, false, 12, 13, 14, 0, 0, 0);
        a.vop_full(VPERMT2D, false, 15, 15, 12, 0, 0, 0); // index aliases the table
        a.wide(VPERMT2D, 0, 2, 4, 0, 0);
        a.wide(VPERMT2D, 2, 2, 6, 0, 0);
    })));
    for wide in [false, true] {
        cases.push(("compresslt16", Box::new(move |a| {
            use solana_sbpf::simd::VCOMPRESSLT16;
            a.vsrl(4, 4, 1); // thresholds of about half the range
            a.vsrl(5, 5, 1);
            a.vop_full(VCOMPRESSLT16, wide, 0, 2, 4, 6, 0, 0);
            a.vins64(15, 6, 0);
            a.vop_full(VCOMPRESSLT16, wide, 8, 8, 4, 7, 0, 0); // destination aliases the data
            a.vins64(15, 7, 1);
            a.vop_full(VCOMPRESSLT16, wide, 10, 12, 10, 3, 0, 0); // destination aliases the thresholds
            a.vins64(15, 3, 2);
            a.vadd(1, 1, 0); // the result used by a cached op
        })));
    }
    for wide in [false, true] {
        // Rosetta 2 shifts by the low bits of a 64 bit count in vpsllvq / vpsrlvq instead of giving 0
        // for counts above 63, so there the variable counts stay in range
        let in_range = solana_sbpf::simd::harness::running_under_rosetta();
        cases.push(("int64", Box::new(move |a| {
            use solana_sbpf::simd::*;
            let (d, x, y) = if wide { (0, 2, 4) } else { (0, 1, 2) };
            a.vsrl(14, 14, 26); // small shift counts
            if in_range {
                a.mov64_imm(3, 63);
                a.vbcast64(13, 3);
                a.vand(14, 14, 13);
                a.vand(y, y, 13);
                if wide {
                    a.vand(y + 1, y + 1, 13);
                }
            }
            for subop in [VADD64, VSUB64, VSLLV64, VSRLV64, VMULU32, VMULLO64] {
                a.vop_full(subop, wide, d, x, y, 0, 0, 0);
                a.vop_full(subop, wide, 6, 6, 14, 0, 0, 0);
            }
            for (subop, n) in [(VSLL64, 13), (VSRL64, 63), (VSRA64, 7), (VSRA64, 0)] {
                a.vop_full(subop, wide, 8, x, 0, 0, 0, n);
            }
            a.vop_full(VMADD52LO, wide, 10, x, y, 0, 0, 0);
            a.vop_full(VMADD52HI, wide, 12, 12, y, 0, 0, 0);
        })));
        cases.push(("masks", Box::new(move |a| {
            use solana_sbpf::simd::*;
            for size in 0..4i16 {
                for (k, pred) in [(0, 1i16), (1, 6), (2, 0 | 8), (3, 2 | 8), (4, 4)].iter() {
                    a.vop_full(VCMP, wide, 0, 2, 4, 6, 0, pred | size << 4);
                    a.vins64(15, 6, (*k % 4) as i16);
                }
                a.lddw(3, 0xa5c3_0ff0_6996_1248);
                a.vop_full(VBLENDM, wide, 8, 2, 4, 0, 3, size);
                a.vop_full(VCOMPRESS, wide, 10, 2, 0, 7, 3, size);
                a.vins64(14, 7, size);
                a.vop_full(VEXPAND, wide, 12, 2, 0, 0, 3, size);
                a.vop_full(VCOMPRESS, wide, 4, 4, 0, 7, 3, size); // in place
                a.vop_full(VBLENDM, wide, 2, 2, 2, 0, 3, size);
            }
        })));
    }
    for wide in [false, true] {
        cases.push(("mont16", Box::new(move |a| {
            use solana_sbpf::simd::VMONT16;
            a.vop_full(VMONT16, wide, 0, 2, 4, 0, 0, 8); // w = v4, w' = v6, q = v8
            a.vop_full(VMONT16, wide, 10, 10, 4, 0, 0, 8); // in place
            a.vop_full(VMONT16, wide, 4, 12, 4, 0, 0, 12); // destination is w, q aliases a
            a.vadd(1, 1, 0);
        })));
    }
    for wide in [false, true] {
        cases.push(("sra64_all_counts", Box::new(move |a| {
            for n in [0i16, 1, 7, 31, 32, 33, 47, 63] {
                a.vop_full(solana_sbpf::simd::VSRA64, wide, 2 * (n as u8 % 4), 8, 0, 0, 0, n);
            }
            for size in [2i16, 3] {
                a.lddw(3, 0x0000_0000_0000_00b5);
                a.vop_full(solana_sbpf::simd::VCOMPRESS, false, 9, 10, 0, 4, 3, size);
                a.vop_full(solana_sbpf::simd::VEXPAND, false, 11, 10, 0, 0, 3, size);
                a.vins64(13, 4, size - 2);
            }
        })));
    }
    for wide in [false, true] {
        cases.push(("ext_ops", Box::new(move |a| {
            use solana_sbpf::simd::*;
            let (d, x, y) = if wide { (0, 2, 4) } else { (0, 1, 2) };
            for (subop, off) in [(VSLL16, 3i16), (VSRL16, 15), (VSRA16, 7), (VSRA32, 31), (VROR64, 13), (VPOPCNT, 0), (VPOPCNT, 1),
                (VPOPCNT, 2), (VPOPCNT, 3), (VLZCNT, 2), (VLZCNT, 3), (VZEXT, 1 << 2), (VSEXT, 3 << 2), (VZEXT, 1 | 3 << 2), (VSEXT, 2 | 3 << 2)] {
                a.vop_full(subop, wide, d, x, 0, 0, 0, off);
                a.vop_full(subop, wide, 6, 6, 0, 0, 0, off);
            }
            for (subop, off) in [(VSLLV16, 0i16), (VSRLV16, 0), (VSRAV16, 0), (VSRAV32, 0), (VUNPCKLO, 1), (VUNPCKHI, 3), (VPACKSS, 0),
                (VPACKUS, 1), (VPERMB, 0), (VMULTISHIFT, 0), (VGF2P8AFFINE, 0x63), (VGF2P8MUL, 0), (VAESENC, 0), (VAESENCLAST, 0),
                (VAESDEC, 0), (VAESDECLAST, 0), (VCLMUL, 0x11), (VPERMI2B, 0)] {
                a.vop_full(subop, wide, d, x, y, 0, 0, off);
                a.vop_full(subop, wide, 8, 8, y, 0, 0, off);
            }
            if !solana_sbpf::simd::harness::running_under_rosetta() {
                a.vop_full(VSRAV64, wide, 10, x, y, 0, 0, 0);
            }
        })));
    }
    cases.push(("wide_bcast", Box::new(|a| {
        a.lddw(3, 0x8765_4321_0fed_cba9);
        a.wide(solana_sbpf::simd::VBCAST32, 2, 0, 0, 3, 0);
        a.wide(solana_sbpf::simd::VBCAST64, 6, 0, 0, 3, 0);
    })));
    cases
}

#[test]
fn new_ops_reference() {
    use solana_sbpf::simd::{compute, from_words, ternlog, words, VecInsn};
    let mut seed = 7u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as u32
    };
    for _ in 0..100 {
        let (d, a, b) = (next(), next(), next());
        assert_eq!(ternlog(d, a, b, 0x96), d ^ a ^ b);
        assert_eq!(ternlog(d, a, b, 0xd2), d ^ (!a & b));
        assert_eq!(ternlog(d, a, b, 0xca), (d & a) | (!d & b));
        assert_eq!(ternlog(d, a, b, 0xf0), d);
        assert_eq!(ternlog(d, a, b, 0xcc), a);
        assert_eq!(ternlog(d, a, b, 0xaa), b);
    }
    let op = |subop| VecInsn { subop, vd: 0, va: 0, vb: 0, wide: false, dst: 0, src: 0, off: 0 };
    let x: [u32; 8] = std::array::from_fn(|_| next());
    let y: [u32; 8] = [0, 1, 5, 13, 16, 31, 32, 7];
    let out = compute(&op(VROLV32), &x, &y);
    for i in 0..8 {
        assert_eq!(out[i], x[i].rotate_left(y[i] % 32), "vrolv count {}", y[i]);
    }
    let (xw, yw) = (words(&x), words(&std::array::from_fn(|_| next())));
    let y2 = from_words(&yw);
    let hi = words(&compute(&op(VMULHI16), &x, &y2));
    let madd = compute(&op(VMADD16), &x, &y2);
    for i in 0..16 {
        assert_eq!(hi[i] as i16 as i32, (xw[i] as i16 as i32 * yw[i] as i16 as i32) >> 16);
    }
    for i in 0..8 {
        let p = |j: usize| xw[j] as i16 as i32 * yw[j] as i16 as i32;
        assert_eq!(madd[i] as i32, p(2 * i).wrapping_add(p(2 * i + 1)));
    }
    assert_eq!(from_words(&words(&x)), x);
}

#[test]
fn vector_ops_interpreter_reference() {
    // Checks the interpreter against a direct Rust model for a few ops
    let input = random_input(1);
    let lanes = |i: usize| solana_sbpf::simd::read_lanes(input[32 * i..].as_ptr());
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.vadd(0, 0, 1)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    let got = solana_sbpf::simd::read_lanes(out.as_ptr());
    let (a, b) = (lanes(0), lanes(1));
    for i in 0..8 {
        assert_eq!(got[i], a[i].wrapping_add(b[i]));
    }
    assert_eq!(got[0], 0, "u32::MAX + 1 must wrap to 0");
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.vror(2, 0, 7)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    let got = solana_sbpf::simd::read_lanes(out[64..].as_ptr());
    assert_eq!(got[1], 0x8000_0000u32.rotate_right(7));
    // little-endian lane interpretation
    assert_eq!(out[64 + 4..64 + 8], 0x8000_0000u32.rotate_right(7).to_le_bytes());
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.vtranspose(0)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    for i in 0..8 {
        for j in 0..8 {
            assert_eq!(solana_sbpf::simd::read_lanes(out[32 * i..].as_ptr())[j], lanes(j)[i]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn vector_ops_jit_matches_interpreter() {
    for seed in 0..4 {
        let input = random_input(seed);
        for (name, body) in op_cases() {
            let (slot_result, _) = run_op(&op_program(VecMode::Slot, &*body), &input, Engine::Interpreter, SimdJitStrategy::Direct);
            for mode in [VecMode::Slot, VecMode::Call] {
                let program = op_program(mode, &*body);
                let (expected, cus) = run_op(&program, &input, Engine::Interpreter, SimdJitStrategy::Direct);
                if mode == VecMode::Call {
                    assert_eq!(expected, slot_result, "{name} call form differs from opcode form");
                }
                for strategy in [SimdJitStrategy::Direct, SimdJitStrategy::RegisterCached] {
                    let (out, jit_cus) = run_op(&program, &input, Engine::Jit, strategy);
                    assert_eq!(out, expected, "{name} {mode:?} {strategy:?} seed={seed}");
                    assert_eq!(cus, jit_cus);
                }
            }
        }
    }
}


use solana_sbpf::{ebpf, error::EbpfError, simd::asm::Asm};

fn engines() -> Vec<(Engine, SimdJitStrategy)> {
    let mut v = vec![(Engine::Interpreter, SimdJitStrategy::Direct)];
    if cfg!(target_arch = "x86_64") {
        v.push((Engine::Jit, SimdJitStrategy::Direct));
        v.push((Engine::Jit, SimdJitStrategy::RegisterCached));
    }
    v
}

/// Runs a program on all engines, asserts identical results, heap, vector memory and CUs
fn run_all(program: &[u8], input: &[u8]) -> (String, Vec<u8>, Vec<u8>, u64) {
    let mut reference: Option<(String, Vec<u8>, Vec<u8>, u64)> = None;
    for (engine, strategy) in engines() {
        let mut runner = Runner::new(program, input, 0x2000, config(strategy)).unwrap();
        runner.budget = 10_000;
        if engine == Engine::Jit {
            runner.jit_compile().unwrap();
        }
        let (cus, result) = runner.run(engine);
        let got = (format!("{:?}", result), runner.heap().to_vec(), runner.vector.as_slice().to_vec(), cus);
        match &reference {
            None => reference = Some(got),
            Some(r) => {
                assert_eq!(r.0, got.0, "result differs {:?} {:?}", engine, strategy);
                assert!(r.1 == got.1, "heap differs {:?} {:?}", engine, strategy);
                assert!(r.2 == got.2, "vector memory differs {:?} {:?}", engine, strategy);
                assert_eq!(r.3, got.3, "CUs differ {:?} {:?}", engine, strategy);
            }
        }
    }
    reference.unwrap()
}

#[test]
fn vector_file_is_not_mapped() {
    let input = random_input(5);
    // The former vector region address is unmapped, scalar accesses fault
    for store in [false, true] {
        let mut a = Asm::new(VecMode::Slot);
        a.vload(0, 1, 0);
        a.lddw(2, 6 << 32);
        if store {
            a.stxb(2, 0, 3);
        } else {
            a.ldxb(4, 2, 0);
        }
        a.mov64_imm(0, 0);
        a.exit();
        let (result, ..) = run_all(&a.finish(), &input);
        assert!(result.contains("AccessViolation"), "{}", result);
    }
    // vload / vstore of several slots straddling the end of the heap
    for store in [false, true] {
        let mut a = Asm::new(VecMode::Slot);
        a.lddw(2, ebpf::MM_HEAP_START + 0x2000 - 3 * 32 + 1);
        if store {
            a.vstore_n(2, 0, 0, 3);
        } else {
            a.vload_n(0, 3, 2, 0);
        }
        a.mov64_imm(0, 0);
        a.exit();
        let (result, ..) = run_all(&a.finish(), &input);
        assert!(result.contains("AccessViolation"), "{}", result);
    }
    // vstore into the read-only program region
    let mut a = Asm::new(VecMode::Slot);
    a.lddw(2, ebpf::MM_RODATA_START);
    a.vstore(2, 0, 0);
    a.exit();
    let (result, ..) = run_all(&a.finish(), &input);
    assert!(result.contains("AccessViolation") && result.contains("Store"), "{}", result);
}

/// Bulk vload / vstore (up to 32 slots with one translation), both forms
#[test]
fn bulk_transfers() {
    let input = test_data(32 * 40, 77);
    for mode in [VecMode::Slot, VecMode::Call] {
        let mut a = Asm::new(mode);
        a.mov64(3, 1);
        a.vload_n(10, 32, 3, 0);
        a.vload_n(42, 8, 3, 32 * 32);
        a.vadd(10, 10, 41); // first and last slot of the first transfer
        a.lddw(4, ebpf::MM_HEAP_START);
        a.vstore_n(4, 7, 10, 32);
        a.vstore_n(4, 7 + 32 * 32, 42, 8);
        a.mov64_imm(0, 0);
        a.exit();
        let (result, heap, ..) = run_all(&a.finish(), &input);
        assert_eq!(result, "Ok(0)");
        assert_eq!(heap[7 + 32..7 + 40 * 32], input[32..40 * 32], "{mode:?}");
        let (x, y) = (solana_sbpf::simd::read_lanes(input.as_ptr()), solana_sbpf::simd::read_lanes(input[31 * 32..].as_ptr()));
        let got = solana_sbpf::simd::read_lanes(heap[7..].as_ptr());
        for i in 0..8 {
            assert_eq!(got[i], x[i].wrapping_add(y[i]));
        }
    }
}

/// 64 bit lane moves, broadcasts and byte masks against a Rust model, both forms
#[test]
fn lane_moves() {
    let input = random_input(40);
    let lanes64 = |i: usize| -> [u64; 4] { std::array::from_fn(|l| u64::from_le_bytes(input[32 * i + 8 * l..][..8].try_into().unwrap())) };
    for mode in [VecMode::Slot, VecMode::Call] {
        let mut a = Asm::new(mode);
        a.mov64(3, 1);
        a.vload_n(0, 4, 3, 0);
        a.lddw(6, ebpf::MM_HEAP_START);
        // r7 = sum of the four lanes of v1, stored at heap + 0x100
        a.mov64_imm(7, 0);
        for lane in 0..4 {
            a.vext64(4, 1, lane);
            a.add64(7, 4);
        }
        a.stxdw(6, 0x100, 7);
        // v0 lane 2 = 0x1122334455667788, v2 = broadcast, v3 lane 3 = v1 lane 0 (from a cached slot)
        a.lddw(5, 0x1122_3344_5566_7788);
        a.vins64(0, 5, 2);
        a.vbcast64(2, 5);
        a.vadd(1, 1, 1);
        a.vext64(4, 1, 0);
        a.vins64(3, 4, 3);
        a.vmovmsk8(8, 3, false);
        a.stxdw(6, 0x108, 8);
        a.vmovmsk8(8, 0, true);
        a.stxdw(6, 0x110, 8);
        a.vstore_n(6, 0, 0, 4);
        a.mov64_imm(0, 0);
        a.exit();
        let (result, heap, ..) = run_all(&a.finish(), &input);
        assert_eq!(result, "Ok(0)", "{mode:?}");
        let word = |at: usize| u64::from_le_bytes(heap[at..at + 8].try_into().unwrap());
        let out = |i: usize, l: usize| word(32 * i + 8 * l);
        assert_eq!(word(0x100), lanes64(1).iter().fold(0u64, |s, x| s.wrapping_add(*x)));
        let mut v0 = lanes64(0);
        v0[2] = 0x1122_3344_5566_7788;
        let doubled: Vec<u64> = (0..4).map(|l| {
            let x = lanes64(1)[l];
            (x as u32).wrapping_mul(2) as u64 | ((((x >> 32) as u32).wrapping_mul(2)) as u64) << 32
        }).collect();
        let mut v3 = lanes64(3);
        v3[3] = doubled[0];
        for l in 0..4 {
            assert_eq!(out(0, l), v0[l]);
            assert_eq!(out(2, l), 0x1122_3344_5566_7788);
            assert_eq!(out(3, l), v3[l]);
        }
        let mask = |bytes: &[u8]| bytes.iter().enumerate().fold(0u64, |m, (i, b)| m | ((b >> 7) as u64) << i);
        assert_eq!(word(0x108), mask(&heap[96..128]));
        assert_eq!(word(0x110), mask(&heap[0..64]));
    }
}

/// Every 512 bit operation equals its definition: the 256 bit operation on both halves (lane-wise
/// ones), or the Rust model (cross lane ones)
#[test]
fn wide_ops_reference() {
    use solana_sbpf::simd::{self, intrinsics::NAMES_512, VecInsn};
    let input = random_input(41);
    for subop in 0..simd::SUBOP_COUNT {
        if NAMES_512[subop as usize].is_none() || subop >= simd::VSLL16 || matches!(subop, simd::VMOVMSK8 | simd::VBCAST32 | simd::VBCAST64 | simd::VTRANSPOSE64 | simd::VPERMT2D | simd::VCOMPRESSLT16 | simd::VMONT16 | simd::VCMP | simd::VBLENDM | simd::VCOMPRESS | simd::VEXPAND) {
            continue;
        }
        let off = match subop {
            simd::VROR32 | simd::VSRL32 | simd::VSLL32 => 13,
            simd::VSHUFD | simd::VBLENDD | simd::VTERNLOG | simd::VBLENDW => 0x6c,
            _ => 0,
        };
        let build = |wide: bool| {
            let body = move |a: &mut Asm| {
                if wide {
                    a.wide(subop, 0, 4, 8, 0, off);
                    a.wide(subop, 10, 10, 12, 0, off);
                } else {
                    for h in 0..2 {
                        a.vop_full(subop, false, h, 4 + h, 8 + h, 0, 0, off);
                        a.vop_full(subop, false, 10 + h, 10 + h, 12 + h, 0, 0, off);
                    }
                }
            };
            op_program(VecMode::Slot, &body)
        };
        let (wide, _) = run_op(&build(true), &input, Engine::Interpreter, SimdJitStrategy::Direct);
        let lane = |buf: &[u8], slot: usize, i: usize| u32::from_le_bytes(buf[32 * slot + 4 * i..][..4].try_into().unwrap());
        let get = |buf: &[u8], pair: usize, i: usize| lane(buf, pair + i / 8, i % 8);
        match subop {
            simd::VPERMD => {
                for i in 0..16 {
                    assert_eq!(get(&wide, 0, i), get(&input, 4, (get(&input, 8, i) & 15) as usize));
                }
            }
            simd::VPERMI2D => {
                for i in 0..16 {
                    let idx = (get(&input, 0, i) & 31) as usize;
                    let expected = if idx < 16 { get(&input, 4, idx) } else { get(&input, 8, idx - 16) };
                    assert_eq!(get(&wide, 0, i), expected, "vpermi2d512 lane {i}");
                }
            }
            _ => {
                let (split, _) = run_op(&build(false), &input, Engine::Interpreter, SimdJitStrategy::Direct);
                assert_eq!(wide, split, "{}", simd::mnemonic(subop, true));
            }
        }
        let _ = VecInsn::decode;
    }
    // vtranspose64 (4 x 4 and 8 x 8) and vrolv64 against the model
    let q = |buf: &[u8], i: usize| u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap());
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.vop_full(simd::VTRANSPOSE64, false, 4, 0, 0, 0, 0, 0)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(q(&out, 16 + 4 * r + c), q(&input, 16 + 4 * c + r));
        }
    }
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.wide(simd::VTRANSPOSE64, 0, 0, 0, 0, 0)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    for r in 0..8 {
        for c in 0..8 {
            assert_eq!(q(&out, 8 * r + c), q(&input, 8 * c + r));
        }
    }
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.wide(simd::VROLV64, 0, 2, 4, 0, 0)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    for i in 0..8 {
        assert_eq!(q(&out, i), q(&input, 8 + i).rotate_left((q(&input, 16 + i) % 64) as u32));
    }
    // compresslt16 (both widths), shufb and permt2d against the model
    let w16 = |buf: &[u8], i: usize| u16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]);
    for wide in [false, true] {
        let n = if wide { 32 } else { 16 };
        let body = move |a: &mut Asm| {
            a.vop_full(simd::VCOMPRESSLT16, wide, 0, 2, 4, 6, 0, 0);
            a.vins64(15, 6, 0);
        };
        let (out, _) = run_op(&op_program(VecMode::Slot, &body), &input, Engine::Interpreter, SimdJitStrategy::Direct);
        let kept: Vec<u16> = (0..n).filter(|&i| w16(&input, 32 + i) < w16(&input, 64 + i)).map(|i| w16(&input, 32 + i)).collect();
        for i in 0..n {
            assert_eq!(w16(&out, i), kept.get(i).copied().unwrap_or(0), "compresslt16 wide={wide} lane {i}");
        }
        assert_eq!(u64::from_le_bytes(out[15 * 32..15 * 32 + 8].try_into().unwrap()), kept.len() as u64);
    }
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.vop_full(simd::VSHUFB, false, 0, 1, 2, 0, 0, 0)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    for i in 0..32 {
        let idx = input[64 + i];
        assert_eq!(out[i], if idx & 0x80 != 0 { 0 } else { input[32 + (i & !15) + (idx & 15) as usize] });
    }
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.wide(simd::VPERMT2D, 0, 4, 2, 0, 0)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    let lane32 = |buf: &[u8], i: usize| u32::from_le_bytes(buf[4 * i..4 * i + 4].try_into().unwrap());
    for i in 0..16 {
        let idx = (lane32(&input, 32 + i) & 31) as usize;
        let expected = if idx < 16 { lane32(&input, idx) } else { lane32(&input, 16 + idx - 16) };
        assert_eq!(lane32(&out, i), expected, "vpermt2d512 lane {i}");
    }
    // 64 bit arithmetic and mask operations against the model
    let q = |buf: &[u8], i: usize| u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap());
    for wide in [false, true] {
        let lanes = if wide { 8 } else { 4 };
        let arith: [(u8, i16, fn(u64, u64, u32) -> u64); 11] = [
            (simd::VADD64, 0, |x, y, _| x.wrapping_add(y)),
            (simd::VSUB64, 0, |x, y, _| x.wrapping_sub(y)),
            (simd::VSLL64, 13, |x, _, n| x << n),
            (simd::VSRL64, 63, |x, _, n| x >> n),
            (simd::VSRA64, 9, |x, _, n| ((x as i64) >> n) as u64),
            (simd::VSLLV64, 0, |x, y, _| if y > 63 { 0 } else { x << y }),
            (simd::VSRLV64, 0, |x, y, _| if y > 63 { 0 } else { x >> y }),
            (simd::VMULU32, 0, |x, y, _| (x as u32 as u64) * (y as u32 as u64)),
            (simd::VMULLO64, 0, |x, y, _| x.wrapping_mul(y)),
            (simd::VMADD52LO, 0, |x, y, _| ((x & ((1 << 52) - 1)) as u128 * (y & ((1 << 52) - 1)) as u128) as u64 & ((1 << 52) - 1)),
            (simd::VMADD52HI, 0, |x, y, _| (((x & ((1 << 52) - 1)) as u128 * (y & ((1 << 52) - 1)) as u128) >> 52) as u64),
        ];
        for (subop, off, f) in arith {
            let body = move |a: &mut Asm| {
                a.vand(4, 4, 6); // shift counts: lanes of v4 / v5 & v6 / v7 (random, often > 63; the
                                 // interpreter only, Rosetta 2 gets vpsllvq wrong for those)
                a.vand(5, 5, 7);
                a.vop_full(subop, wide, 8, 0, 4, 0, 0, off);
            };
            let (out, _) = run_op(&op_program(VecMode::Slot, &body), &input, Engine::Interpreter, SimdJitStrategy::Direct);
            for i in 0..lanes {
                let (x, y) = (q(&input, i), q(&input, 16 + i) & q(&input, 24 + i));
                let acc = if matches!(subop, simd::VMADD52LO | simd::VMADD52HI) { q(&input, 32 + i) } else { 0 };
                assert_eq!(q(&out, 32 + i), acc.wrapping_add(f(x, y, off as u32)), "{} lane {i}", simd::mnemonic(subop, wide));
            }
        }
        for size in 0..4u32 {
            let bytes = 1usize << size;
            let n = 32 * (1 + wide as usize) / bytes;
            let lane = |buf: &[u8], first: usize, i: usize| (0..bytes).fold(0u64, |v, k| v | (buf[first + bytes * i + k] as u64) << (8 * k));
            let signed = |v: u64| ((v << (64 - 8 * bytes)) as i64) >> (64 - 8 * bytes);
            let mask_in = 0xa5c3_0ff0_6996_1248u64;
            for pred in 0..16i16 {
                let body = move |a: &mut Asm| {
                    a.vmov(4, 2); // equal lanes in some places
                    a.vblendd(4, 4, 6, 0x5a);
                    a.vop_full(simd::VCMP, wide, 0, 0, 4, 6, 0, pred | (size as i16) << 4);
                    a.vins64(15, 6, 0);
                };
                let (out, _) = run_op(&op_program(VecMode::Slot, &body), &input, Engine::Interpreter, SimdJitStrategy::Direct);
                let mut b_src = input[4 * 32..4 * 32 + 32 * (1 + wide as usize)].to_vec();
                // v4 = blend(v2, v6, 0x5a) on the low slot, v5 untouched
                for l in 0..8 {
                    let from = if (0x5a >> l) & 1 != 0 { 6 * 32 } else { 2 * 32 };
                    b_src[4 * l..4 * l + 4].copy_from_slice(&input[from + 4 * l..from + 4 * l + 4]);
                }
                let mut mask = 0u64;
                for i in 0..n {
                    let (x, y) = (lane(&input, 0, i), lane(&b_src, 0, i));
                    let ord = if pred & 8 != 0 { x.cmp(&y) } else { signed(x).cmp(&signed(y)) };
                    let t = [ord.is_eq(), ord.is_lt(), ord.is_le(), false, ord.is_ne(), ord.is_ge(), ord.is_gt(), true][(pred & 7) as usize];
                    mask |= (t as u64) << i;
                }
                assert_eq!(q(&out, 60), mask, "cmp size {size} pred {pred} wide {wide}");
            }
            let body = move |a: &mut Asm| {
                a.lddw(3, mask_in);
                a.vop_full(simd::VBLENDM, wide, 8, 0, 4, 0, 3, size as i16);
                a.vop_full(simd::VCOMPRESS, wide, 10, 0, 0, 6, 3, size as i16);
                a.vins64(15, 6, 0);
                a.vop_full(simd::VEXPAND, wide, 12, 0, 0, 0, 3, size as i16);
            };
            let (out, _) = run_op(&op_program(VecMode::Slot, &body), &input, Engine::Interpreter, SimdJitStrategy::Direct);
            let bit = |i: usize| (mask_in >> i) & 1 != 0;
            let kept: Vec<u64> = (0..n).filter(|&i| bit(i)).map(|i| lane(&input, 0, i)).collect();
            let mut k = 0;
            for i in 0..n {
                assert_eq!(lane(&out, 8 * 32, i), if bit(i) { lane(&input, 0, i) } else { lane(&input, 4 * 32, i) }, "blendm");
                assert_eq!(lane(&out, 10 * 32, i), kept.get(i).copied().unwrap_or(0), "compress");
                let e = if bit(i) { k += 1; lane(&input, 0, k - 1) } else { 0 };
                assert_eq!(lane(&out, 12 * 32, i), e, "expand size {size} wide {wide} lane {i}");
            }
            assert_eq!(q(&out, 60), kept.len() as u64, "compress count");
        }
    }
    // mont16 against the model
    for wide in [false, true] {
        let (out, _) = run_op(&op_program(VecMode::Slot, &move |a: &mut Asm| a.vop_full(simd::VMONT16, wide, 10, 0, 2, 0, 0, 6)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
        let w16 = |buf: &[u8], slot: usize, i: usize| i16::from_le_bytes([buf[32 * slot + 2 * i], buf[32 * slot + 2 * i + 1]]) as i32;
        for i in 0..16 * (1 + wide as usize) {
            let (h, l) = (i / 16, i % 16);
            let (a, w, wp, q) = (w16(&input, h, l), w16(&input, 2 + h, l), w16(&input, 4 + h, l), w16(&input, 6 + h, l));
            let t = (a * wp) as i16 as i32;
            let expected = (((a * w) >> 16) - ((t * q) >> 16)) as i16 as i32;
            assert_eq!(w16(&out, 10 + h, l), expected, "mont16 wide={wide} lane {i}");
        }
    }
    // vpermi2d256 against the model
    let (out, _) = run_op(&op_program(VecMode::Slot, &|a| a.vpermi2d(0, 4, 5, false)), &input, Engine::Interpreter, SimdJitStrategy::Direct);
    let lane = |buf: &[u8], slot: usize, i: usize| u32::from_le_bytes(buf[32 * slot + 4 * i..][..4].try_into().unwrap());
    for i in 0..8 {
        let idx = (lane(&input, 0, i) & 15) as usize;
        let expected = if idx < 8 { lane(&input, 4, idx) } else { lane(&input, 5, idx - 8) };
        assert_eq!(lane(&out, 0, i), expected);
    }
}

#[test]
fn unaligned_vector_load_store() {
    let input = random_input(6);
    for offset in [1i16, 3, 7, 13, 31] {
        let mut a = Asm::new(VecMode::Slot);
        a.vload(0, 1, offset);
        a.lddw(2, ebpf::MM_HEAP_START + 5);
        a.vstore(2, offset, 0);
        a.mov64_imm(0, 0);
        a.exit();
        let (result, heap, ..) = run_all(&a.finish(), &input);
        assert_eq!(result, "Ok(0)");
        let at = 5 + offset as usize;
        assert_eq!(heap[at..at + 32], input[offset as usize..offset as usize + 32]);
    }
}

#[test]
fn register_cache_branches_calls_and_pressure() {
    let input = random_input(8);
    let mut a = Asm::new(VecMode::Slot);
    let func = a.new_label();
    let top = a.new_label();
    for i in 0..16u8 {
        a.vload(i, 1, 32 * i as i16);
    }
    a.mov64_imm(6, 5);
    a.bind(top);
    // More live slots than host registers: forces evictions, spills and reloads
    for i in 0..48usize {
        let s = |x: usize| x as u8;
        a.vadd(s(16 + (i % 40)), s(i % 16), s((i * 7) % 16));
        a.vxor(s(i % 16), s(16 + ((i * 3) % 40)), s(i % 16));
        a.vror(s(16 + ((i * 5) % 40)), s(16 + (i % 40)), (i % 31) as i16 + 1);
    }
    a.call(func);
    a.sub64_imm(6, 1);
    a.jne_imm(6, 0, top);
    a.lddw(2, ebpf::MM_HEAP_START);
    for i in 0..56u8 {
        a.vstore(2, 32 * i as i16, i);
    }
    a.mov64_imm(0, 0);
    a.exit();
    a.bind(func);
    a.vshufd(3, 4, 0x1b);
    a.vtranspose(8);
    a.vpermd(17, 18, 3);
    a.exit();
    let (result, ..) = run_all(&a.finish(), &input);
    assert_eq!(result, "Ok(0)");
}

#[test]
fn instruction_meter_exhaustion_inside_vector_code() {
    let input = random_input(9);
    for budget in [3u64, 10, 25, 26, 27, 60] {
        let mut a = Asm::new(VecMode::Slot);
        a.vload(0, 1, 0);
        for i in 0..40u8 {
            a.vadd(i % 8, (i + 1) % 8, (i + 2) % 8);
        }
        a.mov64_imm(0, 0);
        a.exit();
        let program = a.finish();
        let mut reference = None;
        for (engine, strategy) in engines() {
            let mut runner = Runner::new(&program, &input, 0x2000, config(strategy)).unwrap();
            runner.budget = budget;
            if engine == Engine::Jit {
                runner.jit_compile().unwrap();
            }
            let (cus, result) = runner.run(engine);
            let got = (format!("{:?}", result), cus);
            if let Some(r) = &reference {
                assert_eq!(r, &got, "budget {} {:?} {:?}", budget, engine, strategy);
            } else {
                reference = Some(got);
            }
        }
    }
}

#[test]
fn zero_initialized_per_invocation() {
    let input = random_input(10);
    let mut a = Asm::new(VecMode::Slot);
    a.lddw(2, ebpf::MM_HEAP_START);
    a.vstore(2, 0, 7); // v7 must read as zero
    a.vload(7, 1, 0); // then dirty it for the next run
    a.mov64_imm(0, 0);
    a.exit();
    let program = a.finish();
    for (engine, strategy) in engines() {
        let mut runner = Runner::new(&program, &input, 0x2000, config(strategy)).unwrap();
        if engine == Engine::Jit {
            runner.jit_compile().unwrap();
        }
        for _ in 0..3 {
            let (_, result) = runner.run(engine);
            assert!(result.is_ok());
            assert!(runner.heap()[..32].iter().all(|b| *b == 0));
            assert_eq!(runner.vector.as_slice()[7 * 32..8 * 32], input[..32]);
        }
    }
}

#[test]
fn detached_and_disabled_and_malformed() {
    let input = random_input(11);
    let mut a = Asm::new(VecMode::Slot);
    a.vadd(0, 1, 2);
    a.exit();
    let program = a.finish();
    // No vector memory attached: the VM allocates its own vector file
    for (engine, strategy) in engines() {
        let mut runner = Runner::new(&program, &input, 0x2000, config(strategy)).unwrap();
        if engine == Engine::Jit {
            runner.jit_compile().unwrap();
        }
        for _ in 0..2 {
            let (_, result) = runner.run_detached(engine);
            assert!(result.is_ok(), "{:?}", result);
        }
    }
    // Feature disabled at runtime
    let mut cfg = config(SimdJitStrategy::Direct);
    cfg.enable_simd_experiment = false;
    let mut runner = Runner::new(&program, &input, 0x2000, cfg).unwrap();
    let (_, result) = runner.run(Engine::Interpreter);
    assert!(matches!(result, solana_sbpf::error::StableResult::Err(EbpfError::UnsupportedInstruction)));
    if cfg!(target_arch = "x86_64") {
        assert!(runner.jit_compile().is_err());
    }
    // Malformed encodings are rejected by the verifier
    for (imm, dst, src, off) in [
        (0x3f << 18 | 1 << 25, 0u8, 0u8, 0i16), // unknown sub-operation (127)
        (1 << 26, 0, 0, 0),           // reserved bit
        ((0x06 << 18) as i32, 0, 0, 32), // rotation >= 32
        ((0x10 << 18) as i32, 0, 0, 32), // shift >= 32
        ((0x02 << 18) as i32, 3, 0, 0), // dst register of an op without scalar result
        ((0x02 << 18) as i32, 0, 2, 0), // unused src register
        ((0x1f << 18) as i32, 10, 0, 0), // vext64 into r10
        ((0x1f << 18) as i32, 1, 0, 4), // vext64 lane out of range
        ((0x02 << 18 | 1 << 24 | 1) as i32, 0, 0, 0), // 512 bit op on an odd slot
        ((0x08 << 18 | 1 << 24) as i32, 0, 0, 0), // vtranspose8 has no 512 bit form
        ((32 << 12) as i32, 0, 1, 0),    // vload of 33 slots
    ] {
        let insn = ebpf::Insn { ptr: 0, opc: ebpf::VOP, dst, src, off, imm: imm as i64 };
        let mut bytes = insn.to_vec();
        bytes.extend_from_slice(&ebpf::Insn { opc: ebpf::EXIT, ..Default::default() }.to_vec());
        assert!(Runner::new(&bytes, &input, 0x2000, config(SimdJitStrategy::Direct)).is_err(), "{:#x}", imm);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn unsupported_host_cpu() {
    let mut a = Asm::new(VecMode::Slot);
    a.vadd(0, 1, 2);
    a.exit();
    let mut cfg = config(SimdJitStrategy::RegisterCached);
    cfg.simd_assume_avx2 = false;
    let runner = Runner::new(&a.finish(), &random_input(12), 0x2000, cfg).unwrap();
    let result = runner.jit_compile();
    if std::is_x86_feature_detected!("avx2") {
        assert!(result.is_ok());
    } else {
        // e.g. Rosetta 2, which hides AVX2 from CPUID
        assert!(matches!(result, Err(EbpfError::SimdHostUnsupported)), "{:?}", result);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn generated_code_uses_avx2_and_cache_removes_memory_traffic() {
    let program = blake3_program::microbench_program(64, VecMode::Slot);
    let mut stats = Vec::new();
    for strategy in [SimdJitStrategy::Direct, SimdJitStrategy::RegisterCached] {
        let runner = Runner::new(&program, &blake3_program::build_input(&[]), HEAP_SIZE, config(strategy)).unwrap();
        runner.jit_compile().unwrap();
        stats.push(runner.executable.get_compiled_program().unwrap().simd_stats);
    }
    let (direct, cached) = (stats[0], stats[1]);
    // Direct: one load + one store per op (256 ops), cached: v0 stays in a register
    assert!(direct.vector_loads >= 256 && direct.vector_stores >= 256, "{:?}", direct);
    assert!(cached.vector_stores < 8 && cached.vector_loads < 16, "{:?}", cached);
}


#[test]
fn blake3_one_mebibyte() {
    let data = test_data(1 << 20, 11);
    let expected = *blake3::hash(&data).as_bytes();
    let mut configs = vec![(Kernel::Simd8, VecMode::Slot, Engine::Interpreter, SimdJitStrategy::Direct)];
    if cfg!(target_arch = "x86_64") {
        configs.push((Kernel::Scalar, VecMode::Slot, Engine::Jit, SimdJitStrategy::Direct));
        configs.push((Kernel::Simd8, VecMode::Slot, Engine::Jit, SimdJitStrategy::Direct));
        configs.push((Kernel::Simd8, VecMode::Slot, Engine::Jit, SimdJitStrategy::RegisterCached));
        configs.push((Kernel::Simd1, VecMode::Slot, Engine::Jit, SimdJitStrategy::RegisterCached));
        configs.push((Kernel::Simd8, VecMode::Call, Engine::Jit, SimdJitStrategy::RegisterCached));
        configs.push((Kernel::Simd8, VecMode::Call, Engine::Interpreter, SimdJitStrategy::Direct));
    }
    for (kernel, mode, engine, strategy) in configs {
        let (digest, _) = hash_with(kernel, mode, engine, strategy, &data);
        assert_eq!(digest, expected, "{:?} {:?} {:?} {:?}", kernel, mode, engine, strategy);
    }
}

/// Writes the machine code of the microbenchmark loop body (repeat = 4) for inspection:
/// `JIT_DUMP_DIR=... cargo test --test simd dump_jit_code -- --ignored`
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore]
fn dump_jit_code() {
    let dir = std::env::var("JIT_DUMP_DIR").unwrap();
    let repeat = 4;
    for (name, mode, strategy) in [
        ("slot_direct", VecMode::Slot, SimdJitStrategy::Direct),
        ("slot_cached", VecMode::Slot, SimdJitStrategy::RegisterCached),
        ("call_direct", VecMode::Call, SimdJitStrategy::Direct),
        ("call_cached", VecMode::Call, SimdJitStrategy::RegisterCached),
    ] {
        let program = blake3_program::microbench_program(repeat, mode);
        let mut cfg = config(strategy);
        cfg.noop_instruction_rate = 0;
        let runner = Runner::new(&program, &blake3_program::build_input(&[]), HEAP_SIZE, cfg).unwrap();
        runner.jit_compile().unwrap();
        let compiled = runner.executable.get_compiled_program().unwrap();
        let pcs = compiled.pc_section();
        // lddw + mov64 + 5 vloads + ldxdw, then the loop: 4 * repeat vector ops, sub64, jne.
        // A vector op is one instruction, or two in the call form (mov64 r1 + call), a vload
        // three (mov64 r2 + mov64 r1 + call)
        let (per_load, per_op) = if mode == VecMode::Call { (3, 2) } else { (1, 1) };
        let head = 2 + 1 + 5 * per_load + 1;
        let start = (pcs[head] & i32::MAX as u32) as usize;
        let end = (pcs[head + 4 * repeat * per_op + 2] & i32::MAX as u32) as usize;
        std::fs::write(format!("{}/{}.bin", dir, name), &compiled.text_section()[start..end]).unwrap();
        std::fs::write(format!("{}/{}.stats", dir, name), format!("{:?}\n", compiled.simd_stats)).unwrap();
    }
}


// Vector operations as static calls (SIMD-0645 style intrinsics)

use solana_sbpf::simd::intrinsics;

/// The call form compiles to the same vector code as the opcode form, with no slow paths
#[cfg(target_arch = "x86_64")]
#[test]
fn intrinsic_calls_lower_to_inline_avx2() {
    for kernel in [Kernel::Simd1, Kernel::Simd8] {
        for strategy in [SimdJitStrategy::Direct, SimdJitStrategy::RegisterCached] {
            let stats = |mode| {
                let program = blake3_program::hash_program(kernel, mode);
                let runner = Runner::new(&program, &blake3_program::build_input(&[]), HEAP_SIZE, config(strategy)).unwrap();
                runner.jit_compile().unwrap();
                runner.executable.get_compiled_program().unwrap().simd_stats
            };
            let (opcode, call) = (stats(VecMode::Slot), stats(VecMode::Call));
            assert_eq!(call.intrinsic_slow_paths, 0, "{kernel:?} {strategy:?} {call:?}");
            assert_eq!(call.vector_instructions, opcode.vector_instructions);
            // Same lowering; the extra `mov64 r5` instructions only shift a few meter checkpoints,
            // which write back dirty slots
            let close = |a: usize, b: usize| a.abs_diff(b) * 50 <= b.max(1);
            assert!(close(call.vector_loads, opcode.vector_loads), "{kernel:?} {strategy:?} {call:?} {opcode:?}");
            assert!(close(call.vector_stores, opcode.vector_stores), "{kernel:?} {strategy:?} {call:?} {opcode:?}");
            assert!(call.spills.abs_diff(opcode.spills) * 20 <= opcode.spills.max(1), "{call:?} {opcode:?}");
        }
    }
}

/// Operands which aren't a compile time constant take the syscall slow path with identical results
#[test]
fn intrinsic_dynamic_operands_slow_path() {
    use solana_sbpf::simd::{VecInsn, VADD32, VSTORE, VXOR};
    let input = random_input(21);
    let mut packed_input = input.clone();
    let packed = intrinsics::pack(&VecInsn { subop: VADD32, vd: 0, va: 1, vb: 2, wide: false, dst: 0, src: 0, off: 0 });
    packed_input.extend_from_slice(&packed.to_le_bytes());
    let mut a = Asm::new(VecMode::Call);
    a.mov64(3, 1);
    for i in 0..4u8 {
        a.vload(i, 3, 32 * i as i16);
    }
    // r1 loaded from memory
    a.ldxdw(1, 3, input.len() as i16);
    a.raw(ebpf::CALL_IMM, 0, 0, 0, intrinsics::hash(VADD32, false) as i32 as i64);
    // r1 constant, but set before a block boundary
    a.mov64_imm(1, packed as i64);
    let join = a.new_label();
    a.ja(join);
    a.bind(join);
    a.raw(ebpf::CALL_IMM, 0, 0, 0, intrinsics::hash(VXOR, false) as i32 as i64);
    // Fast path again, mixing with the slow path results
    a.vadd(3, 0, 3);
    a.lddw(2, ebpf::MM_HEAP_START);
    for i in 0..4u8 {
        a.vstore(2, 32 * i as i16, i);
    }
    a.mov64(0, 1);
    a.exit();
    let program = a.finish();
    let (result, heap, _, _) = run_all(&program, &packed_input);
    // r1 still holds the operands of the last call (the intrinsics leave it unchanged)
    let last = intrinsics::pack(&VecInsn { subop: VSTORE, vd: 0, va: 3, vb: 0, wide: false, dst: 0, src: 2, off: 96 });
    assert_eq!(result, format!("Ok({})", last));
    let lanes = |i: usize, buf: &[u8]| -> Vec<u32> { buf[32 * i..32 * i + 32].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect() };
    let (v1, v2, v3) = (lanes(1, &input), lanes(2, &input), lanes(3, &input));
    // vadd(v0, v1, v2) then vxor(v0, v1, v2) with the same packed operands: v0 = v1 ^ v2
    let expected0: Vec<u32> = (0..8).map(|i| v1[i] ^ v2[i]).collect();
    assert_eq!(lanes(0, &heap), expected0);
    let expected3: Vec<u32> = (0..8).map(|i| expected0[i].wrapping_add(v3[i])).collect();
    assert_eq!(lanes(3, &heap), expected3);
    #[cfg(target_arch = "x86_64")]
    {
        let runner = Runner::new(&program, &packed_input, 0x2000, config(SimdJitStrategy::RegisterCached)).unwrap();
        runner.jit_compile().unwrap();
        let stats = runner.executable.get_compiled_program().unwrap().simd_stats;
        assert_eq!(stats.intrinsic_slow_paths, 2, "{stats:?}");
    }
}

/// Malformed packed operands fail identically on every engine
#[test]
fn intrinsic_malformed_operands() {
    let input = random_input(22);
    for (subop, packed) in [
        (solana_sbpf::simd::VADD32, -1i64),                   // high bits set
        (solana_sbpf::simd::VADD32, 1 << 24),                 // non canonical bit
        (solana_sbpf::simd::VLOAD, 250 | 7 << 24),            // 8 slots from v250 run past the last slot
        (solana_sbpf::simd::VEXT64, 4 << 8),                  // lane out of range
        (solana_sbpf::simd::VBCAST32, 1 << 8),                // vbcast32 has no source slot
        (solana_sbpf::simd::VROR32, 40 << 16),                // rotation out of range
        (solana_sbpf::simd::VSRL32, 32 << 16),                // shift out of range
        (solana_sbpf::simd::VTRANSPOSE8, 250),                // transpose runs past the last slot
    ] {
        let mut a = Asm::new(VecMode::Call);
        a.mov64_imm(1, packed);
        a.raw(ebpf::CALL_IMM, 0, 0, 0, intrinsics::hash(subop, false) as i32 as i64);
        a.exit();
        let (result, ..) = run_all(&a.finish(), &input);
        assert!(result.contains("UnsupportedInstruction"), "{subop} {packed:#x}: {result}");
    }
}

/// A runtime which doesn't know the intrinsics rejects the program at deployment (unknown static
/// syscall), it is never misinterpreted
#[test]
fn intrinsics_unknown_to_runtime_fail_verification() {
    use solana_sbpf::{
        elf::Executable,
        program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
        simd::harness::HarnessContext,
        verifier::{LocalVerifier, VerifierError},
    };
    let mut a = Asm::new(VecMode::Call);
    a.vadd(0, 1, 2);
    a.exit();
    let program = a.finish();
    let verify = |register: bool| {
        let mut loader = BuiltinProgram::new_loader(config(SimdJitStrategy::Direct));
        if register {
            intrinsics::register(&mut loader).unwrap();
        }
        Executable::<HarnessContext>::from_text_bytes(&program, std::sync::Arc::new(loader), SBPFVersion::V4, FunctionRegistry::default())
            .unwrap()
            .verify::<LocalVerifier>()
    };
    let hash = intrinsics::hash(solana_sbpf::simd::VADD32, false);
    let result = verify(false);
    assert!(matches!(result, Err(EbpfError::VerifierError(VerifierError::InvalidSyscall(h))) if h == hash), "{:?}", result);
    assert!(verify(true).is_ok());
}

/// New arithmetic ops against a Rust model, and slots beyond 63 (call form only)
#[test]
fn arithmetic_ops_and_high_slots() {
    use solana_sbpf::simd::read_lanes;
    let input = random_input(30);
    let lanes = |i: usize| read_lanes(input[32 * i..].as_ptr());
    let (x, y) = (lanes(0), lanes(1));
    let mut a = Asm::new(VecMode::Call);
    a.mov64(3, 1);
    a.vload(200, 3, 0);
    a.vload(255, 3, 32);
    a.vsub(130, 200, 255);
    a.vmullo(131, 200, 255);
    a.vminu(132, 200, 255);
    a.vsrl(133, 200, 7);
    a.vsll(134, 255, 9);
    a.vtranspose(240);
    a.mov64_imm(4, 13);
    a.vbcast(150, 4);
    a.vsllv(135, 200, 150);
    a.vsrlv(136, 255, 150);
    a.vandn(137, 200, 255);
    // Operations which may take the host fallback, on slots beyond 63
    a.vop_full(solana_sbpf::simd::VMADD52HI, false, 138, 200, 255, 0, 0, 0);
    a.vop_full(solana_sbpf::simd::VCMP, false, 0, 200, 255, 0, 0, 0x21);
    a.vins64(139, 0, 1);
    a.lddw(2, ebpf::MM_HEAP_START);
    for (i, slot) in [130u8, 131, 132, 133, 134, 247, 135, 136, 137, 138, 139].iter().enumerate() {
        a.vstore(2, 32 * i as i16, *slot);
    }
    a.mov64_imm(0, 0);
    a.exit();
    let (result, heap, _, _) = run_all(&a.finish(), &input);
    assert_eq!(result, "Ok(0)");
    let out = |i: usize| read_lanes(heap[32 * i..].as_ptr());
    for l in 0..8 {
        assert_eq!(out(0)[l], x[l].wrapping_sub(y[l]));
        assert_eq!(out(1)[l], x[l].wrapping_mul(y[l]));
        assert_eq!(out(2)[l], x[l].min(y[l]));
        assert_eq!(out(3)[l], x[l] >> 7);
        assert_eq!(out(4)[l], y[l] << 9);
        assert_eq!(out(6)[l], x[l] << 13);
        assert_eq!(out(7)[l], y[l] >> 13);
        assert_eq!(out(8)[l], !x[l] & y[l]);
    }
    // vtranspose8 v240..v247: v247 lane 0 is v240 lane 7, which is zero (the file starts zeroed)
    assert_eq!(out(5), [0; 8]);
    let q = |i: usize, l: usize| out(i)[2 * l] as u64 | (out(i)[2 * l + 1] as u64) << 32;
    let xq = |l: usize| x[2 * l] as u64 | (x[2 * l + 1] as u64) << 32;
    let yq = |l: usize| y[2 * l] as u64 | (y[2 * l + 1] as u64) << 32;
    let m52 = (1u64 << 52) - 1;
    for l in 0..4 {
        assert_eq!(q(9, l), (((xq(l) & m52) as u128 * (yq(l) & m52) as u128) >> 52) as u64);
    }
    let lt: u64 = (0..8).filter(|&l| (x[l] as i32) < (y[l] as i32)).map(|l| 1 << l).sum();
    assert_eq!(q(10, 1), lt);
}

/// callx may enter functions but not the middle of a run of vector code (where the JIT keeps vector
/// state in registers); programs with callx still get the register cache
#[test]
fn callx_into_vector_code() {
    let input = random_input(31);
    let build = |vaddr: u64, mid: bool| {
        let mut a = Asm::new(VecMode::Slot);
        let target = vaddr + 8 * if mid { 8 } else { 7 };
        a.lddw(2, ebpf::MM_HEAP_START); // pc 0..2
        a.lddw(3, target); // pc 2..4
        a.raw(ebpf::CALL_REG, 3, 0, 0, 0); // pc 4 (callx r3, dst register in V4)
        a.mov64_imm(0, 5); // pc 5
        a.exit(); // pc 6
        a.vadd(0, 0, 1); // pc 7: function entry
        a.vadd(0, 0, 1); // pc 8: inside the vector run
        a.vstore(2, 0, 0);
        a.exit();
        a.finish()
    };
    let vaddr = Runner::new(&build(0, false), &input, 0x2000, config(SimdJitStrategy::Direct)).unwrap().executable.get_text_bytes().0;
    let (result, ..) = run_all(&build(vaddr, false), &input);
    assert_eq!(result, "Ok(5)");
    let (result, ..) = run_all(&build(vaddr, true), &input);
    assert!(result.contains("UnsupportedInstruction"), "{result}");
    #[cfg(target_arch = "x86_64")]
    {
        let runner = Runner::new(&build(vaddr, false), &input, 0x2000, config(SimdJitStrategy::RegisterCached)).unwrap();
        runner.jit_compile().unwrap();
        let stats = runner.executable.get_compiled_program().unwrap().simd_stats;
        assert!(stats.cache_hits > 0, "{stats:?}");
    }
}

/// The prebuilt Falcon-512 programs (simd-falcon/fixtures) verify a PQClean signature and reject a
/// tampered one, with identical CUs in every engine; the vector program lowers every intrinsic
/// call inline and uses the register cache
#[test]
fn falcon_programs() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("simd-falcon/fixtures");
    let bytes = std::fs::read(dir.join("sigs.bin")).unwrap();
    let len = u16::from_le_bytes([bytes[666], bytes[667]]) as usize;
    let (sig, msg) = (&bytes[..666], &bytes[668..668 + len]);
    let input = |sig: &[u8], msg: &[u8]| {
        let mut v = vec![0u8; 8];
        v.extend_from_slice(&((666 + msg.len()) as u64).to_le_bytes());
        v.extend_from_slice(sig);
        v.extend_from_slice(msg);
        v.extend_from_slice(&[7; 32]);
        v
    };
    let mut bad = msg.to_vec();
    bad[0] ^= 1;
    for name in ["falcon_simd.so", "falcon_blueshift.so"] {
        let elf = std::fs::read(dir.join(name)).unwrap();
        for (data, expected) in [(input(sig, msg), "Ok(0)"), (input(sig, &bad), "Ok(3)")] {
            let mut reference = None;
            for (engine, strategy) in engines() {
                let mut r = Runner::from_elf(&elf, &data, 32 * 1024, config(strategy)).unwrap();
                r.budget = 1_400_000;
                if engine == Engine::Jit {
                    r.jit_compile().unwrap();
                }
                let (cus, result) = r.run(engine);
                assert_eq!(format!("{result:?}"), expected, "{name} {engine:?} {strategy:?}");
                assert_eq!(*reference.get_or_insert(cus), cus, "{name} CUs differ");
                #[cfg(target_arch = "x86_64")]
                if engine == Engine::Jit && name == "falcon_simd.so" {
                    let stats = r.executable.get_compiled_program().unwrap().simd_stats;
                    assert_eq!(stats.intrinsic_slow_paths, 0);
                    // Table rows (twiddles, prepared pubkey, Keccak constants) from .rodata
                    assert!(stats.static_loads > 100 && stats.translations < 16, "{:?}", stats);
                    if strategy == SimdJitStrategy::RegisterCached {
                        assert!(stats.cache_hits > 3_000, "{:?}", stats);
                    }
                }
            }
        }
    }
}

fn falcon_input() -> Vec<u8> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("simd-falcon/fixtures");
    let bytes = std::fs::read(dir.join("sigs.bin")).unwrap();
    let len = u16::from_le_bytes([bytes[666], bytes[667]]) as usize;
    let mut v = vec![0u8; 8];
    v.extend_from_slice(&((666 + len) as u64).to_le_bytes());
    v.extend_from_slice(&bytes[..666]);
    v.extend_from_slice(&bytes[668..668 + len]);
    v.extend_from_slice(&[7; 32]);
    v
}

/// AVX-512 code generation (forced, so it is also exercised on hosts without AVX-512) never runs
/// out of registers or text space; on AVX-512 hosts the result must match the AVX2 lowering
#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_lowering() {
    let host_avx512 = std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vl");
    let cfg = |strategy, avx512: bool| {
        let mut c = config(strategy);
        c.simd_avx512 = avx512;
        c.simd_force_avx512 = avx512;
        c
    };
    let mut programs: Vec<(String, Vec<u8>, Vec<u8>, bool)> = Vec::new();
    for seed in 0..2 {
        let input = random_input(seed);
        for (name, body) in op_cases() {
            for mode in [VecMode::Slot, VecMode::Call] {
                programs.push((format!("{name} {mode:?} {seed}"), op_program(mode, &*body), input.clone(), false));
            }
        }
    }
    for kernel in [Kernel::Simd1, Kernel::Simd8] {
        programs.push((format!("blake3 {kernel:?}"), blake3_program::hash_program(kernel, VecMode::Call), blake3_program::build_input(&test_data(5000, 1)), false));
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("simd-falcon/fixtures");
    programs.push(("falcon".into(), std::fs::read(dir.join("falcon_simd.so")).unwrap(), falcon_input(), true));
    for (name, program, input, elf) in &programs {
        for strategy in [SimdJitStrategy::Direct, SimdJitStrategy::RegisterCached] {
            let mut results = Vec::new();
            for avx512 in [false, true] {
                let mut runner = if *elf {
                    Runner::from_elf(program, input, 32 * 1024, cfg(strategy, avx512)).unwrap()
                } else {
                    Runner::new(program, input, HEAP_SIZE, cfg(strategy, avx512)).unwrap()
                };
                runner.budget = 2_000_000;
                runner.jit_compile().unwrap_or_else(|e| panic!("{name} {strategy:?} avx512={avx512}: {e:?}"));
                // AVX512_DUMP_DIR=... writes the machine code for inspection (e.g. llvm-objdump)
                if let (Ok(dir), true) = (std::env::var("AVX512_DUMP_DIR"), avx512 && strategy == SimdJitStrategy::RegisterCached) {
                    let file = name.replace(' ', "_");
                    std::fs::write(format!("{dir}/{file}.bin"), runner.executable.get_compiled_program().unwrap().text_section()).unwrap();
                }
                if avx512 && !host_avx512 {
                    continue;
                }
                let (cus, result) = runner.run(Engine::Jit);
                results.push((format!("{result:?}"), cus, runner.heap().to_vec(), runner.vector.as_slice().to_vec()));
            }
            if results.len() == 2 {
                assert!(results[0] == results[1], "{name} {strategy:?}: AVX-512 differs from AVX2");
            }
        }
    }
}


/// Inputs with edge values in every lane width: 0, 1, -1, the minimum and maximum of each width,
/// and shift counts around 31 / 32 / 63 / 64
fn edge_input(variant: u64) -> Vec<u8> {
    let patterns: [u64; 12] = [
        0, 1, u64::MAX, 0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff, 0x8000_0000_8000_0000,
        0x7fff_ffff_7fff_ffff, 0x8000_8000_8000_8000, 0x8080_8080_8080_8080, 31 | 32 << 32, 63 | 64 << 32, 64,
    ];
    let mut data = random_input(variant);
    for (i, chunk) in data.chunks_mut(8).enumerate() {
        if (i as u64 + variant) % 3 != 0 {
            chunk.copy_from_slice(&patterns[(i * 7 + variant as usize) % patterns.len()].to_le_bytes());
        }
    }
    data
}

/// Every lowering on this host gives the same results and CUs as the interpreter: the default JIT
/// (AVX-512, or the AVX2 templates), the JIT restricted to AVX2 and the JIT running every operation
/// through the reference implementation, on random and edge value inputs
#[cfg(target_arch = "x86_64")]
#[test]
fn tiers_agree() {
    let host_avx512 = std::is_x86_feature_detected!("avx512f");
    let in_range = solana_sbpf::simd::harness::running_under_rosetta();
    let mut inputs: Vec<Vec<u8>> = (0..3).map(random_input).collect();
    inputs.extend((0..4).map(edge_input));
    for (k, input) in inputs.iter().enumerate() {
        for (name, body) in op_cases() {
            // Rosetta 2 gets vpsllvq / vpsrlvq wrong for some counts above 63
            if in_range && k >= 3 && name.starts_with("int64") {
                continue;
            }
            for mode in [VecMode::Slot, VecMode::Call] {
                let program = op_program(mode, &*body);
                let (expected, cus) = run_op(&program, input, Engine::Interpreter, SimdJitStrategy::Direct);
                let mut tiers = vec![("default", true, false), ("reference", true, true)];
                if host_avx512 {
                    tiers.push(("avx2", false, false));
                }
                for (tier, avx512, reference) in tiers {
                    for strategy in [SimdJitStrategy::Direct, SimdJitStrategy::RegisterCached] {
                        let mut c = config(strategy);
                        c.simd_avx512 = avx512;
                        c.simd_force_reference = reference;
                        let mut runner = Runner::new(&program, input, 0x2000, c).unwrap();
                        runner.jit_compile().unwrap();
                        let (jit_cus, result) = runner.run(Engine::Jit);
                        assert!(result.is_ok(), "{name} {tier} {result:?}");
                        assert!(runner.heap()[..16 * 32] == expected[..], "{name} {mode:?} {tier} {strategy:?} input {k}");
                        assert_eq!(jit_cus, cus, "{name} {tier}");
                    }
                }
            }
        }
    }
}

/// Without AVX-512, only 8 / 16 bit and 512 bit compress / expand take the host call
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_templates_cover_the_avx512_operations() {
    use solana_sbpf::simd::*;
    let mut a = Asm::new(VecMode::Slot);
    a.lddw(3, 0xa5);
    for wide in [false, true] {
        for subop in [VSRA64, VMULLO64, VMADD52LO, VMADD52HI] {
            a.vop_full(subop, wide, 0, 2, 4, 0, 0, if subop == VSRA64 { 40 } else { 0 });
        }
        for size in 0..4i16 {
            a.vop_full(VCMP, wide, 0, 2, 4, 5, 0, 1 | size << 4);
            a.vop_full(VBLENDM, wide, 0, 2, 4, 0, 3, size);
        }
    }
    for size in [2i16, 3] {
        a.vop_full(VCOMPRESS, false, 0, 2, 0, 5, 3, size);
        a.vop_full(VEXPAND, false, 0, 2, 0, 0, 3, size);
    }
    a.vop_full(VCOMPRESS, false, 0, 2, 0, 5, 3, 1); // 16 bit: host call
    a.vop_full(VEXPAND, true, 0, 2, 0, 0, 3, 2); // 512 bit: host call
    a.exit();
    let mut c = config(SimdJitStrategy::RegisterCached);
    c.simd_avx512 = false;
    let runner = Runner::new(&a.finish(), &random_input(1), 0x2000, c).unwrap();
    runner.jit_compile().unwrap();
    let stats = runner.executable.get_compiled_program().unwrap().simd_stats;
    assert_eq!(stats.intrinsic_slow_paths, 2, "{stats:?}");
}

// ---------------------------------------------------------------------------------------------
// Exhaustive / systematic tests: an operation runs over a stream of records (a, b, d operands and
// a mask) in one loop; the interpreter must match an independent model written here, and every
// lowering of the host (default JIT, forced reference, AVX2 only on AVX-512 hosts) must match the
// interpreter bit for bit, CUs included.

/// Runs `subop` (width `wide`, immediate `off`) over `records` (each a ++ b ++ d ++ u64 mask),
/// returns per record the result vector ++ the scalar result (u64)
fn stream_op(subop: u8, wide: bool, off: i16, records: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use solana_sbpf::simd::{uses_src, writes_dst};
    let h = 1 + wide as u8;
    let n = 32 * h as usize;
    let mut a = Asm::new(VecMode::Call);
    a.mov64(6, 1);
    a.lddw(7, ebpf::MM_HEAP_START);
    a.mov64_imm(8, records.len() as i64);
    let top = a.new_label();
    a.bind(top);
    a.vload_n(0, 3 * h, 6, 0);
    a.ldxdw(3, 6, 3 * n as i16);
    a.mov64_imm(4, 0);
    let (src, dst) = (if uses_src(subop) { 3 } else { 0 }, if writes_dst(subop) { 4 } else { 0 });
    a.vop_full(subop, wide, 2 * h, 0, h, dst, src, off);
    a.vstore_n(7, 0, 2 * h, h);
    a.stxdw(7, n as i16, 4);
    a.add64_imm(6, 3 * n as i64 + 8);
    a.add64_imm(7, n as i64 + 8);
    a.sub64_imm(8, 1);
    a.jne_imm(8, 0, top);
    a.mov64_imm(0, 0);
    a.exit();
    let program = a.finish();
    let input: Vec<u8> = records.iter().flatten().copied().collect();
    let heap_size = records.len() * (n + 8) + 64;
    let run = |engine: Engine, strategy, avx512: bool, reference: bool| {
        let mut c = config(strategy);
        c.simd_avx512 = avx512;
        c.simd_force_reference = reference;
        let mut runner = Runner::new(&program, &input, heap_size, c).unwrap();
        runner.budget = 1 << 32;
        if engine == Engine::Jit {
            runner.jit_compile().unwrap();
        }
        let (cus, result) = runner.run(engine);
        assert!(result.is_ok(), "{} {result:?}", solana_sbpf::simd::mnemonic(subop, wide));
        (runner.heap()[..records.len() * (n + 8)].to_vec(), cus)
    };
    let (expected, cus) = run(Engine::Interpreter, SimdJitStrategy::Direct, true, false);
    #[cfg(target_arch = "x86_64")]
    {
        let mut tiers = vec![(SimdJitStrategy::RegisterCached, true, false), (SimdJitStrategy::Direct, true, false), (SimdJitStrategy::Direct, true, true)];
        if std::is_x86_feature_detected!("avx512f") {
            tiers.push((SimdJitStrategy::RegisterCached, false, false));
        }
        for (strategy, avx512, reference) in tiers {
            let (got, jit_cus) = run(Engine::Jit, strategy, avx512, reference);
            assert!(got == expected, "{} off {off:#x}: {strategy:?} avx512={avx512} reference={reference} differs from the interpreter", solana_sbpf::simd::mnemonic(subop, wide));
            assert_eq!(jit_cus, cus);
        }
    }
    expected.chunks(n + 8).map(|c| c.to_vec()).collect()
}

/// Records from operand generators (`a`, `b`, `d` get the record index and the operand size)
fn records(count: usize, n: usize, f: &dyn Fn(usize, usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, u64)) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| {
            let (a, b, d, m) = f(i, n);
            assert!(a.len() == n && b.len() == n && d.len() == n);
            [a, b, d, m.to_le_bytes().to_vec()].concat()
        })
        .collect()
}

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn rand_bytes(r: &mut dyn FnMut() -> u64, n: usize) -> Vec<u8> {
    (0..n).map(|_| r() as u8).collect()
}

/// Values worth hitting in lanes of `size` bytes, mixed with random ones
fn edge_lanes(r: &mut dyn FnMut() -> u64, n: usize, size: usize) -> Vec<u8> {
    let bits = 8 * size as u32;
    let max = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let edges = [0, 1, 2, max, max >> 1, (max >> 1) + 1, 15, 16, 31, 32, 33, 63, 64, 65, 0x55 * (max / 0xff), r() & max];
    (0..n / size).flat_map(|_| {
        let v = if r() % 3 == 0 { r() & max } else { edges[(r() % edges.len() as u64) as usize] };
        v.to_le_bytes()[..size].to_vec()
    }).collect()
}

fn lanes(x: &[u8], size: usize) -> Vec<u64> {
    x.chunks(size).map(|c| c.iter().rev().fold(0u64, |v, b| v << 8 | *b as u64)).collect()
}

fn sext(v: u64, size: usize) -> i64 {
    ((v << (64 - 8 * size)) as i64) >> (64 - 8 * size)
}

fn model_gf_mul(a: u8, b: u8) -> u8 {
    // Schoolbook carry-less product, then reduction modulo 0x11b
    let mut p = 0u16;
    for i in 0..8 {
        if (b >> i) & 1 != 0 {
            p ^= (a as u16) << i;
        }
    }
    for i in (8..15).rev() {
        if (p >> i) & 1 != 0 {
            p ^= 0x11b << (i - 8);
        }
    }
    p as u8
}

#[test]
fn exhaustive_byte_ops() {
    use solana_sbpf::simd::*;
    for wide in [false, true] {
        let n = 32 * (1 + wide as usize);
        // gf2p8mul: every pair of bytes
        let pairs = 65536 / n;
        let recs = records(pairs, n, &|i, n| {
            let a: Vec<u8> = (0..n).map(|k| ((i * n + k) >> 8) as u8).collect();
            let b: Vec<u8> = (0..n).map(|k| (i * n + k) as u8).collect();
            (a, b, vec![0; n], 0)
        });
        let out = stream_op(VGF2P8MUL, wide, 0, &recs);
        for (rec, o) in recs.iter().zip(&out) {
            for k in 0..n {
                assert_eq!(o[k], model_gf_mul(rec[k], rec[n + k]), "gf2p8mul {:#x} {:#x}", rec[k], rec[n + k]);
            }
        }
        // gf2p8affine: every byte under the identity, the AES affine matrix and random matrices
        for (matrix, imm) in [(0x0102_0408_1020_4080u64, 0i16), (0xf1e3_c78f_1f3e_7cf8, 0x63), (0x8040_2010_0804_0201, 0xff), (0x1234_5678_9abc_def0, 0x5a)] {
            let recs = records(256 / n * 1, n, &|i, n| {
                let a: Vec<u8> = (0..n).map(|k| (i * n + k) as u8).collect();
                (a, matrix.to_le_bytes().repeat(n / 8), vec![0; n], 0)
            });
            let out = stream_op(VGF2P8AFFINE, wide, imm, &recs);
            for (rec, o) in recs.iter().zip(&out) {
                for k in 0..n {
                    let m = matrix.to_le_bytes();
                    let expected = (0..8).fold(0u8, |r, bit| r | (((m[7 - bit] & rec[k]).count_ones() as u8 & 1) ^ ((imm as u8 >> bit) & 1)) << bit);
                    assert_eq!(o[k], expected, "gf2p8affine {matrix:#x} {imm:#x} x={:#x}", rec[k]);
                }
            }
        }
        // permb / permi2b: every index value, multishift: every control byte
        let mut r = rng(5 + wide as u64);
        let recs = records(256 / n * 2, n, &|i, n| {
            let mut r = rng(i as u64 + 99);
            ((0..n).map(|_| r() as u8).collect(), (0..n).map(|k| (i * n + k) as u8).collect(), (0..n).map(|k| ((i * n + k) as u8).rotate_left(3)).collect(), 0)
        });
        let _ = r();
        let out = stream_op(VPERMB, wide, 0, &recs);
        for (rec, o) in recs.iter().zip(&out) {
            for k in 0..n {
                assert_eq!(o[k], rec[rec[n + k] as usize % n], "permb");
            }
        }
        let out = stream_op(VPERMI2B, wide, 0, &recs);
        for (rec, o) in recs.iter().zip(&out) {
            for k in 0..n {
                let j = rec[2 * n + k] as usize % (2 * n);
                assert_eq!(o[k], if j < n { rec[j] } else { rec[n + j - n] }, "permi2b");
            }
        }
        let out = stream_op(VMULTISHIFT, wide, 0, &recs);
        for (rec, o) in recs.iter().zip(&out) {
            for k in 0..n {
                let q = u64::from_le_bytes(rec[k & !7..(k & !7) + 8].try_into().unwrap());
                let c = rec[n + k] as u32 % 64;
                assert_eq!(o[k], ((q >> c) | if c == 0 { 0 } else { q << (64 - c) }) as u8, "multishift");
            }
        }
        // popcnt: every byte and every u16; edge and random u32 / u64
        for size in 0..4usize {
            let count = match size { 0 => 256 / n, 1 => 65536 * 2 / n, _ => 64 };
            let recs = records(count, n, &|i, n| {
                let a: Vec<u8> = match size {
                    0 => (0..n).map(|k| (i * n + k) as u8).collect(),
                    1 => (0..n / 2).flat_map(|k| ((i * n / 2 + k) as u16).to_le_bytes()).collect(),
                    _ => { let mut r = rng(i as u64); edge_lanes(&mut r, n, 1 << size) }
                };
                (a, vec![0; n], vec![0; n], 0)
            });
            let out = stream_op(VPOPCNT, wide, size as i16, &recs);
            for (rec, o) in recs.iter().zip(&out) {
                let (x, y) = (lanes(&rec[..n], 1 << size), lanes(&o[..n], 1 << size));
                for k in 0..x.len() {
                    assert_eq!(y[k], x[k].count_ones() as u64, "popcnt size {size}");
                }
            }
        }
        // lzcnt: every leading zero count
        for size in [2usize, 3] {
            let bits = 8 << size;
            let recs = records(bits + 8, n, &|i, n| {
                let mut r = rng(i as u64 * 7);
                let a: Vec<u8> = (0..n / (1 << size)).flat_map(|k| {
                    let lz = (i + k) % (bits + 1);
                    let v = if lz == bits { 0 } else { (1u64 << (bits - 1 - lz)) | (r() & ((1u64 << (bits - 1 - lz)) - 1)) };
                    v.to_le_bytes()[..1 << size].to_vec()
                }).collect();
                (a, vec![0; n], vec![0; n], 0)
            });
            let out = stream_op(VLZCNT, wide, size as i16, &recs);
            for (rec, o) in recs.iter().zip(&out) {
                let (x, y) = (lanes(&rec[..n], 1 << size), lanes(&o[..n], 1 << size));
                for k in 0..x.len() {
                    assert_eq!(y[k], (x[k].leading_zeros() - (64 - bits as u32)) as u64, "lzcnt size {size}");
                }
            }
        }
    }
}

#[test]
fn exhaustive_shifts_and_widths() {
    use solana_sbpf::simd::*;
    let rosetta = solana_sbpf::simd::harness::running_under_rosetta();
    for wide in [false, true] {
        let n = 32 * (1 + wide as usize);
        let data = |seed: u64, size: usize| records(24, n, &|i, n| {
            let mut r = rng(seed * 1000 + i as u64);
            (edge_lanes(&mut r, n, size), edge_lanes(&mut r, n, size), edge_lanes(&mut r, n, size), r())
        });
        // Immediate shifts and rotates: every count
        for (subop, size, counts) in [(VSLL16, 2, 16), (VSRL16, 2, 16), (VSRA16, 2, 16), (VSRA32, 4, 32), (VROR64, 8, 64)] {
            let recs = data(subop as u64, size);
            for c in 0..counts {
                let out = stream_op(subop, wide, c as i16, &recs);
                for (rec, o) in recs.iter().zip(&out) {
                    let (x, y) = (lanes(&rec[..n], size), lanes(&o[..n], size));
                    for k in 0..x.len() {
                        let bits = 8 * size as u32;
                        let mask = if bits == 64 { u64::MAX } else { (1 << bits) - 1 };
                        let e = match subop {
                            VSLL16 => (x[k] << c) & mask,
                            VSRL16 => x[k] >> c,
                            VSRA16 | VSRA32 => (sext(x[k], size) >> c) as u64 & mask,
                            _ => x[k].rotate_right(c),
                        };
                        assert_eq!(y[k], e, "{} count {c}", mnemonic(subop, wide));
                    }
                }
            }
        }
        // Variable shifts: counts 0..=70 and huge ones in every lane
        for (subop, size) in [(VSLLV16, 2), (VSRLV16, 2), (VSRAV16, 2), (VSRAV32, 4), (VSRAV64, 8)] {
            let recs = records(48, n, &|i, n| {
                let mut r = rng(i as u64 + 31 * subop as u64);
                let counts: Vec<u8> = (0..n / size).flat_map(|k| {
                    let c: u64 = match (i + k) % 4 {
                        // Rosetta 2 gets vpsrlvq wrong for counts above 63 with small low bits
                        3 if !(rosetta && size == 8) => r(),
                        _ => ((i * 3 + k) % 71) as u64,
                    };
                    c.to_le_bytes()[..size].to_vec()
                }).collect();
                (edge_lanes(&mut r, n, size), counts, vec![0; n], 0)
            });
            let out = stream_op(subop, wide, 0, &recs);
            for (rec, o) in recs.iter().zip(&out) {
                let (x, c, y) = (lanes(&rec[..n], size), lanes(&rec[n..2 * n], size), lanes(&o[..n], size));
                let bits = 8 * size as u64;
                let mask = if bits == 64 { u64::MAX } else { (1 << bits) - 1 };
                for k in 0..x.len() {
                    let e = match subop {
                        VSLLV16 => if c[k] >= bits { 0 } else { (x[k] << c[k]) & mask },
                        VSRLV16 => if c[k] >= bits { 0 } else { x[k] >> c[k] },
                        _ => (sext(x[k], size) >> c[k].min(bits - 1)) as u64 & mask,
                    };
                    assert_eq!(y[k], e, "{} x {:#x} count {}", mnemonic(subop, wide), x[k], c[k]);
                }
            }
        }
        // Unpacks: every lane size
        for size in 0..4usize {
            let recs = data(77 + size as u64, 1 << size);
            for subop in [VUNPCKLO, VUNPCKHI] {
                let out = stream_op(subop, wide, size as i16, &recs);
                for (rec, o) in recs.iter().zip(&out) {
                    let s = 1 << size;
                    let mut e = Vec::new();
                    for lane in 0..n / 16 {
                        let base = 16 * lane + if subop == VUNPCKHI { 8 } else { 0 };
                        for k in 0..8 / s {
                            e.extend_from_slice(&rec[base + s * k..base + s * (k + 1)]);
                            e.extend_from_slice(&rec[n + base + s * k..n + base + s * (k + 1)]);
                        }
                    }
                    assert_eq!(o[..n], e[..], "{} size {size}", mnemonic(subop, wide));
                }
            }
        }
        // Packs: every i16 for 16 -> 8, edge i32 for 32 -> 16
        for (size, count) in [(2usize, 65536 * 2 / n / 2), (4, 64)] {
            let recs = records(count, n, &|i, n| {
                let mut r = rng(i as u64 + 5);
                if size == 2 {
                    let a: Vec<u8> = (0..n / 2).flat_map(|k| ((i * n + k) as u16).to_le_bytes()).collect();
                    let b: Vec<u8> = (0..n / 2).flat_map(|k| ((i * n + n / 2 + k) as u16).to_le_bytes()).collect();
                    (a, b, vec![0; n], 0)
                } else {
                    (edge_lanes(&mut r, n, 4), edge_lanes(&mut r, n, 4), vec![0; n], 0)
                }
            });
            for subop in [VPACKSS, VPACKUS] {
                let out = stream_op(subop, wide, (size / 4) as i16, &recs);
                for (rec, o) in recs.iter().zip(&out) {
                    let narrow = size / 2;
                    let (lo, hi) = if subop == VPACKSS { (-(1i64 << (8 * narrow - 1)), (1i64 << (8 * narrow - 1)) - 1) } else { (0, (1i64 << (8 * narrow)) - 1) };
                    let mut e = Vec::new();
                    for lane in 0..n / 16 {
                        for src in [&rec[..n], &rec[n..2 * n]] {
                            for v in lanes(&src[16 * lane..16 * lane + 16], size) {
                                e.extend_from_slice(&(sext(v, size).clamp(lo, hi) as u64).to_le_bytes()[..narrow]);
                            }
                        }
                    }
                    assert_eq!(o[..n], e[..], "{} size {size}", mnemonic(subop, wide));
                }
            }
        }
        // Zero / sign extension: all six widenings, every byte / u16
        for (from, to) in [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            let recs = records(65536 * 2 / n / 4, n, &|i, n| {
                let a: Vec<u8> = (0..n / 2).flat_map(|k| ((i * n / 2 + k) as u16).to_le_bytes()).collect();
                let mut r = rng(i as u64);
                (if from == 2 { edge_lanes(&mut r, n, 4) } else { a }, vec![0; n], vec![0; n], 0)
            });
            for subop in [VZEXT, VSEXT] {
                let out = stream_op(subop, wide, (from | to << 2) as i16, &recs);
                for (rec, o) in recs.iter().zip(&out) {
                    let (fs, ts) = (1usize << from, 1usize << to);
                    let src = lanes(&rec[..n * fs / ts], fs);
                    let e: Vec<u8> = src.iter().flat_map(|v| {
                        let w = if subop == VSEXT { sext(*v, fs) as u64 } else { *v };
                        w.to_le_bytes()[..ts].to_vec()
                    }).collect();
                    assert_eq!(o[..n], e[..], "{} {from}->{to}", mnemonic(subop, wide));
                }
            }
        }
    }
}

#[test]
fn exhaustive_masks_and_compress() {
    use solana_sbpf::simd::*;
    for wide in [false, true] {
        let n = 32 * (1 + wide as usize);
        // compress / expand: every mask of the u32 / u64 lanes of a 256 bit operand, random masks
        // otherwise (and for 8 / 16 bit lanes)
        for size in 0..4usize {
            let lanes_n = n >> size;
            let every_mask = lanes_n <= 16;
            let count = if every_mask { 1 << lanes_n } else { 128 };
            let recs = records(count, n, &|i, n| {
                let mut r = rng(i as u64 + 1);
                let mask = if every_mask { i as u64 } else { r() };
                (rand_bytes(&mut r, n), rand_bytes(&mut r, n), vec![0; n], mask)
            });
            for subop in [VCOMPRESS, VEXPAND, VBLENDM] {
                let out = stream_op(subop, wide, size as i16, &recs);
                for (rec, o) in recs.iter().zip(&out) {
                    let s = 1 << size;
                    let mask = u64::from_le_bytes(rec[3 * n..3 * n + 8].try_into().unwrap());
                    let (a, b) = (lanes(&rec[..n], s), lanes(&rec[n..2 * n], s));
                    let bit = |k: usize| (mask >> k) & 1 != 0;
                    let kept: Vec<u64> = (0..a.len()).filter(|&k| bit(k)).map(|k| a[k]).collect();
                    let mut j = 0;
                    let e: Vec<u64> = (0..a.len()).map(|k| match subop {
                        VCOMPRESS => kept.get(k).copied().unwrap_or(0),
                        VEXPAND => if bit(k) { j += 1; a[j - 1] } else { 0 },
                        _ => if bit(k) { a[k] } else { b[k] },
                    }).collect();
                    assert_eq!(lanes(&o[..n], s), e, "{} size {size} mask {mask:#x}", mnemonic(subop, wide));
                    if subop == VCOMPRESS {
                        assert_eq!(u64::from_le_bytes(o[n..n + 8].try_into().unwrap()), kept.len() as u64);
                    }
                }
            }
        }
        // cmp: every size x predicate x signedness on edge values with many equal lanes
        for size in 0..4usize {
            let recs = records(32, n, &|i, n| {
                let mut r = rng(i as u64 + 1000);
                let a = edge_lanes(&mut r, n, 1 << size);
                let mut b = edge_lanes(&mut r, n, 1 << size);
                for k in (0..n).step_by(3 << size) {
                    b[k..k + (1 << size)].copy_from_slice(&a[k..k + (1 << size)].to_vec());
                }
                (a, b, vec![0; n], 0)
            });
            for pred in 0..16i16 {
                let out = stream_op(VCMP, wide, pred | (size as i16) << 4, &recs);
                for (rec, o) in recs.iter().zip(&out) {
                    let s = 1 << size;
                    let (a, b) = (lanes(&rec[..n], s), lanes(&rec[n..2 * n], s));
                    let mut mask = 0u64;
                    for k in 0..a.len() {
                        let ord = if pred & 8 != 0 { a[k].cmp(&b[k]) } else { sext(a[k], s).cmp(&sext(b[k], s)) };
                        let t = [ord.is_eq(), ord.is_lt(), ord.is_le(), false, ord.is_ne(), ord.is_ge(), ord.is_gt(), true][(pred & 7) as usize];
                        mask |= (t as u64) << k;
                    }
                    assert_eq!(u64::from_le_bytes(o[n..n + 8].try_into().unwrap()), mask, "cmp size {size} pred {pred}");
                }
            }
        }
    }
}

/// AES rounds: the FIPS-197 C.1 example (AES-128) encrypted with aesenc / aesenclast and decrypted
/// with aesdec / aesdeclast (equivalent inverse cipher) in every 128 bit lane; random rounds and
/// carry-less products through every lowering
#[test]
fn aes_and_clmul() {
    use solana_sbpf::simd::*;
    // Test side AES helpers (S-box by its definition)
    fn mul(a: u8, b: u8) -> u8 { model_gf_mul(a, b) }
    let inv = |x: u8| if x == 0 { 0 } else { (1..=255u8).find(|y| mul(x, *y) == 1).unwrap() };
    let sbox: Vec<u8> = (0..=255u8).map(|x| { let v = inv(x); v ^ v.rotate_left(1) ^ v.rotate_left(2) ^ v.rotate_left(3) ^ v.rotate_left(4) ^ 0x63 }).collect();
    let key: Vec<u8> = (0..16).collect();
    let pt: Vec<u8> = (0..16).map(|i| (i * 0x11) as u8).collect();
    let ct = [0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a];
    let mut rk: Vec<Vec<u8>> = vec![key.clone()];
    let mut rcon = 1u8;
    for _ in 0..10 {
        let p = rk.last().unwrap().clone();
        let mut t = [sbox[p[13] as usize] ^ rcon, sbox[p[14] as usize], sbox[p[15] as usize], sbox[p[12] as usize]];
        rcon = mul(rcon, 2);
        let mut k = vec![0u8; 16];
        for w in 0..4 {
            for b in 0..4 {
                k[4 * w + b] = p[4 * w + b] ^ t[b];
                t[b] = k[4 * w + b];
            }
        }
        rk.push(k);
    }
    let inv_mix = |c: &[u8]| -> Vec<u8> {
        (0..16).map(|i| { let (r, col) = (i % 4, i / 4); (0..4).fold(0, |acc, k| acc ^ mul([14, 11, 13, 9][(k + 4 - r) % 4], c[k + 4 * col])) }).collect()
    };
    for wide in [false, true] {
        let lanes = 2 * (1 + wide as usize);
        let rep = |x: &[u8]| x.repeat(lanes);
        let h = 1 + wide as u8;
        let mut a = Asm::new(VecMode::Call);
        a.mov64(6, 1);
        a.vload_n(0, h, 6, 0); // state
        for (round, k) in rk.iter().enumerate() {
            a.vload_n(2 * h, h, 6, 32 * h as i16 * (1 + round as i16));
            let subop = match round { 0 => VXOR, 10 => VAESENCLAST, _ => VAESENC };
            a.vop_full(subop, wide, 0, 0, 2 * h, 0, 0, 0);
            let _ = k;
        }
        // Decryption of the result: state ^ rk10, aesdec with InvMixColumns(rk9..rk1), aesdeclast rk0
        let dk_base = 32 * h as i16 * 12;
        for round in 0..11i16 {
            a.vload_n(2 * h, h, 6, dk_base + 32 * h as i16 * round);
            let subop = match round { 0 => VXOR, 10 => VAESDECLAST, _ => VAESDEC };
            a.vop_full(subop, wide, 4 * h, if round == 0 { 0 } else { 4 * h }, 2 * h, 0, 0, 0);
        }
        a.lddw(7, ebpf::MM_HEAP_START);
        a.vstore_n(7, 0, 0, h);
        a.vstore_n(7, 64 * h as i16, 4 * h, h);
        a.mov64_imm(0, 0);
        a.exit();
        let mut input = rep(&pt);
        for k in &rk {
            input.extend(rep(k));
        }
        input.extend(rep(&rk[10]));
        for r in (1..10).rev() {
            input.extend(rep(&inv_mix(&rk[r])));
        }
        input.extend(rep(&rk[0]));
        let program = a.finish();
        for (engine, strategy) in engines() {
            let mut runner = Runner::new(&program, &input, 0x2000, config(strategy)).unwrap();
            if engine == Engine::Jit {
                runner.jit_compile().unwrap();
            }
            let (_, result) = runner.run(engine);
            assert!(result.is_ok());
            assert_eq!(runner.heap()[..32 * h as usize], rep(&ct)[..], "AES-128 encryption {engine:?} {strategy:?}");
            assert_eq!(runner.heap()[64 * h as usize..96 * h as usize], rep(&pt)[..], "AES-128 decryption {engine:?} {strategy:?}");
        }
        // Random rounds and products through every lowering
        let n = 32 * h as usize;
        let recs = records(64, n, &|i, n| { let mut r = rng(i as u64 + 3); (rand_bytes(&mut r, n), rand_bytes(&mut r, n), vec![0; n], 0) });
        for subop in [VAESENC, VAESENCLAST, VAESDEC, VAESDECLAST] {
            stream_op(subop, wide, 0, &recs);
        }
        for imm in [0x00i16, 0x01, 0x10, 0x11] {
            let out = stream_op(VCLMUL, wide, imm, &recs);
            for (rec, o) in recs.iter().zip(&out) {
                for lane in 0..n / 16 {
                    let q = |x: &[u8], k: usize| u64::from_le_bytes(x[16 * lane + 8 * k..16 * lane + 8 * k + 8].try_into().unwrap());
                    let (x, y) = (q(&rec[..n], imm as usize & 1), q(&rec[n..2 * n], (imm as usize >> 4) & 1));
                    let p = (0..64).filter(|i| (y >> i) & 1 != 0).fold(0u128, |r, i| r ^ (x as u128) << i);
                    assert_eq!(o[16 * lane..16 * lane + 16], p.to_le_bytes(), "clmul {imm:#x}");
                }
            }
        }
    }
}
