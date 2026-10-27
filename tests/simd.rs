#![cfg(feature = "simd-experiment")]
//! Tests of the experimental SIMD extension

use std::convert::TryInto;
use solana_sbpf::{
    simd::{
        asm::VecMode,
        blake3_program::{self, Kernel, HEAP_OUT, HEAP_SIZE},
        harness::{config, Engine, Runner},
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
                for mode in [VecMode::Slot, VecMode::Memory(9)] {
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
            for mode in [VecMode::Slot, VecMode::Memory(9)] {
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
    for i in 0..16u8 {
        a.vload(i, 1, 32 * i as i16);
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
    for amount in [0i16, 1, 7, 8, 12, 16, 31] {
        cases.push(("vror", Box::new(move |a| a.vror(2, 3, amount))));
        cases.push(("vror_inplace", Box::new(move |a| a.vror(3, 3, amount))));
    }
    cases
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
            for mode in [VecMode::Slot, VecMode::Memory(9)] {
                let program = op_program(mode, &*body);
                let (expected, cus) = run_op(&program, &input, Engine::Interpreter, SimdJitStrategy::Direct);
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
fn vector_region_bounds_and_scalar_access() {
    let input = random_input(5);
    // Scalar accesses to the first and last byte of the region work and alias slot memory
    let mut a = Asm::new(VecMode::Slot);
    a.vload(0, 1, 0);
    a.lddw(2, ebpf::MM_VECTOR_START);
    a.mov64_imm(3, 0x5a);
    a.stxb(2, 2047, 3); // last byte of v63
    a.stxb(2, 3, 3); // partial write into the cached v0
    a.vadd(1, 0, 0);
    a.ldxb(4, 2, 2047);
    a.lddw(5, ebpf::MM_HEAP_START);
    a.stxdw(5, 0x1000, 4);
    a.vstore(5, 0, 1);
    a.mov64_imm(0, 0);
    a.exit();
    let (result, heap, vector, _) = run_all(&a.finish(), &input);
    assert_eq!(result, "Ok(0)");
    assert_eq!(heap[0x1000], 0x5a);
    assert_eq!(vector[2047], 0x5a);
    let mut v0 = input[0..32].to_vec();
    v0[3] = 0x5a;
    let v0 = solana_sbpf::simd::read_lanes(v0.as_ptr());
    let v1 = solana_sbpf::simd::read_lanes(heap.as_ptr());
    for i in 0..8 {
        assert_eq!(v1[i], v0[i].wrapping_add(v0[i]));
    }

    // One past the end of the region
    let mut a = Asm::new(VecMode::Slot);
    a.lddw(2, ebpf::MM_VECTOR_START);
    a.ldxb(4, 2, 2048);
    a.exit();
    let (result, ..) = run_all(&a.finish(), &input);
    assert!(result.contains("AccessViolation"), "{}", result);

    // vload / vstore straddling the end of the region
    for store in [false, true] {
        let mut a = Asm::new(VecMode::Slot);
        a.lddw(2, ebpf::MM_VECTOR_START + 2048 - 16);
        if store {
            a.vstore(2, 0, 0);
        } else {
            a.vload(0, 2, 0);
        }
        a.exit();
        let (result, ..) = run_all(&a.finish(), &input);
        assert!(result.contains("AccessViolation"), "{}", result);
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
fn memory_mode_invalid_base_and_readonly() {
    let input = random_input(7);
    // Base register points nowhere
    let mut a = Asm::new(VecMode::Memory(9));
    a.mov64_imm(9, 0x1234);
    a.vadd(0, 1, 2);
    a.exit();
    let (result, ..) = run_all(&a.finish(), &input);
    assert!(result.contains("AccessViolation"), "{}", result);
    // Vector file in the read-only program region: loads work, the store fails
    let mut a = Asm::new(VecMode::Memory(9));
    a.mov64_imm(9, 0);
    a.vxor(0, 1, 2);
    a.exit();
    for _ in 0..16 {
        a.mov64_imm(0, 0); // make the read-only region large enough for the loads
    }
    let (result, ..) = run_all(&a.finish(), &input);
    assert!(result.contains("AccessViolation") && result.contains("Store"), "{}", result);
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
    for i in 0..48u8 {
        a.vadd(16 + (i % 40), i % 16, (i * 7) % 16);
        a.vxor(i % 16, 16 + ((i * 3) % 40), i % 16);
        a.vror(16 + ((i * 5) % 40), 16 + (i % 40), (i % 31) as i16 + 1);
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
    // No vector memory attached: UnsupportedInstruction, not a host fault
    for (engine, strategy) in engines() {
        let mut runner = Runner::new(&program, &input, 0x2000, config(strategy)).unwrap();
        if engine == Engine::Jit {
            runner.jit_compile().unwrap();
        }
        let (_, result) = runner.run_detached(engine);
        assert!(matches!(result, solana_sbpf::error::StableResult::Err(EbpfError::UnsupportedInstruction)), "{:?}", result);
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
        (0x7f << 18, 0u8, 0u8, 0i16), // unknown sub-operation
        (1 << 25, 0, 0, 0),           // reserved bit
        ((0x06 << 18) as i32, 0, 0, 32), // rotation >= 32
        ((0x08 << 18 | 60) as i32, 0, 0, 0), // transpose past the last slot
        ((0x02 << 18) as i32, 3, 0, 0), // dst register in slot mode
        ((0x02 << 18) as i32, 0, 2, 0), // unused src register
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
        configs.push((Kernel::Simd8, VecMode::Memory(9), Engine::Jit, SimdJitStrategy::Direct));
        configs.push((Kernel::Simd8, VecMode::Slot, Engine::Jit, SimdJitStrategy::Direct));
        configs.push((Kernel::Simd8, VecMode::Slot, Engine::Jit, SimdJitStrategy::RegisterCached));
        configs.push((Kernel::Simd1, VecMode::Slot, Engine::Jit, SimdJitStrategy::RegisterCached));
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
        ("memory_mode", VecMode::Memory(9), SimdJitStrategy::Direct),
        ("slot_direct", VecMode::Slot, SimdJitStrategy::Direct),
        ("slot_cached", VecMode::Slot, SimdJitStrategy::RegisterCached),
    ] {
        let program = blake3_program::microbench_program(repeat, mode);
        let mut cfg = config(strategy);
        cfg.noop_instruction_rate = 0;
        let runner = Runner::new(&program, &blake3_program::build_input(&[]), HEAP_SIZE, cfg).unwrap();
        runner.jit_compile().unwrap();
        let compiled = runner.executable.get_compiled_program().unwrap();
        let pcs = compiled.pc_section();
        // pc 8 = loop head, then 4 * repeat vector ops, sub64, jne
        let start = (pcs[8] & i32::MAX as u32) as usize;
        let end = (pcs[8 + 4 * repeat + 2] & i32::MAX as u32) as usize;
        std::fs::write(format!("{}/{}.bin", dir, name), &compiled.text_section()[start..end]).unwrap();
        std::fs::write(format!("{}/{}.stats", dir, name), format!("{:?}\n", compiled.simd_stats)).unwrap();
    }
}
