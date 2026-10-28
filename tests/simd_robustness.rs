#![cfg(all(feature = "simd-experiment", unix))]
// Most comparisons run the JIT, which only exists on x86-64
#![cfg_attr(not(target_arch = "x86_64"), allow(dead_code, unused_imports))]
//! Robustness of the vector extension: every engine and JIT tier gives the same results as the
//! interpreter (registers, CUs, guest memory, vector file) for every operation
//!
//! * at the boundary slots, with the vector file between guard pages (out of bounds accesses fault),
//! * with aliased operands and operands in every register cache state,
//! * with misaligned vector files,
//! * with every scalar register as an operand of the opcode form,
//! * on faults in the middle of vector code (the register cache is written back),
//! * at the edges of guest memory regions,
//! * under every AVX-512 feature profile (on AVX-512 hosts) and with code diversification.
//!
//! Stack alignment of the host calls is checked by the JIT itself in debug builds (every call into
//! Rust tests RSP), so run these in debug builds too.

use solana_sbpf::{
    ebpf,
    simd::{
        self,
        asm::{Asm, VecMode},
        harness::{config, Engine, GuardedVectorFile, Runner},
        intrinsics::{hash, pack, unpack, SCALAR_REG},
        VecInsn,
    },
    vm::{Config, SimdJitStrategy},
};

const INPUT_LEN: usize = ebpf::VECTOR_MEMORY_SIZE + 64;
const HEAP_LEN: usize = 0x2000;
/// Offset of the packed operands of a dynamic intrinsic call in the input
const DYNAMIC_OPERANDS: i16 = ebpf::VECTOR_MEMORY_SIZE as i16;

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// Random vector file contents with edge values in some lanes
fn vector_init(seed: u64) -> Vec<u8> {
    let mut r = rng(seed);
    let mut data: Vec<u8> = (0..ebpf::VECTOR_MEMORY_SIZE).map(|_| r() as u8).collect();
    for (i, chunk) in data.chunks_mut(8).enumerate() {
        match r() % 16 {
            0 => chunk.copy_from_slice(&u64::MAX.to_le_bytes()),
            1 => chunk.copy_from_slice(&0u64.to_le_bytes()),
            2 => chunk.copy_from_slice(&(1u64 << 63).to_le_bytes()),
            3 => chunk.copy_from_slice(&0x8000_8000_8000_8000u64.to_le_bytes()),
            4 => chunk.copy_from_slice(&0x7f80_7f80_7f80_7f80u64.to_le_bytes()),
            5 => chunk.copy_from_slice(&(i as u64 % 70).to_le_bytes()), // shift counts, indices
            _ => {}
        }
    }
    data
}

/// The JIT tiers compared with the interpreter on this host
fn jit_tiers() -> Vec<(String, Config)> {
    #[allow(unused_mut)] // only extended on x86-64
    let mut tiers = vec![
        (
            "cached".to_string(),
            config(SimdJitStrategy::RegisterCached),
        ),
        ("direct".to_string(), config(SimdJitStrategy::Direct)),
        (
            "reference".to_string(),
            Config {
                simd_force_reference: true,
                ..config(SimdJitStrategy::Direct)
            },
        ),
        (
            "diversified".to_string(),
            Config {
                noop_instruction_rate: 2,
                ..config(SimdJitStrategy::RegisterCached)
            },
        ),
        (
            "unsanitized".to_string(),
            Config {
                sanitize_user_provided_values: false,
                ..config(SimdJitStrategy::RegisterCached)
            },
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f")
        && config(SimdJitStrategy::RegisterCached).simd_avx512
    {
        tiers.push((
            "avx2".to_string(),
            Config {
                simd_avx512: false,
                ..config(SimdJitStrategy::RegisterCached)
            },
        ));
        for (name, disabled) in simd::features::PROFILES {
            tiers.push((
                name.to_string(),
                Config {
                    simd_disabled_features: disabled,
                    ..config(SimdJitStrategy::RegisterCached)
                },
            ));
        }
    }
    tiers
}

/// Where the vector file of a run lives
#[derive(Clone, Copy, Debug)]
enum Placement {
    /// Between guard pages, starting right after the leading one
    Front,
    /// Between guard pages, ending right at the trailing one
    End,
    /// `n` bytes after a page boundary (misaligned), canaries around it
    Offset(usize),
}

#[derive(Debug, PartialEq)]
struct Outcome {
    result: String,
    cus: u64,
    heap: Vec<u8>,
    vector: Vec<u8>,
}

fn run(
    program: &[u8],
    input: &[u8],
    engine: Engine,
    config: Config,
    placement: Placement,
) -> Outcome {
    let mut runner = match Runner::new(program, input, HEAP_LEN, config) {
        Ok(runner) => runner,
        Err(err) => {
            return Outcome {
                result: format!("load: {err:?}"),
                cus: 0,
                heap: vec![],
                vector: vec![],
            }
        }
    };
    if engine == Engine::Jit {
        if let Err(err) = runner.jit_compile() {
            return Outcome {
                result: format!("compile: {err:?}"),
                cus: 0,
                heap: vec![],
                vector: vec![],
            };
        }
    }
    let mut file = match placement {
        Placement::Front => GuardedVectorFile::new(false, 0),
        Placement::End => GuardedVectorFile::new(true, 0),
        Placement::Offset(n) => GuardedVectorFile::new(false, n),
    };
    let (cus, result) = unsafe { runner.run_with_vector(engine, file.ptr()) };
    assert!(
        file.canary_intact(),
        "{}",
        format!("write next to the vector file ({engine:?}, {placement:?})")
    );
    Outcome {
        result: format!("{result:?}"),
        cus,
        heap: runner.heap().to_vec(),
        vector: file.as_slice().to_vec(),
    }
}

/// Runs `program` in the interpreter and every JIT tier, `Err` describes the first divergence
fn differential(program: &[u8], input: &[u8], placement: Placement) -> Result<Outcome, String> {
    let expected = run(
        program,
        input,
        Engine::Interpreter,
        config(SimdJitStrategy::Direct),
        placement,
    );
    #[cfg(target_arch = "x86_64")]
    for (name, tier) in jit_tiers() {
        let got = run(program, input, Engine::Jit, tier, placement);
        if got != expected {
            let what = if got.result != expected.result {
                format!("result {} != {}", got.result, expected.result)
            } else if got.cus != expected.cus {
                format!("CUs {} != {}", got.cus, expected.cus)
            } else if got.heap != expected.heap {
                let i = (0..got.heap.len())
                    .find(|i| got.heap[*i] != expected.heap[*i])
                    .unwrap();
                format!("heap byte {i}")
            } else {
                let i = (0..got.vector.len())
                    .find(|i| got.vector[*i] != expected.vector[*i])
                    .unwrap();
                format!("slot {} byte {}", i / 32, i % 32)
            };
            return Err(format!("{name} {placement:?}: {what}"));
        }
    }
    Ok(expected)
}

/// Input: the vector file contents, then the packed operands of a dynamic call
fn input_for(init: &[u8], packed: u64) -> Vec<u8> {
    let mut input = init.to_vec();
    input.extend_from_slice(&packed.to_le_bytes());
    input.resize(INPUT_LEN, 0);
    input
}

/// Loads the whole vector file from the input (r6), sets the scalar operand (r2) and mask (r3)
fn prologue(a: &mut Asm, scalar: u64, mask: u64) {
    a.mov64(6, 1);
    for k in 0..8u8 {
        a.vload_n(32 * k, 32, 6, 1024 * k as i16);
    }
    a.lddw(SCALAR_REG, scalar);
    a.lddw(3, mask);
}

/// Stores r0 to the heap, exits
fn epilogue(a: &mut Asm) {
    a.lddw(7, ebpf::MM_HEAP_START);
    a.stxdw(7, HEAP_LEN as i16 - 8, 0);
    a.mov64_imm(0, 0);
    a.exit();
}

/// The packed operands if they decode (operand fields the operation doesn't have are dropped, so
/// different operands may give the same packed value: callers deduplicate)
fn encodable(op: &VecInsn) -> Option<u64> {
    let packed = pack(op);
    unpack(op.subop, op.wide, packed).map(|_| packed)
}

fn op(subop: u8, wide: bool, vd: u8, va: u8, vb: u8, off: i16) -> VecInsn {
    VecInsn {
        subop,
        vd,
        va,
        vb,
        wide,
        dst: 0,
        src: if simd::uses_src(subop) { SCALAR_REG } else { 0 },
        off,
    }
}

/// Emits the call, with the operands as a constant (lowered inline) or loaded from the input (the
/// JIT can't see them: the generic path which decodes and validates at run time)
fn emit_call(a: &mut Asm, op: &VecInsn, dynamic: bool) {
    emit_raw_call(a, op.subop, op.wide, pack(op), dynamic);
}

/// Emits a call with any packed operands (also malformed ones)
fn emit_raw_call(a: &mut Asm, subop: u8, wide: bool, packed: u64, dynamic: bool) {
    if dynamic {
        a.ldxdw(1, 6, DYNAMIC_OPERANDS);
    } else if packed <= i32::MAX as u64 {
        // Like compiled programs (an lddw would write the register cache back first)
        a.mov64_imm(1, packed as i64);
    } else {
        a.lddw(1, packed);
    }
    a.raw(ebpf::CALL_IMM, 0, 0, 0, hash(subop, wide) as i32 as i64);
}

/// Scalar operand for an operation: a heap address for the memory operations
fn scalar_for(subop: u8, r: &mut impl FnMut() -> u64) -> u64 {
    match subop {
        simd::VLOAD | simd::VSTORE | simd::VLOADM | simd::VSTOREM => ebpf::MM_HEAP_START + 32,
        _ => r(),
    }
}

/// All operations: (subop, wide) with an intrinsic
fn all_operations() -> Vec<(u8, bool)> {
    let mut ops = Vec::new();
    for subop in 0..simd::SUBOP_COUNT {
        for wide in [false, true] {
            if simd::intrinsics::name(subop, wide).is_some() {
                ops.push((subop, wide));
            }
        }
    }
    ops
}

/// Number of operations with an intrinsic
fn all_operations_count() -> usize {
    (0..simd::SUBOP_COUNT)
        .map(|subop| {
            [false, true]
                .iter()
                .filter(|w| simd::intrinsics::name(subop, **w).is_some())
                .count()
        })
        .sum()
}

/// Immediates to try: every value which selects a lane size, predicate, shuffle, slot count...
const OFF_CANDIDATES: [i16; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 0x19, 0x1b, 0x20, 0x31, 0x39, 0x63, 0x93, 0xe4,
];

/// The immediates of `OFF_CANDIDATES` which the interpreter accepts for the operation (with slots
/// 0 / 2 / 4), at most `limit`, spread over the candidates
fn valid_offs(subop: u8, wide: bool, limit: usize) -> Vec<i16> {
    let init = vector_init(1);
    let mut seen = std::collections::HashSet::new();
    let valid: Vec<i16> = OFF_CANDIDATES
        .iter()
        .copied()
        .filter(|off| {
            // The memory operations take a slot count / mask register in vb
            let vb = if matches!(
                subop,
                simd::VLOAD | simd::VSTORE | simd::VLOADM | simd::VSTOREM
            ) {
                0
            } else {
                4
            };
            let op = op(subop, wide, 8, 0, vb, *off);
            let Some(packed) = encodable(&op) else {
                return false;
            };
            if !seen.insert(packed) {
                return false;
            }
            let mut a = Asm::new(VecMode::Call);
            prologue(&mut a, scalar_for(subop, &mut rng(2)), u64::MAX);
            emit_call(&mut a, &op, false);
            epilogue(&mut a);
            let out = run(
                &a.finish(),
                &input_for(&init, packed),
                Engine::Interpreter,
                config(SimdJitStrategy::Direct),
                Placement::Front,
            );
            out.result.starts_with("Ok")
        })
        .collect();
    if valid.len() <= limit {
        return valid;
    }
    (0..limit)
        .map(|i| valid[i * (valid.len() - 1) / (limit - 1).max(1)])
        .collect()
}

/// Every operation at the boundary slots (0 and the top of the 8 / 7 bit operand fields, for 512 bit
/// forms the last even slot) in every operand role, with constant and run time operands: either all
/// engines reject it in the same way or they compute the same, and no engine touches anything
/// outside of the vector file (guard pages, canaries)
#[test]
fn boundary_slots_stay_in_the_vector_file() {
    let init = vector_init(3);
    let mut r = rng(4);
    let (mut cases, mut rejected) = (0, 0);
    for (subop, wide) in all_operations() {
        let offs = valid_offs(subop, wide, 3);
        assert!(!offs.is_empty(), "{}", simd::mnemonic(subop, wide));
        for off in offs {
            // 0, around the 6 / 7 bit field limits and the top; every one of the last 32 slots
            // (multi slot operations) with constant operands only
            let mut slot_sets: Vec<(u8, u8, u8, bool)> = Vec::new();
            for t in [
                0u8, 1, 2, 30, 31, 32, 33, 62, 63, 64, 126, 127, 128, 252, 253, 254, 255,
            ] {
                slot_sets.extend([
                    (t, 0, 0, true),
                    (0, t, 0, true),
                    (0, 0, t, true),
                    (t, t, t, true),
                ]);
            }
            for t in 224..=255u8 {
                slot_sets.extend([(t, 0, 0, false), (0, t, 0, false), (0, 0, t, false)]);
            }
            let mut seen = std::collections::HashSet::new();
            for (vd, va, vb, variants) in slot_sets {
                // Also operands which don't round trip (fields too narrow for the slot), stray bits
                // and the bits above the operand fields
                let packed = pack(&op(subop, wide, vd, va, vb, off));
                let flipped = packed ^ 1 << (r() % 32);
                let runs = if variants {
                    vec![
                        (packed, false),
                        (packed, true),
                        (flipped, false),
                        (packed | 1 << 29, false),
                    ]
                } else {
                    vec![(packed, false)]
                };
                for (packed, dynamic) in runs {
                    if !seen.insert((packed, dynamic)) {
                        continue;
                    }
                    let mut a = Asm::new(VecMode::Call);
                    prologue(&mut a, scalar_for(subop, &mut r), r());
                    emit_raw_call(&mut a, subop, wide, packed, dynamic);
                    epilogue(&mut a);
                    let program = a.finish();
                    // Against the trailing guard page if an operand is in the upper half, else against
                    // the leading one
                    let placement = if vd.max(va).max(vb) >= 128 {
                        Placement::End
                    } else {
                        Placement::Front
                    };
                    match differential(&program, &input_for(&init, packed), placement) {
                        Ok(out) => rejected += !out.result.starts_with("Ok") as usize,
                        Err(err) => panic!(
                            "{} vd {vd} va {va} vb {vb} off {off:#x} packed {packed:#x} dynamic {dynamic}: {err}",
                            simd::mnemonic(subop, wide)
                        ),
                    }
                    cases += 1;
                }
            }
        }
    }
    assert!(
        cases > 10_000 && rejected > 1_000,
        "{}",
        format!("{cases} cases, {rejected} rejected")
    );
}

/// Bulk and masked load / store with every slot count up to the end of the vector file and beyond
#[test]
fn bulk_transfers_at_the_end_of_the_vector_file() {
    use simd::{VLOAD, VLOADM, VSTORE, VSTOREM};
    let init = vector_init(5);
    let mut r = rng(6);
    for subop in [VLOAD, VSTORE, VLOADM, VSTOREM] {
        for wide in [false, true] {
            if simd::intrinsics::name(subop, wide).is_none() {
                continue;
            }
            for first in [0u8, 1, 200, 223, 224, 225, 240, 254, 255] {
                for count in [0u8, 1, 2, 15, 30, 31, 32, 33, 63, 255] {
                    let (vd, va) = if matches!(subop, VLOAD | VLOADM) {
                        (first, 0)
                    } else {
                        (0, first)
                    };
                    for off in [0i16, 1, 2, 3, 0x40] {
                        let op = op(subop, wide, vd, va, count, off);
                        let packed = pack(&op);
                        let mut a = Asm::new(VecMode::Call);
                        prologue(&mut a, ebpf::MM_HEAP_START, r());
                        emit_call(&mut a, &op, r().is_multiple_of(2));
                        epilogue(&mut a);
                        if let Err(err) =
                            differential(&a.finish(), &input_for(&init, packed), Placement::End)
                        {
                            panic!(
                                "{} first {first} count {count} off {off}: {err}",
                                simd::mnemonic(subop, wide)
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Every operation with every aliasing of its operands, with the operands cold (in memory), cached
/// and dirty, or evicted, in misaligned vector files
#[test]
fn aliased_operands_in_every_cache_state() {
    let init = vector_init(7);
    let mut r = rng(8);
    let (mut cases, mut tested) = (0, 0);
    for (subop, wide) in all_operations() {
        if matches!(
            subop,
            simd::VLOAD | simd::VSTORE | simd::VLOADM | simd::VSTOREM
        ) {
            continue;
        }
        let offs = valid_offs(subop, wide, 4);
        assert!(!offs.is_empty(), "{}", simd::mnemonic(subop, wide));
        tested += 1;
        for off in offs {
            // Low slots, and the top slots (slot 255 + 1 must not wrap anywhere)
            let top = if wide {
                (254u8, 252, 250)
            } else {
                (255u8, 253, 251)
            };
            let mut seen = std::collections::HashSet::new();
            let patterns: Vec<(u8, u8, u8)> = [(8u8, 12u8, 16u8), top]
                .iter()
                .flat_map(|&(x, y, z)| vec![(x, y, z), (x, x, y), (x, y, x), (x, y, y), (x, x, x)])
                .collect();
            for (vd, va, vb) in patterns {
                let op = op(subop, wide, vd, va, vb, off);
                let Some(packed) = encodable(&op) else {
                    continue;
                };
                if !seen.insert(packed) {
                    continue;
                }
                for state in ["cold", "dirty", "evicted"] {
                    let mut a = Asm::new(VecMode::Call);
                    prologue(&mut a, scalar_for(subop, &mut r), r());
                    if state != "cold" {
                        // vadd s, s, s: caches the slot with a value which differs from memory
                        for s in [
                            vd,
                            va,
                            vb,
                            vd | 1,
                            va | 1,
                            vb | 1,
                            vb.wrapping_add(2),
                            off as u8 & 63,
                        ] {
                            a.vop_full(simd::VADD32, false, s, s, s, 0, 0, 0);
                        }
                    }
                    if state == "evicted" {
                        // More slots than registers: the dirty operands are spilled
                        for s in 100..140u8 {
                            a.vop_full(simd::VADD32, false, s, s, s, 0, 0, 0);
                        }
                    }
                    emit_call(&mut a, &op, false);
                    // Reads the result back through the cache
                    a.vop_full(simd::VXOR, false, 200, vd, va, 0, 0, 0);
                    epilogue(&mut a);
                    let placement = Placement::Offset([0, 1, 8, 16, 31][cases % 5]);
                    if let Err(err) =
                        differential(&a.finish(), &input_for(&init, packed), placement)
                    {
                        panic!(
                            "{} vd {vd} va {va} vb {vb} off {off:#x} {state}: {err}",
                            simd::mnemonic(subop, wide)
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    // Every operation but the memory transfers
    let transfers = all_operations()
        .iter()
        .filter(|(subop, _)| {
            matches!(
                *subop,
                simd::VLOAD | simd::VSTORE | simd::VLOADM | simd::VSTOREM
            )
        })
        .count();
    assert_eq!(tested + transfers, all_operations_count());
    assert!(cases > 3_000, "{}", cases);
}

/// The opcode form (scalar registers as operands) with every dst / src register for the
/// operations with scalar inputs or outputs, including dst == src
#[test]
fn every_scalar_register_as_operand() {
    use simd::*;
    let init = vector_init(9);
    let mut r = rng(10);
    let ops: &[(u8, bool, u8, u8, u8, i16)] = &[
        (VCMP, false, 0, 2, 4, 0x01),
        (VCMP, true, 0, 2, 4, 0x19),
        (VTESTM, false, 0, 2, 4, 2),
        (VTESTM, true, 0, 2, 4, 3),
        (VMOVMSK8, false, 0, 2, 0, 0),
        (VEXT64, false, 0, 2, 0, 3),
        (VCOMPRESS, false, 6, 2, 0, 0),
        (VCOMPRESS, false, 6, 2, 0, 2),
        (VCOMPRESS, true, 6, 2, 0, 1),
        (VEXPAND, false, 6, 2, 0, 0),
        (VEXPAND, true, 6, 2, 0, 3),
        (VBLENDM, false, 6, 2, 4, 1),
        (VBLENDM, true, 6, 2, 4, 0),
        (VINS64, false, 6, 0, 0, 2),
        (VBCAST32, false, 6, 0, 0, 0),
        (VBCAST64, true, 6, 0, 0, 0),
        (VCONFLICT, true, 6, 2, 0, 2),
    ];
    for &(subop, wide, vd, va, vb, off) in ops {
        for dst in 0..10u8 {
            for src in 0..10u8 {
                if !writes_dst(subop) && dst != 0 || !uses_src(subop) && src != 0 {
                    continue;
                }
                let mut a = Asm::new(VecMode::Slot);
                a.mov64(6, 1);
                a.vload_n(0, 32, 6, 0);
                for reg in 0..10u8 {
                    a.lddw(reg, r());
                }
                a.vop_full(subop, wide, vd, va, vb, dst, src, off);
                // Every register to the heap (r9 through the stack)
                a.stxdw(10, -8, 9);
                a.lddw(9, ebpf::MM_HEAP_START);
                for reg in 0..9u8 {
                    a.stxdw(9, 8 * reg as i16, reg);
                }
                a.ldxdw(0, 10, -8);
                a.stxdw(9, 72, 0);
                a.mov64_imm(0, 0);
                a.exit();
                if let Err(err) = differential(&a.finish(), &input_for(&init, 0), Placement::Front)
                {
                    panic!("{} dst r{dst} src r{src}: {err}", mnemonic(subop, wide));
                }
            }
        }
    }
}

/// Faults (access violations, the instruction meter) in the middle of vector code leave the same
/// vector file as the interpreter: the register cache is written back before every possible throw
#[test]
fn faults_write_back_the_vector_file() {
    use simd::*;
    let init = vector_init(11);
    for fault in [
        "load",
        "store",
        "masked load",
        "masked store",
        "scalar load",
        "scalar store",
        "divide",
    ] {
        for wide in [false, true] {
            let mut a = Asm::new(VecMode::Call);
            prologue(&mut a, 0, 0x5555_5555_5555_5555);
            // Dirty cached slots
            a.vop_full(VADD32, wide, 40, 2, 4, 0, 0, 0);
            a.vop_full(VXOR, wide, 2, 40, 6, 0, 0, 0);
            a.lddw(SCALAR_REG, 8 * ebpf::MM_REGION_SIZE); // unmapped
            match fault {
                "load" => a.vop_full(VLOAD, false, 50, 0, 0, 0, SCALAR_REG, 0),
                "store" => a.vop_full(VSTORE, false, 0, 50, 0, 0, SCALAR_REG, 0),
                "masked load" => a.vop_full(VLOADM, wide, 50, 0, 0, 0, SCALAR_REG, 2),
                "masked store" => a.vop_full(VSTOREM, wide, 0, 50, 0, 0, SCALAR_REG, 2),
                "scalar load" => a.ldxdw(0, SCALAR_REG, 0),
                "scalar store" => a.stxdw(SCALAR_REG, 0, 0),
                _ => {
                    a.mov64_imm(4, 0);
                    a.raw(ebpf::DIV64_REG, 0, 4, 0, 0);
                }
            }
            epilogue(&mut a);
            let out = differential(&a.finish(), &input_for(&init, 0), Placement::Front)
                .unwrap_or_else(|err| panic!("{}", format!("{fault} wide {wide}: {err}")));
            assert!(out.result.starts_with("Err"), "{fault}: {}", out.result);
        }
    }
}

/// The instruction meter running out at any instruction (checkpoints every few instructions) leaves
/// the vector file written back: the cached JIT ends with the same state as the direct JIT, which
/// writes every result to memory. (The JIT checks the meter only at checkpoints, so neither has to
/// match the interpreter's state after this error.)
#[cfg(target_arch = "x86_64")]
#[test]
fn meter_exhaustion_writes_back_the_vector_file() {
    use simd::*;
    let init = vector_init(16);
    let mut a = Asm::new(VecMode::Call);
    prologue(&mut a, 7, 0x0f0f);
    for k in 0..40u8 {
        let s = 2 * (k % 12);
        a.vop_full(VADD32, k % 3 == 0, s, s + 2, s + 4, 0, 0, 0);
        if k % 4 == 0 {
            a.raw(ebpf::ADD64_IMM, 4, 0, 0, 1);
        }
        if k % 7 == 0 {
            a.lddw(5, k as u64);
        }
    }
    epilogue(&mut a);
    let program = a.finish();
    let input = input_for(&init, 0);
    for distance in [1, 2, 3, 5, 10_000] {
        for budget in 0..80 {
            let outcome = |strategy| {
                let mut c = config(strategy);
                c.instruction_meter_checkpoint_distance = distance;
                let mut runner = Runner::new(&program, &input, HEAP_LEN, c).unwrap();
                runner.budget = budget;
                runner.jit_compile().unwrap();
                let mut file = GuardedVectorFile::new(false, 0);
                let (_, result) = unsafe { runner.run_with_vector(Engine::Jit, file.ptr()) };
                (format!("{result:?}"), file.as_slice().to_vec())
            };
            let (direct, cached) = (
                outcome(SimdJitStrategy::Direct),
                outcome(SimdJitStrategy::RegisterCached),
            );
            assert!(
                direct == cached,
                "{}",
                format!(
                    "distance {distance} budget {budget}: {} / {}",
                    direct.0, cached.0
                )
            );
        }
    }
}

/// Bulk and masked load / store at the edges of the heap: inside, crossing the end by one byte, a
/// lane or a slot, and entirely outside; all engines fault at the same accesses
#[test]
fn transfers_at_the_edges_of_guest_memory() {
    use simd::*;
    let init = vector_init(12);
    let mut r = rng(13);
    let end = ebpf::MM_HEAP_START + HEAP_LEN as u64;
    for (subop, wide, count) in [
        (VLOAD, false, 0u8),
        (VLOAD, false, 2),
        (VSTORE, false, 0),
        (VSTORE, false, 3),
        (VLOADM, false, 0),
        (VLOADM, true, 0),
        (VSTOREM, false, 0),
        (VSTOREM, true, 0),
    ] {
        let bytes = 32 * (count as u64 + 1) * (1 + wide as u64);
        for back in [bytes + 64, bytes, bytes - 1, bytes - 4, bytes - 8, 31, 1, 0] {
            for size in 0..4i16 {
                if !matches!(subop, VLOADM | VSTOREM) && size > 0 {
                    continue;
                }
                for mask in [0u64, 1, 1 << 63, u64::MAX, r()] {
                    let (vd, va) = if matches!(subop, VLOAD | VLOADM) {
                        (60, 0)
                    } else {
                        (0, 60)
                    };
                    let op = op(subop, wide, vd, va, count, size);
                    let packed = pack(&op);
                    let mut a = Asm::new(VecMode::Call);
                    prologue(&mut a, end - back, mask);
                    emit_call(&mut a, &op, false);
                    epilogue(&mut a);
                    if let Err(err) =
                        differential(&a.finish(), &input_for(&init, packed), Placement::Front)
                    {
                        panic!(
                            "{} {back} bytes before the end, size {size}, mask {mask:#x}: {err}",
                            mnemonic(subop, wide)
                        );
                    }
                }
            }
        }
    }
}

/// Every AVX-512 feature profile generates code for every operation (executed by the tiers above on
/// AVX-512 hosts; here only generated, so this also runs on hosts without AVX-512)
#[cfg(target_arch = "x86_64")]
#[test]
fn every_feature_profile_compiles_every_operation() {
    let init = vector_init(14);
    let mut programs = Vec::new();
    for (subop, wide) in all_operations() {
        for off in valid_offs(subop, wide, 2) {
            let op = op(subop, wide, 8, 0, 4, off);
            let Some(packed) = encodable(&op) else {
                continue;
            };
            let mut a = Asm::new(VecMode::Call);
            prologue(&mut a, scalar_for(subop, &mut rng(15)), u64::MAX);
            emit_call(&mut a, &op, false);
            epilogue(&mut a);
            programs.push((a.finish(), packed));
        }
    }
    for (_, disabled) in simd::features::PROFILES
        .iter()
        .chain(&[("none", simd::features::ALL)])
    {
        for (program, packed) in &programs {
            let c = Config {
                simd_force_avx512: true,
                simd_disabled_features: *disabled,
                ..config(SimdJitStrategy::RegisterCached)
            };
            let runner = Runner::new(program, &input_for(&init, *packed), HEAP_LEN, c).unwrap();
            runner.jit_compile().unwrap();
        }
    }
}

/// Every valid scalar instruction between the operand setup of an intrinsic call and the call
/// (the JIT tracks the constant in r1 to lower calls inline), and between two vector operations
/// (R11 tracking, which scalar instructions flush the register cache, jumps to the next instruction
/// which start a block, exits in the middle). The scalar instruction writes the operand registers
/// r1 / r2 / r3 and r0 in some configurations, memory operations use heap pointers (r5, r9), large
/// immediates take the sanitized path (which uses R11).
#[test]
fn scalar_instructions_between_vector_ops() {
    use simd::*;
    let init = vector_init(17);
    let heap = ebpf::MM_HEAP_START + 64;
    // (dst, src, off, imm)
    let configs: [(u8, u8, i16, i64); 10] = [
        (1, 5, 0, 1),
        (1, 2, 0, 0x100),
        (2, 9, 8, 0x12_3457),
        (3, 1, 0, -1),
        (0, 3, 16, 3),
        (4, 1, 0, 0x7fff_0001),
        (9, 0, 24, 7),
        (5, 9, 0, 2),
        (6, 0, 0, 1),
        // r0 (the result of the first call) as the second call's operands
        (1, 0, 0, 0),
    ];
    // Vector operations exercising different lowerings and scalar operands
    let ops: [VecInsn; 7] = [
        op(VADD32, false, 8, 12, 16, 0),
        op(VCMP, false, 0, 8, 12, 1),      // writes r0
        op(VCOMPRESS, false, 20, 8, 0, 2), // mask in r3, writes r0
        op(VBCAST32, false, 22, 0, 0, 0),  // scalar in r2
        op(VTESTM, true, 0, 8, 12, 2),     // AVX2: two halves through R11
        op(VLOAD, false, 24, 0, 0, 32),    // address in r2
        op(VSRLV64, false, 26, 8, 12, 0),  // template (Rosetta workaround)
    ];
    let after = op(VXOR, false, 28, 20, 8, 0);
    let (mut valid, mut cases, mut completed) = (0, 0, 0);
    let mut opcodes: Vec<Option<u8>> = (0..=255u8)
        .filter(|opc| {
            !matches!(
                *opc,
                ebpf::CALL_IMM | ebpf::CALL_REG | ebpf::LD_DW_IMM | ebpf::VOP
            )
        })
        .map(Some)
        .collect();
    opcodes.push(None); // lddw
    for opc in opcodes {
        for (k, &(dst, src, off, imm)) in configs.iter().enumerate() {
            let jump = opc.is_some_and(|o| {
                matches!(o & ebpf::BPF_CLS_MASK, ebpf::BPF_JMP64 | ebpf::BPF_JMP32)
            });
            let off = if jump { 0 } else { off };
            let scalar = |a: &mut Asm| match opc {
                Some(opc) => a.raw(opc, dst, src, off, imm),
                None => a.lddw(dst, 0x8000_0000_0000_0001u64.wrapping_mul(imm as u64)),
            };
            // Valid in this context at all? (verifier)
            {
                let mut a = Asm::new(VecMode::Call);
                scalar(&mut a);
                a.mov64_imm(0, 0);
                a.exit();
                if Runner::new(
                    &a.finish(),
                    &input_for(&init, 0),
                    HEAP_LEN,
                    config(SimdJitStrategy::Direct),
                )
                .is_err()
                {
                    continue;
                }
            }
            valid += 1;
            for (j, first) in ops.iter().enumerate() {
                for placement in ["after operand setup", "between vector ops"] {
                    let mut a = Asm::new(VecMode::Call);
                    prologue(&mut a, heap, 0x0f0f_00ff_5555_aaaa);
                    a.lddw(5, heap);
                    a.lddw(9, heap + 128);
                    a.lddw(4, 0x0102_0304_0506_0708);
                    a.mov64_imm(0, 5);
                    // Dirty cached operands
                    for s in [8u8, 12, 16, 20] {
                        a.vop_full(VADD32, false, s, s, s, 0, 0, 0);
                    }
                    let second = &ops[(j + 1 + k) % ops.len()];
                    if placement == "after operand setup" {
                        a.mov64_imm(1, pack(first) as i64);
                        scalar(&mut a);
                        a.raw(
                            ebpf::CALL_IMM,
                            0,
                            0,
                            0,
                            hash(first.subop, first.wide) as i32 as i64,
                        );
                    } else {
                        a.mov64_imm(1, pack(first) as i64);
                        a.raw(
                            ebpf::CALL_IMM,
                            0,
                            0,
                            0,
                            hash(first.subop, first.wide) as i32 as i64,
                        );
                        scalar(&mut a);
                        // r1 still holds the first operation's operands unless the scalar
                        // instruction wrote it: call the second one with whatever r1 holds now
                        // half the time
                        if k % 2 == 0 {
                            a.mov64_imm(1, pack(second) as i64);
                        }
                        a.raw(
                            ebpf::CALL_IMM,
                            0,
                            0,
                            0,
                            hash(second.subop, second.wide) as i32 as i64,
                        );
                    }
                    a.vop_full(after.subop, false, after.vd, after.va, after.vb, 0, 0, 0);
                    // Every register to the heap (r9 through the stack)
                    a.stxdw(10, -8, 9);
                    a.lddw(9, ebpf::MM_HEAP_START + 1024);
                    for reg in 0..9u8 {
                        a.stxdw(9, 8 * reg as i16, reg);
                    }
                    a.ldxdw(0, 10, -8);
                    a.stxdw(9, 72, 0);
                    a.mov64_imm(0, 0);
                    a.exit();
                    let outcome = differential(&a.finish(), &input_for(&init, 0), Placement::Front);
                    completed += outcome
                        .as_ref()
                        .is_ok_and(|out| out.result.starts_with("Ok"))
                        as usize;
                    if let Err(err) = outcome {
                        panic!(
                            "{}",
                            format!(
                                "{} r{dst} r{src} off {off} imm {imm:#x}, {placement}, {}: {err}",
                                opc.map_or("lddw".to_string(), |o| format!("opcode {o:#04x}")),
                                mnemonic(first.subop, first.wide)
                            )
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    eprintln!("{valid} valid instructions, {cases} cases, {completed} ran to the end");
    assert!(
        valid > 400 && cases > 5_000 && completed * 2 > cases,
        "{}",
        format!("{valid} valid instructions, {cases} cases, {completed} ran to the end")
    );
}

/// Cross-program invocations (a syscall which runs another program in its own VM, like Agave's CPI):
/// the callee starts with a zeroed vector file every time (also on its second invocation, after
/// leaving dirty slots behind), the caller's vector file (including slots dirty in the JIT's
/// register cache at the call) is exactly as without the invocation, for every combination of
/// engines and JIT strategies of caller and callee, with an attached or a VM owned vector file.
#[test]
fn cross_program_invocations_keep_vector_files_apart() {
    use simd::{
        harness::{Callee, Engine},
        *,
    };
    let init = vector_init(18);
    let invoke = ebpf::hash_symbol_name(b"sol_test_invoke_") as i32 as i64;
    // Caller: random slots, dirty cached results, two invocations, then every slot to the heap
    let caller = {
        let mut a = Asm::new(VecMode::Call);
        prologue(&mut a, 0x1234_5678, u64::MAX);
        for s in [8u8, 12, 16, 20, 254] {
            a.vop_full(VADD32, false, s, s, 1, 0, 0, 0);
        }
        a.vop_full(VXOR, true, 40, 8, 12, 0, 0, 0);
        a.raw(ebpf::CALL_IMM, 0, 0, 0, invoke);
        a.mov64(6, 0); // the callee's r0
        a.vop_full(VADD32, false, 9, 8, 255, 0, 0, 0);
        a.raw(ebpf::CALL_IMM, 0, 0, 0, invoke);
        a.vop_full(VADD32, false, 10, 9, 20, 0, 0, 0);
        a.lddw(7, ebpf::MM_HEAP_START);
        for k in 0..8u8 {
            a.vstore_n(7, 1024 * k as i16, 32 * k, 32);
        }
        a.mov64(0, 6);
        a.exit();
        a.finish()
    };
    // Callee: its slots to the heap (must be zero), then dirty slots of its own, r0 = 42
    let callee = {
        let mut a = Asm::new(VecMode::Call);
        a.lddw(7, ebpf::MM_HEAP_START);
        for k in 0..8u8 {
            a.vstore_n(7, 1024 * k as i16, 32 * k, 32);
        }
        a.mov64_imm(2, -1);
        for s in [0u8, 8, 9, 12, 254, 255] {
            a.vop_full(VBCAST32, false, s, 0, 0, 0, SCALAR_REG, 0);
        }
        a.vop_full(VADD32, true, 40, 40, 40, 0, 0, 0);
        a.mov64_imm(0, 42);
        a.exit();
        a.finish()
    };
    let input = input_for(&init, 0);
    let engines: Vec<(Engine, SimdJitStrategy)> = {
        #[allow(unused_mut)] // only extended on x86-64
        let mut engines = vec![(Engine::Interpreter, SimdJitStrategy::Direct)];
        #[cfg(target_arch = "x86_64")]
        engines.extend([
            (Engine::Jit, SimdJitStrategy::RegisterCached),
            (Engine::Jit, SimdJitStrategy::Direct),
        ]);
        engines
    };
    let runner = |program: &[u8], engine: Engine, strategy| {
        let runner = Runner::new(program, &input, HEAP_LEN, config(strategy)).unwrap();
        if engine == Engine::Jit {
            runner.jit_compile().unwrap();
        }
        runner
    };
    // The caller without invocations (the syscall returns 0, nothing else happens)
    let mut alone = runner(&caller, Engine::Interpreter, SimdJitStrategy::Direct);
    let (alone_cus, result) = alone.run(Engine::Interpreter);
    assert_eq!(format!("{result:?}"), "Ok(0)");
    let expected_slots = alone.heap()[..ebpf::VECTOR_MEMORY_SIZE].to_vec();
    let mut reference = None;
    for &(caller_engine, caller_strategy) in &engines {
        for &(callee_engine, callee_strategy) in &engines {
            for detached in [false, true] {
                let label = format!("caller {caller_engine:?} {caller_strategy:?}, callee {callee_engine:?} {callee_strategy:?}, detached {detached}");
                let mut outer = runner(&caller, caller_engine, caller_strategy);
                outer.set_callee(Some(Callee {
                    runner: Box::new(runner(&callee, callee_engine, callee_strategy)),
                    engine: callee_engine,
                    detached,
                    invocations: 0,
                }));
                let (cus, result) = if detached {
                    outer.run_detached(caller_engine)
                } else {
                    outer.run(caller_engine)
                };
                assert_eq!(format!("{result:?}"), "Ok(42)", "{}", label);
                let inner = outer.take_callee().unwrap();
                assert_eq!(inner.invocations, 2, "{}", label);
                // The callee saw a zeroed vector file (on its second invocation too)
                assert!(
                    inner.runner.heap()[..ebpf::VECTOR_MEMORY_SIZE]
                        .iter()
                        .all(|b| *b == 0),
                    "{}",
                    label
                );
                // The caller's slots are as without the invocations
                assert!(
                    outer.heap()[..ebpf::VECTOR_MEMORY_SIZE] == expected_slots[..],
                    "{}",
                    label
                );
                assert!(cus > alone_cus, "{}", label);
                // Same CUs and memory in every combination
                let outcome = (cus, outer.heap().to_vec(), inner.runner.heap().to_vec());
                match &reference {
                    None => reference = Some(outcome),
                    Some(r) => assert!(*r == outcome, "{}", label),
                }
            }
        }
    }
}
