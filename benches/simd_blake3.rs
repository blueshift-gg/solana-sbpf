//! SIMD experiment benchmarks: BLAKE3 hashing, compression and the register caching microbench.
//!
//! `cargo bench --features simd-experiment --bench simd_blake3`
//!
//! Environment variables:
//! - `SIMD_BENCH_OUT`: output JSON path (default `target/simd-bench/results.json`)
//! - `SIMD_BENCH_QUICK=1`: fewer samples
//! - `SIMD_BENCH_FILTER`: only run benchmarks whose id contains this string
//! - `SIMD_BENCH_SKIP`: comma separated substrings of benchmark ids to skip
//!
//! Timing: every sample runs the program `n` times (n chosen so that a sample takes >= 2 ms), the
//! reported value is the median over samples of the per-run time. Warm execution only (the program
//! is JIT compiled once up front); cold compilation is measured separately.

use solana_sbpf::{
    simd::{
        asm::VecMode,
        blake3_program::{self, Kernel, HEAP_OUT, HEAP_SIZE},
        harness::{config, running_under_rosetta, Engine, Runner},
    },
    vm::SimdJitStrategy,
};
use std::{fmt::Write as _, time::Instant};

#[derive(Clone, Copy)]
struct Variant {
    id: &'static str,
    label: &'static str,
    kernel: Kernel,
    mode: VecMode,
    engine: Engine,
    strategy: SimdJitStrategy,
}

const fn v(
    id: &'static str,
    label: &'static str,
    kernel: Kernel,
    mode: VecMode,
    engine: Engine,
    strategy: SimdJitStrategy,
) -> Variant {
    Variant { id, label, kernel, mode, engine, strategy }
}

use Engine::*;
use Kernel::*;
use SimdJitStrategy::*;
const MEM: VecMode = VecMode::Memory(9);
const SLOT: VecMode = VecMode::Slot;
const CALL: VecMode = VecMode::Call;

const VARIANTS: &[Variant] = &[
    v("A_scalar_jit", "A: scalar sBPF, JIT", Scalar, SLOT, Jit, Direct),
    v("A0_scalar_interp", "scalar sBPF, interpreter", Scalar, SLOT, Interpreter, Direct),
    v("B_simd8_interp", "B: SIMD 8-way, interpreter", Simd8, SLOT, Interpreter, Direct),
    v("B1_simd1_interp", "SIMD 1-state, interpreter", Simd1, SLOT, Interpreter, Direct),
    v("C_simd8_mem_jit", "C: SIMD 8-way, ordinary memory, JIT", Simd8, MEM, Jit, Direct),
    v("D_simd8_slot_direct", "D: SIMD 8-way, vector region, JIT uncached", Simd8, SLOT, Jit, Direct),
    v("E_simd8_slot_cached", "E: SIMD 8-way, vector region, JIT reg-cached", Simd8, SLOT, Jit, RegisterCached),
    v("C1_simd1_mem_jit", "SIMD 1-state, ordinary memory, JIT", Simd1, MEM, Jit, Direct),
    v("D1_simd1_slot_direct", "SIMD 1-state, vector region, JIT uncached", Simd1, SLOT, Jit, Direct),
    v("E1_simd1_slot_cached", "SIMD 1-state, vector region, JIT reg-cached", Simd1, SLOT, Jit, RegisterCached),
    v("G_simd8_call_cached", "G: SIMD 8-way, intrinsic calls, JIT reg-cached", Simd8, CALL, Jit, RegisterCached),
    v("G1_simd1_call_cached", "SIMD 1-state, intrinsic calls, JIT reg-cached", Simd1, CALL, Jit, RegisterCached),
    v("H_simd8_call_direct", "H: SIMD 8-way, intrinsic calls, JIT uncached", Simd8, CALL, Jit, Direct),
    v("H1_simd1_call_direct", "SIMD 1-state, intrinsic calls, JIT uncached", Simd1, CALL, Jit, Direct),
    v("I_simd8_call_interp", "SIMD 8-way, intrinsic calls, interpreter", Simd8, CALL, Interpreter, Direct),
];

const SIZES: &[usize] = &[64, 1024, 2048, 4096, 8192, 16384, 65536, 1 << 20];

fn quick() -> bool {
    std::env::var("SIMD_BENCH_QUICK").is_ok_and(|v| v == "1")
}

fn selected(id: &str) -> bool {
    let skipped = std::env::var("SIMD_BENCH_SKIP")
        .map_or(false, |list| list.split(',').any(|s| !s.is_empty() && id.contains(s)));
    !skipped && std::env::var("SIMD_BENCH_FILTER").map_or(true, |f| id.contains(&f))
}

fn tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

struct Measurement {
    median_ns: f64,
    min_ns: f64,
    mad_ns: f64,
    median_ticks: f64,
    samples: usize,
    runs_per_sample: usize,
}

/// Times `f`, returns robust statistics of the per-call time
fn measure(mut f: impl FnMut()) -> Measurement {
    let (target_samples, min_sample_ns, budget_ns) = if quick() {
        (5usize, 1e6, 0.4e9)
    } else {
        (15usize, 2e6, 2.0e9)
    };
    // Warm up and calibrate
    let start = Instant::now();
    f();
    let single = start.elapsed().as_nanos().max(1) as f64;
    let runs = ((min_sample_ns / single).ceil() as usize).max(1);
    let samples = ((budget_ns / (single * runs as f64)) as usize).clamp(3, target_samples);
    let mut times = Vec::with_capacity(samples);
    let mut ticks = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = tsc();
        let start = Instant::now();
        for _ in 0..runs {
            f();
        }
        let elapsed = start.elapsed().as_nanos() as f64 / runs as f64;
        ticks.push((tsc() - t0) as f64 / runs as f64);
        times.push(elapsed);
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let min_ns = times.iter().cloned().fold(f64::MAX, f64::min);
    let median_ns = median(&mut times);
    let mut deviations: Vec<f64> = times.iter().map(|t| (t - median_ns).abs()).collect();
    Measurement {
        median_ns,
        min_ns,
        mad_ns: median(&mut deviations),
        median_ticks: median(&mut ticks),
        samples,
        runs_per_sample: runs,
    }
}

fn data(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn stats_json(runner: &Runner) -> String {
    #[cfg(target_arch = "x86_64")]
    if let Some(compiled) = runner.executable.get_compiled_program() {
        let s = compiled.simd_stats;
        return format!(
            r#"{{"code_bytes":{},"native_instructions":{},"vector_instructions":{},"vector_loads":{},"vector_stores":{},"memory_operands":{},"spills":{},"reloads":{},"cache_hits":{},"evictions":{},"flush_stores":{},"translations":{},"intrinsic_slow_paths":{}}}"#,
            compiled.machine_code_length(),
            s.native_instructions,
            s.vector_instructions,
            s.vector_loads,
            s.vector_stores,
            s.memory_operands,
            s.spills,
            s.reloads,
            s.cache_hits,
            s.evictions,
            s.flush_stores,
            s.translations,
            s.intrinsic_slow_paths
        );
    }
    let _ = runner;
    "null".to_string()
}

fn jit_available() -> bool {
    cfg!(target_arch = "x86_64")
}

fn main() {
    let mut out = String::new();
    let host = format!(
        "{} {}{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        if running_under_rosetta() { " (Rosetta 2 translated)" } else { "" }
    );
    let cpu = cpu_name();
    eprintln!("host: {host}, cpu: {cpu}");
    let _ = write!(
        out,
        r#"{{"host":"{}","cpu":"{}","rosetta":{},"quick":{},"hash":["#,
        json_escape(&host),
        json_escape(&cpu),
        running_under_rosetta(),
        quick()
    );

    // Full hashing
    let mut first = true;
    for &size in SIZES {
        let input = data(size);
        let expected = *blake3::hash(&input).as_bytes();
        // F: native
        {
            let id = format!("hash/{size}/F_native");
            if selected(&id) {
                let m = measure(|| {
                    std::hint::black_box(blake3::hash(std::hint::black_box(&input)));
                });
                emit(&mut out, &mut first, &id, "F_native", "F: native blake3 crate (host)", size, &m, None, None, "null", None);
            }
        }
        for variant in VARIANTS {
            let id = format!("hash/{size}/{}", variant.id);
            if !selected(&id) {
                continue;
            }
            if variant.engine == Jit && !jit_available() {
                emit_unsupported(&mut out, &mut first, &id, variant, size);
                continue;
            }
            if quick() && size == 1 << 20 && variant.engine == Interpreter && variant.kernel != Simd8 {
                continue;
            }
            let program = blake3_program::hash_program(variant.kernel, variant.mode);
            let mut runner = Runner::new(&program, &blake3_program::build_input(&input), HEAP_SIZE, config(variant.strategy)).unwrap();
            let mut compile_ns = None;
            if variant.engine == Jit {
                let mut times: Vec<f64> = (0..if quick() { 3 } else { 7 })
                    .map(|_| runner.jit_compile().unwrap().as_nanos() as f64)
                    .collect();
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                compile_ns = Some(times[times.len() / 2]);
            }
            let (cus, result) = runner.run(variant.engine);
            assert!(result.is_ok(), "{} {:?}", id, result);
            assert_eq!(runner.heap()[HEAP_OUT..HEAP_OUT + 32], expected, "{}", id);
            let m = measure(|| {
                let (_, result) = runner.run(variant.engine);
                debug_assert!(result.is_ok());
            });
            assert_eq!(runner.heap()[HEAP_OUT..HEAP_OUT + 32], expected, "{}", id);
            let stats = stats_json(&runner);
            emit(&mut out, &mut first, &id, variant.id, variant.label, size, &m, Some(cus), compile_ns, &stats, Some(variant));
        }
    }
    out.push_str(r#"],"compress":["#);

    // Compression function alone
    let mut first = true;
    let iterations = 256u64;
    for (kernel, mode, engine, strategy, id, label, compressions) in [
        (Scalar, SLOT, Jit, Direct, "A_scalar_jit", "scalar, JIT", 1u64),
        (Scalar, SLOT, Interpreter, Direct, "A0_scalar_interp", "scalar, interpreter", 1),
        (Simd1, SLOT, Interpreter, Direct, "B1_simd1_interp", "SIMD 1-state, interpreter", 1),
        (Simd8, SLOT, Interpreter, Direct, "B_simd8_interp", "SIMD 8-way, interpreter", 8),
        (Simd1, MEM, Jit, Direct, "C1_simd1_mem_jit", "SIMD 1-state, ordinary memory, JIT", 1),
        (Simd8, MEM, Jit, Direct, "C_simd8_mem_jit", "SIMD 8-way, ordinary memory, JIT", 8),
        (Simd1, SLOT, Jit, Direct, "D1_simd1_slot_direct", "SIMD 1-state, region, JIT uncached", 1),
        (Simd8, SLOT, Jit, Direct, "D_simd8_slot_direct", "SIMD 8-way, region, JIT uncached", 8),
        (Simd1, SLOT, Jit, RegisterCached, "E1_simd1_slot_cached", "SIMD 1-state, region, JIT cached", 1),
        (Simd8, SLOT, Jit, RegisterCached, "E_simd8_slot_cached", "SIMD 8-way, region, JIT cached", 8),
        (Simd1, CALL, Jit, RegisterCached, "G1_simd1_call_cached", "SIMD 1-state, intrinsic calls, JIT cached", 1),
        (Simd8, CALL, Jit, RegisterCached, "G_simd8_call_cached", "SIMD 8-way, intrinsic calls, JIT cached", 8),
        (Simd1, CALL, Jit, Direct, "H1_simd1_call_direct", "SIMD 1-state, intrinsic calls, JIT uncached", 1),
        (Simd8, CALL, Jit, Direct, "H_simd8_call_direct", "SIMD 8-way, intrinsic calls, JIT uncached", 8),
    ] {
        let full_id = format!("compress/{id}");
        if !selected(&full_id) {
            continue;
        }
        if engine == Jit && !jit_available() {
            continue;
        }
        let program = blake3_program::compress_program(kernel, mode);
        let mut input = blake3_program::build_input(&data(512));
        input[8..16].copy_from_slice(&iterations.to_le_bytes());
        let mut runner = Runner::new(&program, &input, HEAP_SIZE, config(strategy)).unwrap();
        if engine == Jit {
            runner.jit_compile().unwrap();
        }
        let (cus, result) = runner.run(engine);
        assert!(result.is_ok(), "{} {:?}", full_id, result);
        let m = measure(|| {
            runner.run(engine);
        });
        let per = (iterations * compressions) as f64;
        if !first {
            out.push(',');
        }
        first = false;
        let _ = write!(
            out,
            r#"{{"id":"{}","label":"{}","ns_per_compression":{:.2},"cus_per_compression":{:.1},"mad_ns":{:.2},"stats":{}}}"#,
            id, label, m.median_ns / per, cus as f64 / per, m.mad_ns / per, stats_json(&runner)
        );
        eprintln!("{full_id:40} {:>10.1} ns/compression  {:>8.1} CU/compression", m.median_ns / per, cus as f64 / per);
    }
    // Native reference for one compression
    {
        let block = data(64);
        let m = measure(|| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(std::hint::black_box(&block));
            std::hint::black_box(hasher.finalize());
        });
        if !first {
            out.push(',');
        }
        let _ = write!(out, r#"{{"id":"F_native","label":"native blake3, 64 byte hash","ns_per_compression":{:.2},"cus_per_compression":null,"mad_ns":{:.2},"stats":null}}"#, m.median_ns, m.mad_ns);
    }
    out.push_str(r#"],"microbench":["#);

    // Register caching microbenchmark
    let mut first = true;
    let iterations = 1000u64;
    for repeat in [1usize, 4, 16, 64] {
        for (mode, engine, strategy, id) in [
            (SLOT, Interpreter, Direct, "interp"),
            (MEM, Jit, Direct, "C_mem"),
            (SLOT, Jit, Direct, "D_direct"),
            (SLOT, Jit, RegisterCached, "E_cached"),
            (CALL, Jit, Direct, "H_call_direct"),
            (CALL, Jit, RegisterCached, "G_call_cached"),
        ] {
            let full_id = format!("micro/{repeat}/{id}");
            if !selected(&full_id) || (engine == Jit && !jit_available()) {
                continue;
            }
            let program = blake3_program::microbench_program(repeat, mode);
            let mut input = blake3_program::build_input(&data(0));
            for (i, byte) in input[0x20..0x20 + 160].iter_mut().enumerate() {
                *byte = (i * 37 + 11) as u8;
            }
            input[8..16].copy_from_slice(&iterations.to_le_bytes());
            let reference = blake3_program::microbench_reference(&input, repeat, iterations);
            let mut runner = Runner::new(&program, &input, HEAP_SIZE, config(strategy)).unwrap();
            if engine == Jit {
                runner.jit_compile().unwrap();
            }
            let (cus, result) = runner.run(engine);
            assert!(result.is_ok(), "{} {:?}", full_id, result);
            assert_eq!(solana_sbpf::simd::read_lanes(runner.heap()[HEAP_OUT..].as_ptr()), reference, "{}", full_id);
            let m = measure(|| {
                runner.run(engine);
            });
            let ops = (iterations as usize * repeat * 4) as f64;
            if !first {
                out.push(',');
            }
            first = false;
            let _ = write!(
                out,
                r#"{{"id":"{}","repeat":{},"ns_per_op":{:.3},"ns_per_iteration":{:.2},"cus":{},"stats":{}}}"#,
                id, repeat, m.median_ns / ops, m.median_ns / iterations as f64, cus, stats_json(&runner)
            );
            eprintln!("{full_id:30} {:>8.3} ns/vector-op", m.median_ns / ops);
        }
    }
    out.push_str("]}");
    let path = std::env::var("SIMD_BENCH_OUT").unwrap_or_else(|_| "target/simd-bench/results.json".to_string());
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&path, out).unwrap();
    eprintln!("wrote {path}");
}

#[allow(clippy::too_many_arguments)]
fn emit(
    out: &mut String,
    first: &mut bool,
    id: &str,
    variant_id: &str,
    label: &str,
    size: usize,
    m: &Measurement,
    cus: Option<u64>,
    compile_ns: Option<f64>,
    stats: &str,
    variant: Option<&Variant>,
) {
    if !*first {
        out.push(',');
    }
    *first = false;
    let mib_s = size as f64 / (1 << 20) as f64 / (m.median_ns * 1e-9);
    let engine = variant.map_or("native", |v| if v.engine == Jit { "jit" } else { "interpreter" });
    let _ = write!(
        out,
        r#"{{"id":"{}","variant":"{}","label":"{}","engine":"{}","size":{},"median_ns":{:.1},"min_ns":{:.1},"mad_ns":{:.1},"ticks":{:.1},"mib_s":{:.3},"cus":{},"compile_ns":{},"samples":{},"runs_per_sample":{},"supported":true,"stats":{}}}"#,
        id,
        variant_id,
        json_escape(label),
        engine,
        size,
        m.median_ns,
        m.min_ns,
        m.mad_ns,
        m.median_ticks,
        mib_s,
        cus.map_or("null".to_string(), |c| c.to_string()),
        compile_ns.map_or("null".to_string(), |c| format!("{c:.0}")),
        m.samples,
        m.runs_per_sample,
        stats
    );
    eprintln!("{id:40} {:>14.1} ns  {:>10.2} MiB/s  CUs {:?}", m.median_ns, mib_s, cus);
}

fn emit_unsupported(out: &mut String, first: &mut bool, id: &str, variant: &Variant, size: usize) {
    if !*first {
        out.push(',');
    }
    *first = false;
    let _ = write!(
        out,
        r#"{{"id":"{}","variant":"{}","label":"{}","engine":"jit","size":{},"supported":false}}"#,
        id,
        variant.id,
        json_escape(variant.label),
        size
    );
    eprintln!("{id:40} UNSUPPORTED (no x86_64 JIT on this host)");
}

fn cpu_name() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|c| {
                c.lines()
                    .find(|l| l.starts_with("model name"))
                    .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown".to_string())
}
