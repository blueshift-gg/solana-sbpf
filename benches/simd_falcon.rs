//! Falcon-512 verification: blueshift-gg/solana-falcon512 (scalar sBPF) against the same verifier
//! with Firedancer's vectorized NTT / norm written with the vector intrinsics (simd-falcon/).
//!
//! Both programs are prebuilt SBPFv3 ELFs in simd-falcon/fixtures (`cargo-build-sbf --arch v3`),
//! embed the same compile time prepared pubkey and take `[signature (666)][message]` as instruction
//! data. Every signature in fixtures/sigs.bin is verified by both programs on every engine, a
//! tampered copy must be rejected, then CUs and wall clock time are recorded.
//!
//! `cargo bench --features simd-experiment --bench simd_falcon`
//! Env: `FALCON_BENCH_OUT` (JSON path, default target/simd-bench/falcon.json),
//! `FALCON_BENCH_SIGS` (number of signatures timed, default all), `SIMD_BENCH_QUICK=1`.

use solana_sbpf::{
    error::ProgramResult,
    simd::harness::{config, running_under_rosetta, Engine, Runner},
    vm::SimdJitStrategy,
};
use std::{fmt::Write, path::Path, time::Instant};

const SIG_LEN: usize = 666;
const HEAP: usize = 32 * 1024;

struct Program {
    id: &'static str,
    label: &'static str,
    elf: Vec<u8>,
}

fn input(sig: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&((SIG_LEN + msg.len()) as u64).to_le_bytes());
    data.extend_from_slice(sig);
    data.extend_from_slice(msg);
    data.extend_from_slice(&[7u8; 32]); // program id
    data
}

fn load_sigs(path: &Path) -> Vec<(Vec<u8>, Vec<u8>)> {
    let bytes = std::fs::read(path).expect("fixtures/sigs.bin");
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let sig = bytes[i..i + SIG_LEN].to_vec();
        let len = u16::from_le_bytes([bytes[i + SIG_LEN], bytes[i + SIG_LEN + 1]]) as usize;
        let msg = bytes[i + SIG_LEN + 2..i + SIG_LEN + 2 + len].to_vec();
        out.push((sig, msg));
        i += SIG_LEN + 2 + len;
    }
    out
}

fn runner(p: &Program, data: &[u8], engine: Engine) -> Runner {
    let mut r = Runner::from_elf(&p.elf, data, HEAP, config(SimdJitStrategy::RegisterCached))
        .expect("load");
    r.budget = 1_400_000;
    if engine == Engine::Jit {
        r.jit_compile().expect("jit");
    }
    r
}

fn exit_code(result: &ProgramResult) -> Option<u64> {
    match result {
        ProgramResult::Ok(v) => Some(*v),
        ProgramResult::Err(_) => None,
    }
}

/// Median ns per run, with enough repetitions per sample to last ~2 ms
fn time(r: &mut Runner, engine: Engine, quick: bool) -> f64 {
    let start = Instant::now();
    r.run(engine);
    let single = start.elapsed().as_nanos().max(1) as f64;
    let runs = ((2e6 / single).ceil() as usize).max(1);
    let samples = if quick { 3 } else { 9 };
    let mut times: Vec<f64> = (0..samples)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..runs {
                r.run(engine);
            }
            t.elapsed().as_nanos() as f64 / runs as f64
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn stats(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("simd-falcon/fixtures");
    let programs = [
        Program {
            id: "blueshift",
            label: "blueshift-gg/solana-falcon512 (scalar sBPF)",
            elf: std::fs::read(dir.join("falcon_blueshift.so")).unwrap(),
        },
        Program {
            id: "simd",
            label: "Firedancer Falcon-512 with vector intrinsics (16 bit NTT, vector SHAKE)",
            elf: std::fs::read(dir.join("falcon_simd.so")).unwrap(),
        },
    ];
    let sigs = load_sigs(&dir.join("sigs.bin"));
    let quick = std::env::var("SIMD_BENCH_QUICK").is_ok_and(|v| v == "1");
    let timed = std::env::var("FALCON_BENCH_SIGS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(sigs.len())
        .min(sigs.len());
    let mut engines = vec![Engine::Interpreter];
    if cfg!(target_arch = "x86_64") {
        engines.push(Engine::Jit);
    }
    eprintln!(
        "host: {} {}{}, {} signatures",
        std::env::consts::ARCH,
        std::env::consts::OS,
        if running_under_rosetta() {
            " (Rosetta 2)"
        } else {
            ""
        },
        sigs.len()
    );

    // Correctness: every signature accepted, tampered message and signature rejected, CUs equal
    // across engines
    let mut cus: Vec<Vec<u64>> = vec![Vec::new(); programs.len()];
    for (sig, msg) in &sigs {
        let mut bad_msg = msg.clone();
        if bad_msg.is_empty() {
            bad_msg.push(1)
        } else {
            bad_msg[0] ^= 1
        }
        let mut bad_sig = sig.clone();
        bad_sig[100] ^= 0x10;
        for (pi, p) in programs.iter().enumerate() {
            let mut reference = None;
            for &engine in &engines {
                let (cu, result) = runner(p, &input(sig, msg), engine).run(engine);
                assert_eq!(
                    exit_code(&result),
                    Some(0),
                    "{} rejected a valid signature on {:?}: {:?}",
                    p.id,
                    engine,
                    result
                );
                match reference {
                    None => reference = Some(cu),
                    Some(r) => assert_eq!(r, cu, "{} CUs differ between engines", p.id),
                }
                for bad in [input(sig, &bad_msg), input(&bad_sig, msg)] {
                    let (_, result) = runner(p, &bad, engine).run(engine);
                    assert_ne!(
                        exit_code(&result),
                        Some(0),
                        "{} accepted a tampered input on {:?}",
                        p.id,
                        engine
                    );
                }
            }
            cus[pi].push(reference.unwrap());
        }
    }
    eprintln!(
        "correctness: {} signatures accepted, tampered inputs rejected, by both programs on {:?}",
        sigs.len(),
        engines
    );

    let mut out = String::new();
    let _ = write!(
        out,
        r#"{{"host":"{} {}","rosetta":{},"signatures":{},"programs":["#,
        std::env::consts::ARCH,
        std::env::consts::OS,
        running_under_rosetta(),
        sigs.len()
    );
    for (pi, p) in programs.iter().enumerate() {
        let mut c: Vec<f64> = cus[pi].iter().map(|x| *x as f64).collect();
        let (cmin, cmed, cmax) = stats(&mut c);
        let _ = write!(
            out,
            r#"{}{{"id":"{}","label":"{}","elf_bytes":{},"cus":{{"min":{},"median":{},"max":{}}},"cu_list":{:?},"engines":{{"#,
            if pi > 0 { "," } else { "" },
            p.id,
            p.label,
            p.elf.len(),
            cmin,
            cmed,
            cmax,
            cus[pi]
        );
        for (ei, &engine) in engines.iter().enumerate() {
            let mut ns: Vec<f64> = sigs[..timed]
                .iter()
                .map(|(sig, msg)| time(&mut runner(p, &input(sig, msg), engine), engine, quick))
                .collect();
            let list = ns.clone();
            let (tmin, tmed, tmax) = stats(&mut ns);
            let compile = if engine == Engine::Jit {
                let r = Runner::from_elf(
                    &p.elf,
                    &input(&sigs[0].0, &sigs[0].1),
                    HEAP,
                    config(SimdJitStrategy::RegisterCached),
                )
                .unwrap();
                let mut v: Vec<f64> = (0..5)
                    .map(|_| {
                        let r = Runner::from_elf(
                            &p.elf,
                            &input(&sigs[0].0, &sigs[0].1),
                            HEAP,
                            config(SimdJitStrategy::RegisterCached),
                        )
                        .unwrap();
                        r.jit_compile().unwrap().as_nanos() as f64
                    })
                    .collect();
                drop(r);
                stats(&mut v).1
            } else {
                0.0
            };
            let name = if engine == Engine::Jit {
                "jit"
            } else {
                "interpreter"
            };
            eprintln!(
                "{:10} {:12} CUs median {:>8}  time median {:>10.1} µs (min {:.1}, max {:.1})",
                p.id,
                name,
                cmed,
                tmed / 1e3,
                tmin / 1e3,
                tmax / 1e3
            );
            let _ = write!(
                out,
                r#"{}"{}":{{"ns":{{"min":{:.1},"median":{:.1},"max":{:.1}}},"ns_list":{:?},"compile_ns":{:.0}}}"#,
                if ei > 0 { "," } else { "" },
                name,
                tmin,
                tmed,
                tmax,
                list.iter().map(|x| x.round()).collect::<Vec<_>>(),
                compile
            );
        }
        out.push_str("}}");
    }
    out.push_str("]}");
    let path = std::env::var("FALCON_BENCH_OUT")
        .unwrap_or_else(|_| "target/simd-bench/falcon.json".to_string());
    if let Some(parent) = Path::new(&path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&path, out).unwrap();
    eprintln!("wrote {path}");
}
