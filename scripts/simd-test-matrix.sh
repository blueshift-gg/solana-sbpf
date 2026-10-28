#!/usr/bin/env bash
# Runs every test configuration of the SIMD experiment on this host and prints one line per run.
#
# Every test compares the interpreter with each JIT tier of the host. On an AVX-512 host those tiers
# include AVX-512 (all detected extensions), AVX2 only, and every CPU feature profile of
# simd::features::PROFILES (Skylake-SP, Cascade Lake, Cannon Lake, Ice Lake / Zen 4), so the first
# run already checks AVX-512 against AVX2 against the interpreter.
#
#   scripts/simd-test-matrix.sh            # everything (the debug runs take a few minutes)
#   QUICK=1 scripts/simd-test-matrix.sh    # release runs only
set -u
cd "$(dirname "$0")/.."
F=simd-experiment,jit
failed=0
run() {
    local label=$1
    shift
    local out
    out=$("$@" 2>&1)
    local status=$?
    local passed failures
    passed=$(grep -E "^test result" <<<"$out" | awk '{p += $4} END {print p + 0}')
    failures=$(grep -E "^test result" <<<"$out" | awk '{f += $6} END {print f + 0}')
    if [ $status -ne 0 ]; then
        failed=1
        echo "FAIL  $label: $passed passed, $failures failed"
        grep -E "panicked at|^thread|signal:|^error" -A2 <<<"$out" | head -40 | sed 's/^/      /'
    else
        echo "ok    $label: $passed passed"
    fi
}

echo "host: $(uname -sm)"
if [ -r /proc/cpuinfo ]; then
    echo "cpu:  $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
    echo "flags:$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|avx512.*|gfni|vpclmulqdq)$' | tr '\n' ' ')"
fi

run "release, all tests (AVX-512 host: AVX-512 vs AVX2 vs every profile)" \
    cargo test --release --features $F
run "release, AVX2 as the default tier (SBPF_SIMD_AVX512=0)" \
    env SBPF_SIMD_AVX512=0 cargo test --release --features $F --test simd --test simd_robustness
run "release, tracer feature" \
    cargo test --release --features $F,tracer --test simd --test simd_robustness
run "release, jit-enable-host-stack-frames feature" \
    cargo test --release --features $F,jit-enable-host-stack-frames --test simd --test simd_robustness
run "release, SIMD off (upstream tests)" \
    cargo test --release
if [ -z "${QUICK:-}" ]; then
    # Debug builds: the JIT checks the stack alignment before every call into Rust, overflow checks
    run "debug, SIMD tests (stack alignment checks)" \
        cargo test --features $F --test simd --test simd_robustness
    run "debug, jit-enable-host-stack-frames" \
        cargo test --features $F,jit-enable-host-stack-frames --test simd --test simd_robustness
fi
exit $failed
