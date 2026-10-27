# SIMD experiment: benchmark results

> **Read the environment section first.** None of these numbers were measured on native x86-64
> hardware. The AVX2 JIT numbers come from macOS Rosetta 2 (primary) and QEMU's `-cpu max`
> emulation. The ratios between variants are informative. The absolute nanoseconds aren't. Re-run on real hardware before drawing
> production conclusions (commands at the end).

## Environments

| id | where | what can run | notes |
|---|---|---|---|
| **mac** | macOS Rosetta 2 on an Apple M3 Max, `x86_64-apple-darwin` build | everything, including AVX2 | Primary environment. Rosetta 2 runs AVX2 but hides it from CPUID, so the harness sets `simd_assume_avx2` when `sysctl.proc_translated` is 1. 256-bit ops become pairs of 128-bit NEON ops. Near-native speed for scalar code. |
| **qemu** | `qemu-x86_64 7.2 -cpu max` (TCG) in an arm64 Linux container on an Apple M3 Max | everything, including AVX2 | CPUID reports AVX2, so no override was needed. Roughly 10× slower than Rosetta for scalar code. AVX2 is emulated by helper functions, which probably *penalizes* the SIMD variants compared with real hardware. |
| **rosetta** | Docker `linux/amd64`, Rosetta for Linux ("VirtualApple @ 2.50GHz") | interpreter, scalar JIT, native | Rosetta for Linux has **no AVX at all** (`vmovdqu ymm` raises SIGILL), so the SIMD JIT variants can't run here. This is the most realistic x86 timing we have for the baselines. |
| **arm** | native macOS arm64, Apple M3 Max | interpreter, native | No JIT on arm64. A native interpreter reference. |

The macOS Rosetta 2 translation daemon (`oahd`) wedged once during crash debugging, so the first round
of x86 numbers was taken in Docker (qemu, rosetta). After it was restarted, the full test suite (450
tests, 14 of them SIMD) passed as `x86_64-apple-darwin` under Rosetta 2, and the benchmark was re-run
there (`target/simd-bench/results-x86-rosetta-macos.json`).

QEMU's scalar code is disproportionately slow, so its SIMD ratios run higher than Rosetta 2's. That
gap is largest for short messages: E at 64 B is 9.0× on qemu but 2.3× on Rosetta 2. Treat the
Rosetta 2 ratios as the conservative estimate.

**Method:**
- Each data point is the median over up to 15 samples of the per-run time. Each sample batches
  enough runs to last at least 2 ms.
- Execution is warm: compiled once, timed afterwards. JIT compile time is reported separately as the
  median of 7 cold compiles.
- The input is identical for all variants (`i % 251`, the BLAKE3 test vector pattern). Every
  benchmark asserts the digest against `blake3::hash` before and after timing.
- CUs: every instruction costs 1, vector instructions included. CU counts are listed but never used
  as evidence of speed.
- Cycles per byte are not reported because there's no meaningful TSC under emulation. The harness
  records `rdtsc` ticks, which become meaningful on real hardware.

Variant key (plan section 8):
- **A** scalar BLAKE3 in sBPF, JIT
- **B** SIMD BLAKE3 in the interpreter
- **C** AVX2 JIT, vector state in *ordinary VM memory* (memory mode)
- **D** AVX2 JIT, dedicated vector region, uncached
- **E** dedicated region with register caching
- **F** native `blake3` crate, a host reference only

SIMD8 is the 8-way kernel (with SIMD1 for tails). SIMD1 is the single-state kernel throughout.

## Full hash throughput (MiB/s), macOS Rosetta 2

| variant | 64 B | 1 KiB | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 64 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A scalar, JIT | 21.33 | 22.65 | 22.18 | 21.69 | 21.52 | 21.43 | 21.45 | 21.29 |
| scalar, interp | 4.16 | 4.04 | 4.09 | 4.08 | 4.04 | 3.95 | 3.98 | 3.93 |
| B SIMD8, interp | 7.78 | 8.27 | 8.00 | 7.98 | 22.36 | 23.08 | 24.04 | 24.33 |
| SIMD1, interp | 7.79 | 8.29 | 8.13 | 7.97 | 7.91 | 7.84 | 7.78 | 7.80 |
| C SIMD8, ordinary mem, JIT | 7.06 | 7.33 | 7.14 | 7.01 | 24.04 | 25.53 | 27.19 | 27.13 |
| D SIMD8, region, uncached | 32.25 | 37.67 | 36.34 | 36.07 | 181.88 | 204.35 | 225.64 | 233.75 |
| E SIMD8, region, cached | 48.18 | 60.54 | 58.19 | 57.79 | 286.84 | 326.20 | 349.58 | 366.58 |
| SIMD1, ordinary mem, JIT | 7.07 | 7.33 | 7.09 | 7.02 | 6.86 | 6.89 | 6.85 | 6.87 |
| SIMD1, region, uncached | 32.33 | 36.70 | 36.58 | 35.74 | 35.60 | 35.35 | 35.76 | 35.24 |
| SIMD1, region, cached | 48.00 | 60.25 | 57.77 | 57.66 | 57.37 | 57.46 | 57.62 | 57.34 |
| F native blake3 (host) | 510.58 | 567.83 | 1054.44 | 1875.68 | 1908.17 | 1956.92 | 1969.55 | 1965.72 |

## Speedup vs A (scalar sBPF, JIT), macOS Rosetta 2

| variant | 64 B | 1 KiB | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 64 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A scalar, JIT | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| scalar, interp | 0.20× | 0.18× | 0.18× | 0.19× | 0.19× | 0.18× | 0.19× | 0.18× |
| B SIMD8, interp | 0.36× | 0.36× | 0.36× | 0.37× | 1.04× | 1.08× | 1.12× | 1.14× |
| SIMD1, interp | 0.37× | 0.37× | 0.37× | 0.37× | 0.37× | 0.37× | 0.36× | 0.37× |
| C SIMD8, ordinary mem, JIT | 0.33× | 0.32× | 0.32× | 0.32× | 1.12× | 1.19× | 1.27× | 1.27× |
| D SIMD8, region, uncached | 1.51× | 1.66× | 1.64× | 1.66× | 8.45× | 9.54× | 10.52× | 10.98× |
| E SIMD8, region, cached | 2.26× | 2.67× | 2.62× | 2.66× | 13.33× | 15.22× | 16.30× | 17.22× |
| SIMD1, ordinary mem, JIT | 0.33× | 0.32× | 0.32× | 0.32× | 0.32× | 0.32× | 0.32× | 0.32× |
| SIMD1, region, uncached | 1.52× | 1.62× | 1.65× | 1.65× | 1.65× | 1.65× | 1.67× | 1.65× |
| SIMD1, region, cached | 2.25× | 2.66× | 2.61× | 2.66× | 2.67× | 2.68× | 2.69× | 2.69× |
| F native blake3 (host) | 23.94× | 25.07× | 47.55× | 86.49× | 88.67× | 91.33× | 91.83× | 92.32× |

## Full hash throughput (MiB/s), qemu

| variant | 64 B | 1 KiB | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 64 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A scalar, JIT | 1.69 | 1.89 | 1.81 | 1.78 | 1.80 | 1.77 | 1.82 | 1.84 |
| scalar, interp | 0.92 | 0.96 | 0.94 | 0.92 | 0.91 | 0.91 | 0.91 | 0.90 |
| B SIMD8, interp | 1.06 | 1.15 | 1.12 | 1.10 | 3.02 | 3.18 | 3.29 | 3.31 |
| SIMD1, interp | 1.07 | 1.15 | 1.09 | 1.10 | 1.09 | 1.09 | 1.08 | 1.08 |
| C SIMD8, ordinary mem, JIT | 0.84 | 0.90 | 0.86 | 0.86 | 2.14 | 2.25 | 2.34 | 2.36 |
| D SIMD8, region, uncached | 7.94 | 10.57 | 10.29 | 9.97 | 24.85 | 26.03 | 26.87 | 27.09 |
| E SIMD8, region, cached | 15.21 | 29.15 | 28.62 | 27.44 | 45.62 | 46.45 | 47.05 | 47.48 |
| SIMD1, ordinary mem, JIT | 0.85 | 0.89 | 0.87 | 0.86 | 0.86 | 0.86 | 0.85 | 0.85 |
| SIMD1, region, uncached | 7.88 | 10.56 | 10.31 | 10.17 | 10.15 | 10.09 | 10.09 | 10.07 |
| SIMD1, region, cached | 15.27 | 29.02 | 28.61 | 28.32 | 28.42 | 28.47 | 28.43 | 28.37 |
| F native blake3 (host) | 51.24 | 58.58 | 106.21 | 103.60 | 125.92 | 127.53 | 128.16 | 128.44 |

## Speedup vs A (scalar sBPF, JIT), qemu

| variant | 64 B | 1 KiB | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 64 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A scalar, JIT | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| scalar, interp | 0.54× | 0.51× | 0.52× | 0.52× | 0.51× | 0.51× | 0.50× | 0.49× |
| B SIMD8, interp | 0.63× | 0.61× | 0.62× | 0.62× | 1.68× | 1.79× | 1.81× | 1.80× |
| SIMD1, interp | 0.63× | 0.61× | 0.60× | 0.62× | 0.61× | 0.62× | 0.60× | 0.59× |
| C SIMD8, ordinary mem, JIT | 0.50× | 0.47× | 0.48× | 0.49× | 1.19× | 1.27× | 1.29× | 1.28× |
| D SIMD8, region, uncached | 4.70× | 5.59× | 5.67× | 5.61× | 13.83× | 14.67× | 14.77× | 14.72× |
| E SIMD8, region, cached | 9.00× | 15.41× | 15.78× | 15.43× | 25.39× | 26.18× | 25.86× | 25.79× |
| SIMD1, ordinary mem, JIT | 0.51× | 0.47× | 0.48× | 0.48× | 0.48× | 0.48× | 0.46× | 0.46× |
| SIMD1, region, uncached | 4.67× | 5.58× | 5.69× | 5.72× | 5.65× | 5.69× | 5.55× | 5.47× |
| SIMD1, region, cached | 9.03× | 15.33× | 15.77× | 15.92× | 15.82× | 16.05× | 15.63× | 15.41× |
| F native blake3 (host) | 30.32× | 30.95× | 58.56× | 58.25× | 70.08× | 71.88× | 70.45× | 69.78× |

## Baselines on Rosetta for Linux (MiB/s)

| variant | 64 B | 1 KiB | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 64 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A scalar, JIT | 17.96 | 19.68 | 19.03 | 18.83 | 18.72 | 18.59 | 18.57 | 18.61 |
| scalar, interp | 3.38 | 3.68 | 3.52 | 3.34 | 3.40 | 3.44 | 3.49 | 3.48 |
| B SIMD8, interp | 7.11 | 7.70 | 7.45 | 7.32 | 21.05 | 22.11 | 23.14 | 23.46 |
| SIMD1, interp | 7.13 | 7.71 | 7.49 | 7.40 | 7.37 | 7.30 | 7.30 | 7.28 |
| F native blake3 (host) | 809.73 | 567.49 | 1075.46 | 1881.18 | 1933.65 | 1959.99 | 1972.72 | 1969.96 |

## Interpreter on native arm64 (MiB/s)

| variant | 64 B | 1 KiB | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 64 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar, interp | 11.58 | 12.30 | 12.15 | 11.51 | 11.51 | 11.73 | 11.83 | 11.83 |
| B SIMD8, interp | 13.55 | 15.07 | 14.42 | 14.00 | 41.38 | 43.15 | 45.57 | 44.91 |
| SIMD1, interp | 14.12 | 15.12 | 14.46 | 13.43 | 14.00 | 14.02 | 14.33 | 14.10 |
| F native blake3 (host) | 1008.88 | 1070.72 | 1036.09 | 1952.96 | 1856.53 | 1967.42 | 1948.65 | 2011.20 |

## Simulated CU consumption (identical in every environment and engine)

| variant | 64 B | 1 KiB | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 64 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A scalar, JIT | 2,163 | 33,723 | 69,553 | 141,205 | 284,489 | 571,037 | 2,290,245 | 36,673,685 |
| scalar, interp | 2,163 | 33,723 | 69,553 | 141,205 | 284,489 | 571,037 | 2,290,245 | 36,673,685 |
| B SIMD8, interp | 387 | 5,517 | 11,394 | 23,135 | 16,270 | 30,965 | 119,035 | 1,879,535 |
| SIMD1, interp | 387 | 5,517 | 11,389 | 23,125 | 46,577 | 93,461 | 374,685 | 5,998,445 |
| C SIMD8, ordinary mem, JIT | 387 | 5,517 | 11,394 | 23,135 | 16,270 | 30,965 | 119,035 | 1,879,535 |
| D SIMD8, region, uncached | 387 | 5,517 | 11,394 | 23,135 | 16,270 | 30,965 | 119,035 | 1,879,535 |
| E SIMD8, region, cached | 387 | 5,517 | 11,394 | 23,135 | 16,270 | 30,965 | 119,035 | 1,879,535 |
| SIMD1, ordinary mem, JIT | 387 | 5,517 | 11,389 | 23,125 | 46,577 | 93,461 | 374,685 | 5,998,445 |
| SIMD1, region, uncached | 387 | 5,517 | 11,389 | 23,125 | 46,577 | 93,461 | 374,685 | 5,998,445 |
| SIMD1, region, cached | 387 | 5,517 | 11,389 | 23,125 | 46,577 | 93,461 | 374,685 | 5,998,445 |
| F native blake3 (host) | – | – | – | – | – | – | – | – |

## Compression function alone

qemu (SIMD8 rows are per compression, with 8 compressions per call):

| kernel / engine | ns per compression | CU per compression |
|---|---:|---:|
| scalar, JIT | 31,774 | 2095.1 |
| scalar, interpreter | 62,738 | 2095.1 |
| SIMD 1-state, interpreter | 52,436 | 333.1 |
| SIMD 8-way, interpreter | 17,694 | 112.9 |
| SIMD 1-state, ordinary memory, JIT | 67,589 | 333.1 |
| SIMD 8-way, ordinary memory, JIT | 24,555 | 112.9 |
| SIMD 1-state, region, JIT uncached | 5,509 | 333.1 |
| SIMD 8-way, region, JIT uncached | 2,283 | 112.9 |
| SIMD 1-state, region, JIT cached | 1,789 | 333.1 |
| SIMD 8-way, region, JIT cached | 1,376 | 112.9 |
| native blake3, 64 byte hash | 1,352 | – |

rosetta:

| kernel / engine | ns per compression | CU per compression |
|---|---:|---:|
| scalar, JIT | 3,070 | 2095.1 |
| scalar, interpreter | 16,389 | 2095.1 |
| SIMD 1-state, interpreter | 7,818 | 333.1 |
| SIMD 8-way, interpreter | 2,539 | 112.9 |
| native blake3, 64 byte hash | 120 | – |

## Register caching microbenchmark

`v0 += v1; v0 ^= v2; v0 += v3; v0 ^= v4`, repeated N times in one basic block, 1000 iterations.
Values are ns per vector op.

macOS Rosetta 2:

| repeat | interp | C_mem | D_direct | E_cached | D ÷ E |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.46 | 19.48 | 5.62 | 3.62 | 1.55× |
| 4 | 21.86 | 20.37 | 6.52 | 2.68 | 2.43× |
| 16 | 20.94 | 19.55 | 7.72 | 3.69 | 2.09× |
| 64 | 20.60 | 19.63 | 7.88 | 3.75 | 2.10× |

qemu:

| repeat | interp | C_mem | D_direct | E_cached |
|---:|---:|---:|---:|---:|
| 1 | 187.17 | 206.62 | 18.59 | 11.74 |
| 4 | 178.24 | 208.48 | 17.93 | 9.12 |
| 16 | 170.63 | 209.65 | 16.82 | 7.90 |
| 64 | 169.75 | 229.22 | 16.78 | 7.76 |

Interpreter only, rosetta: N=1: 24.0 ns, N=4: 22.2 ns, N=16: 21.2 ns, N=64: 20.6 ns.
Native arm64: N=1: 11.9 ns, N=4: 11.6 ns, N=16: 10.9 ns, N=64: 10.7 ns.

### Generated assembly (N = 4, loop body)

**E, register-cached:**
- `v0` stays in `ymm0` for all 16 ops.
- `v1..v4` are read-only, so they're used as memory operands from L1 instead of taking registers.
- There is exactly one load before the ops and one store at the block end, before the meter check
  and back-edge.

```asm
mov r11,QWORD PTR [rdi+0x20d79d41]
test r11,r11
jne 0x1b
mov r11d,0x8
jmp 0xfffffffffffff5a7
mov r11,QWORD PTR [rdi+0x20d79d41]
vmovdqu ymm0,YMMWORD PTR [r11]
vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x20]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x40]
vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x60]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x80]

vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x20]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x40]
vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x60]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x80]

vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x20]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x40]
vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x60]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x80]

vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x20]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x40]
vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x60]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x80]

sub rdx,0x1
mov r11,QWORD PTR [rdi+0x20d79d41]
test r11,r11
jne 0xb5
mov r11d,0x19
jmp 0xfffffffffffff5a7
mov r11,QWORD PTR [rdi+0x20d79d41]
vmovdqu YMMWORD PTR [r11],ymm0
mov r11d,0x19022271
add r11,0xffffffffe6fddda8
cmp r10,r11
jbe 0xfffffffffffff4d9
add r10,0xffffffffffffffee
cmp rdx,0x0
jne 0x0
add r10,0x12

```

**D, direct:** a load and a store around every op (first five ops shown).

```asm
mov r11,QWORD PTR [rdi+0x20d79d41]
test r11,r11
jne 0x1b
mov r11d,0x8
jmp 0xfffffffffffff5a7
mov r11,QWORD PTR [rdi+0x20d79d41]
vmovdqu ymm0,YMMWORD PTR [r11]
vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x20]
vmovdqu YMMWORD PTR [r11],ymm0
vmovdqu ymm0,YMMWORD PTR [r11]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x40]
vmovdqu YMMWORD PTR [r11],ymm0
vmovdqu ymm0,YMMWORD PTR [r11]
vpaddd ymm0,ymm0,YMMWORD PTR [r11+0x60]
vmovdqu YMMWORD PTR [r11],ymm0
vmovdqu ymm0,YMMWORD PTR [r11]
vpxor ymm0,ymm0,YMMWORD PTR [r11+0x80]
```

**C, memory mode:** every op computes the three VM addresses, saves the caller-saved registers, and
calls the host translation (`MemoryMapping` lookup and bounds checks for all operands). Then it
reloads the host pointers and only then executes `vpaddd`. One op is shown:

```asm
mov r11d,0x0
add r11,r14
mov QWORD PTR [rdi+0x20d79d49],r11
mov r11d,0x20
add r11,r14
mov QWORD PTR [rdi+0x20d79d51],r11
mov r11d,0x0
add r11,r14
mov QWORD PTR [rdi+0x20d79d59],r11
vzeroupper
push rax
push rcx
push rdx
push rsi
push rdi
push r8
push r9
push r10
push r11
movabs rcx,0x802000200020

lea rdx,[rdi+0x20d79d49]
mov rsi,QWORD PTR [rdi+0x20d79d19]
lea rdi,[rdi+0x20d79cd9]
movabs rax,0x40000df120

rex.W call rax
add rsp,0x0
pop r11
pop r10
pop r9
pop r8
pop rdi
pop rsi
pop rdx
pop rcx
pop rax
cmp QWORD PTR [rdi+0x20d79cd9],0x0

mov r11d,0x8
jne 0xfffffffffffff560
mov r11,QWORD PTR [rdi+0x20d79d49]
vmovdqu ymm0,YMMWORD PTR [r11]
mov r11,QWORD PTR [rdi+0x20d79d51]
vpaddd ymm0,ymm0,YMMWORD PTR [r11]
mov r11,QWORD PTR [rdi+0x20d79d59]
vmovdqu YMMWORD PTR [r11],ymm0
mov r11d,0x0
add r11,r14
mov QWORD PTR [rdi+0x20d79d49],r11
mov r11d,0x40
add r11,r14
```

## Code generation statistics (full hash program, static counts)

| variant | code bytes | native insns | vector insns | vmovdqu loads | vmovdqu stores | mem operands | spills | reloads | cache hits | translation calls (static) | compile time (QEMU) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A scalar, JIT | 28,626 | 6,259 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.27 ms |
| C SIMD8, ordinary mem, JIT | 219,095 | 52,032 | 1177 | 1136 | 1198 | 823 | 0 | 0 | 0 | 1177 | 1.58 ms |
| D SIMD8, region, uncached | 45,167 | 6,952 | 1177 | 1136 | 1198 | 823 | 0 | 0 | 0 | 37 | 0.70 ms |
| E SIMD8, region, cached | 31,100 | 5,609 | 1177 | 478 | 498 | 127 | 411 | 396 | 1354 | 37 | 0.58 ms |
| SIMD1, ordinary mem, JIT | 65,034 | 15,345 | 331 | 285 | 331 | 254 | 0 | 0 | 0 | 331 | 0.49 ms |
| SIMD1, region, uncached | 17,059 | 2,775 | 331 | 285 | 331 | 254 | 0 | 0 | 0 | 13 | 0.24 ms |
| SIMD1, region, cached | 10,831 | 2,218 | 331 | 23 | 26 | 9 | 0 | 0 | 507 | 13 | 0.19 ms |

The compile times are QEMU times, so expect about 10× less on hardware. The ratios between variants
hold.

## Findings

1. **SIMD vs scalar sBPF.**
   - At 1 MiB, E (SIMD8, cached) is **17.2×** faster than A under Rosetta 2 (**25.8×** on qemu),
     and D (uncached) is **11.0×** faster (**14.7×** on qemu).
   - Short messages use the single-state kernel and gain **2.3×** at 64 B and **2.7×** at 1 KiB
     under Rosetta 2. qemu shows 9× and 15×, but its scalar code is disproportionately slow.
   - The 8-way kernel helps from 8 KiB up (a full group of 8 chunks). Below that, SIMD8 and SIMD1
     are identical.
   - Part of the gain is from avoiding scalar sBPF's expensive memory accesses: every scalar
     load/store in this JIT is a host call, and scalar BLAKE3 needs about 10 per G function.
2. **Memory traffic removed by register residency.**
   - In the microbench the generated loop body goes from 2 memory ops per vector op to 2 per
     *block*, confirmed in the assembly above.
   - E is 1.6–2.2× faster per op than D, and the gap grows with block length (amortized block
     overhead).
   - On the full BLAKE3 program, caching removes 58% of loads and stores for SIMD8 (1136/1198 →
     478/498) and 92% for SIMD1 (285/331 → 23/26). That gives E/D = **1.57×** for SIMD8 and
     **1.63×** for SIMD1 under Rosetta 2 (1.75× and 2.8× on qemu).
   - In the Rosetta 2 microbench, E is 1.55–2.43× faster per op than D.
   - SIMD8 is limited by register pressure: 32 live slots (16 state + 16 message) on 15 YMM
     registers cause 411 spills and 396 reloads (static count, almost all in the compress8 body) with naive LRU.
     Belady-style eviction would help. Keeping message words as memory operands, which is what the
     official AVX2 code does, would help more.
3. **Does the dedicated region help?** Yes, by a wide margin, but mostly for an implementation
   reason.
   - C (identical program, vector file in the heap) is only **1.27×** faster than scalar and
     **8.6×** slower than D under Rosetta 2 (1.28× and 11.5× on qemu). Every memory-mode op pays a host call to translate and bounds-check
     up to three operands.
   - With statically known slot indices, the JIT checks the region once and addresses
     `[r11 + 32*i]` directly. Statically known addresses are also what makes register caching
     possible at all.
   - An inline fast path for translation would narrow C/D, but couldn't enable caching: dynamic
     addresses can alias.
4. **Distance to native.**
   - Under Rosetta 2, native `blake3` runs at 1966 MiB/s against **367 MiB/s** for E and 21.3 MiB/s
     for A, so native is **5.4×** faster than E. On qemu the gap is 2.7×.
   - Both are translated, and Rosetta 2 splits every 256-bit op in two, so real AVX2 hardware may
     land elsewhere.
   - Remaining overheads:
     - the per-block flushes and translation calls for the 16 message `vload`s
     - the 2 transposes
     - spills
     - instruction meter checkpoints
     - scalar tree logic
5. **CUs.** Because each vector op costs 1 CU like any instruction, SIMD8 uses **19.5×** fewer CUs
   than scalar (36.7M → 1.88M at 1 MiB). Whether a vector op *should* cost 1 CU is a pricing
   decision; it isn't a measurement.
6. **Interpreter.**
   - SIMD8 in the interpreter beats the scalar interpreter by 3.7× (qemu), 6.7× (rosetta) and
     3.8× (arm), and 6.2× under macOS Rosetta 2. It's even 1.14–1.26× faster than the scalar JIT
     under both Rosettas at 1 MiB.
   - A vector op costs 10–24 ns in the interpreter: generic operand planning, translation and lane
     loops. Interpreter vector ops could be made much cheaper.

## Is it worth pursuing?

The architecture does what it set out to do:
- vector ops visible to the JIT
- lowered to native AVX2 one-to-one
- operands statically known
- virtual registers kept resident across arithmetic

It gives large speedups over scalar sBPF, and register caching is measurable and visible in the
machine code. The dedicated region is justified, chiefly because static slot addresses enable direct
addressing and caching.

The costs are just as clear:
- **Complexity:** the cache's coherence rules (block boundaries, aliasing scalar accesses,
  checkpoints, callx) produced two crashes during this experiment, both from scratch-register
  tracking. Every JIT change near memory accesses will have to respect them.
- **Register pressure:** wide kernels with more than 15 live vectors need a better allocator.

A syscall comparison (`sol_math`) wasn't implemented, so **no claim is made that this beats a
syscall**. A fair test would implement a batched BLAKE3 round interpreter as a syscall and time it in
the same harness. Given that variant E is within 5.4× of native under Rosetta 2 (2.7× on qemu), a syscall can at best
win that remaining factor, and only for workloads it was designed for.

## Reproduce

```sh
# correctness (any host; JIT tests need x86_64)
cargo test --release --features simd-experiment --test simd

# benchmarks -> target/simd-bench/results.json
cargo bench --features simd-experiment --bench simd_blake3
# under Rosetta 2 / Rosetta for Linux / CPUs hiding AVX2: SBPF_SIMD_ASSUME_AVX2=1
# quick run: SIMD_BENCH_QUICK=1; subset: SIMD_BENCH_FILTER=hash/65536 SIMD_BENCH_SKIP=C_,interp
```

On real x86 hardware, pin the frequency (performance governor, turbo off), use `taskset -c 2`, and
compare `ticks` (rdtsc) per byte across variants.
