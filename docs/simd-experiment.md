# Experimental SIMD support for sBPF

> Research prototype. Not a production API, not a proposed ISA extension. Everything lives behind
> the `simd-experiment` cargo feature **and** the runtime flag `Config::enable_simd_experiment`.

## Question

Can compiler/JIT-visible vector operations, backed by virtual wide registers, give sBPF programs
efficient native SIMD, as a general-purpose alternative to a syscall that interprets arithmetic
(`sol_math`)? BLAKE3 is the workload.

## Implementation strategy (and why not `CALL_IMM` builtins)

`BuiltinFunctionDefinition::codegen(jit: &mut JitCompiler<C>)` lets a builtin emit inline code. But
the hook gets **no instruction operands**, and the interpreter path passes `r1..r5` dynamically. A
vector op would have to carry its slot numbers in scalar registers, which the JIT can't resolve at
compile time, so it can't allocate host registers for them. Static syscalls (SBPFv3+) also reserve
`src` to tell syscalls apart from internal calls, which leaves only `dst`/`off` for operands.

So the prototype uses **one feature-gated opcode**, `VOP = 0xe7`. That is the ALU64 class with
operation `0xe0`, which is otherwise unused. All operands are encoded statically:

```text
opc = 0xe7
dst = vector file base register (memory mode only, must be 0 in slot mode)
src = scalar register (address for vload/vstore, value for vbcast32)
off = memory offset (vload/vstore) or 8 bit immediate (vror32x8, vshufd, vblendd)
imm = bits  0.. 5  vd   destination slot
      bits  6..11  va   first source slot
      bits 12..17  vb   second source slot
      bits 18..23  sub-operation
      bit  24      memory mode
      bits 25..31  reserved (zero)
```

Each `VOP` is exactly one well-defined VM operation. There's no sequence interpreter.

| sub-op | mnemonic | semantics (8 little-endian u32 lanes) | AVX2 lowering |
|---|---|---|---|
| 0x00 | `vload256 vd, [rS+off]` | `v[d] = mem[r+off .. +32]` | `vmovdqu` |
| 0x01 | `vstore256 [rS+off], va` | `mem[..] = v[a]` | `vmovdqu` |
| 0x02 | `vadd32x8 vd, va, vb` | wrapping u32 add | `vpaddd` |
| 0x03 | `vxor256` | xor | `vpxor` |
| 0x04 | `vand256` | and | `vpand` |
| 0x05 | `vor256` | or | `vpor` |
| 0x06 | `vror32x8 vd, va, n` | `rotate_right(n)`, n in 0..32 | `vpshufb` for 8/16, else `vpsrld`+`vpslld`+`vpor` |
| 0x07 | `vshufd vd, va, imm8` | pshufd within each 128 bit half | `vpshufd` |
| 0x08 | `vtranspose8 vd` | transposes the 8x8 u32 matrix in `vd..vd+8` | 8×`vpunpck{l,h}dq`, 8×`vpunpck{l,h}qdq`, 8×`vperm2i128` |
| 0x09 | `vbcast32 vd, rS` | `[rS as u32; 8]` | `vmovd` + `vpbroadcastd` |
| 0x0a | `vmov vd, va` | copy | `vmovdqa` |
| 0x0b | `vpermd vd, va, vb` | `v[d][i] = v[a][v[b][i] & 7]` | `vpermd` |
| 0x0c | `vblendd vd, va, vb, imm8` | per-lane select | `vpblendd` |

The plan's minimum set (`vload/vstore/vadd/vxor/vror/vand/vor`) wasn't enough for real BLAKE3, so
four more ops were added:
- **8-way kernel:** needs the transpose (message words of 8 chunks to lanes) and `vbcast32`
  (counters, flags, IV).
- **Single-state kernel:** needs `vshufd` (diagonalization) and `vpermd`/`vblendd` (in-register
  message permutation).

## Intrinsic call form (SIMD-0645 style, no new opcode)

The operations are also available without extending the ISA, as static syscalls in the style of
`sol_multi3` (SIMD-0645). This is `VecMode::Call` in the assembler, `src/simd/intrinsics.rs`, and the
form used by compiled Rust programs (`simd-falcon/`).

```text
mov64 r1, <packed operands>       ; compile time constant
call  sol_vadd32x8                ; CALL_IMM, src = 0, imm = murmur32("sol_vadd32x8")
```

* **Encoding:** one static call identifier per operation, `sol_<mnemonic>` (21 of them, including
  `sol_vsub32x8`, `sol_vmullo32x8`, `sol_vminu32x8`, `sol_vsrl32x8`, `sol_vsll32x8`,
  `sol_vsllv32x8`, `sol_vsrlv32x8`, `sol_vandn256`, which exist only for compiled programs such as the
  Falcon verifier). `dst`, `src` and `off` of the `CALL_IMM` stay zero, so the bytecode is plain eBPF.
* **ABI:** `r1` holds the packed operands, `r2` the scalar operand (vload/vstore address, vbcast
  value). A Rust wrapper `sol_vadd32x8(ops(V0, V0, V1))` compiles to exactly `mov64 r1, imm` +
  `call`, and compilers reload `r1` before every call because it is caller saved.
* **Packing:** 8 bit slot numbers (256 slots, 8 KiB region): `vd | va << 8 | vb << 16`, an imm8 in
  bits 16..24 for one-source ops, a 16 bit offset in bits 8..24 for vload/vstore. `vblendd` uses
  7 bit slots (`vd | va << 7 | vb << 14 | imm8 << 21`). Every value is below `2^29`.
* **Results:** `r0 = 0`, every other register unchanged, 1 CU for the `CALL_IMM`, so each vector
  op costs 2 CUs including the `mov64`.
* **Execution:** every engine runs the reference semantics. The JIT's `codegen` hook checks a block
  local constant tracker. If `r1` holds a valid constant it emits the AVX2 code of the opcode form,
  register caching included, plus `mov eax, 0`. Otherwise it emits a regular syscall to the
  reference implementation. Results and CUs are identical either way.
* **Compatibility:** a runtime that doesn't know the identifiers rejects the program at deployment
  (`LocalVerifier`, `InvalidSyscall`), and an unregistered call fails with `UnsupportedInstruction`.

### callx and the register cache

Compiled programs always contain some `callx` (e.g. in `core::fmt`). A `callx` may enter any
instruction, but the JIT keeps vector state in registers inside a run of vector code. With the
experiment enabled, `callx` therefore may not enter an instruction that follows a vector op with only
pure scalar ALU instructions in between and no block start (`simd::SimdAnalysis::callx_forbidden`).
Both engines throw `UnsupportedInstruction` for such a target; the JIT marks those entries in the pc
table. Compiled code only ever targets function entries, so this never triggers in practice, and the
register cache stays on for programs with `callx`.

## Vector memory region

* `MM_VECTOR_START = 0x6 << 32`. `0x5 << 32` stays reserved for the static-sysvars proposal.
* 256 slots × 32 bytes = 8 KiB. Slot `vN` is at `0x600000000 + 32*N`. The opcode form addresses
  slots 0..64 (6 bit fields), the intrinsic call form all 256.
* The embedder owns a `simd::VectorMemory`, maps it with `region()` like stack and heap, and attaches
  it with `EbpfVm::set_vector_memory`. `execute_program` zeroes it on every invocation.
* It's ordinary mapped memory, so normal scalar loads and stores work on it with the usual bounds and
  permission checks (tested). If it isn't attached, slot-mode ops fail with `UnsupportedInstruction`
  in both engines. They never fault the host.

**Slot mode** vs **memory mode** (imm bit 24) run *the same program*. Only the location of the vector
file differs:
* **Slot mode:** the vector file is the dedicated region. Its host address is known, so the JIT
  addresses it directly and can cache slots in YMM registers.
* **Memory mode:** the vector file is at `r[dst]` in ordinary VM memory (heap in the benchmarks).
  Every operand is translated and bounds-checked through `MemoryMapping` at runtime, which is how
  every other sBPF memory access in this JIT works. This is benchmark variant **C**. It tells us
  whether the dedicated region earns its keep.

## Interpreter (reference semantics)

`simd::interpret` is the reference. Lanes are little-endian u32, with Rust `wrapping_add` and
`rotate_right`.

**Translation order:** operands that need translation (memory addresses, and every slot in memory
mode) go through `simd::translate_operands` in a fixed order: sources first, then the destination.
The JIT calls the same function (`jit_translate_operands`, an `extern "C"` shim), so the reported
error for an instruction with several bad operands is identical in both engines.

**Two-pass translation:** the access violation handler can replace regions, which would invalidate
host pointers translated earlier. So the first pass may run the handler and the second pass only
re-resolves addresses. That leaves all host pointers valid at the same time.

**Read-before-write:** all sources are read before the destination is written, so overlapping and
aliasing operands are well defined.

## Metering

Every vector instruction costs **1 CU**, the same as any other sBPF instruction. It's provisional,
but explicit and identical in both engines. The tests assert that interpreter and JIT CU counts are
equal for every program, including runs that exhaust the meter in the middle of vector code. CU
counts aren't used as evidence of speed.

## AVX2 JIT

### Encoding

`x86::VexInstruction` is a new 2/3-byte VEX encoder. It supports register, `[base+disp]` and
RIP-relative operands. The existing emitter had no VEX support. The `vpshufb` byte masks for the
8- and 16-bit rotations are emitted once into the text section, after the subroutines, and read
RIP-relative.

### Host feature check

`JitCompiler::new` returns `EbpfError::SimdHostUnsupported` if the program contains vector ops and
the host has no AVX2. `Config::simd_assume_avx2` skips the check. It exists only for Rosetta 2, which
executes AVX2 but hides it from CPUID: the bench harness sets it automatically when
`sysctl.proc_translated == 1`. No host-dependent semantics: the JIT either emits AVX2 or refuses to
compile. It never switches to a different implementation.

### Strategy A: direct, memory-backed (`SimdJitStrategy::Direct`)

Every op loads its operands from the slot memory and stores the result back:

```asm
mov     r11, [rdi + vector_base]   ; once per run of vector instructions
vmovdqu ymm0, [r11 + 32*a]
vpaddd  ymm0, ymm0, [r11 + 32*b]
vmovdqu [r11 + 32*d], ymm0
```

In memory mode, the addresses are first translated by one host call per instruction (up to three
operands), and the host pointers are read back from `EbpfVm::simd_scratch`.

### Strategy B: basic-block-local register cache (`SimdJitStrategy::RegisterCached`)

`SimdJitState` tracks `{reg, dirty}` per slot over 15 YMM registers (`ymm15` is scratch), with LRU
eviction. The guarantees:

* **Block boundaries.**
  - Block starts are jump targets, internal call targets and the entry point, from a pre-scan.
  - At a block start the cache is written back and emptied *before* the pc offset is recorded. The
    fall-through path pays for the write-back, and jumps arrive with an empty cache.
  - Every branch, call, exit, syscall and scalar memory access (all of which call host code that
    clobbers YMM) writes back dirty slots and invalidates first. The same goes for any other
    instruction that may throw, such as division.
  - Pure scalar ALU instructions leave the cache alone.
* **Aliasing.** Scalar loads and stores might hit the vector region through any dynamic address, so
  they always see coherent memory (write-back first). They also invalidate, so later reads see
  partial writes; a test covers this.
* **Memory operands of vector ops.** `vload`/`vstore` and all memory-mode ops are translated through
  a host call. The cache is flushed and invalidated before them.
* **Exceptional exits.**
  - Meter checkpoints write back dirty slots but keep the registers.
  - Memory faults and calls are preceded by a full flush.
  - So the vector region is up to date at every point where the program can stop.
* **Sources.** A source that isn't cached is used directly as a memory operand
  (`vpaddd ymm, ymm, [r11+disp]`) and doesn't take a register.
* **`vtranspose8`.** It flushes, then transposes in registers and leaves the eight results resident
  and dirty.
* **`callx`.** It can enter any instruction, so programs containing `callx` fall back to
  `Direct`. They also stop tracking R11 across instructions.
* **AVX-SSE transitions.** `vzeroupper` is emitted before every call into host code and on JIT exit.

### Bugs found while building it

Two R11 tracking bugs crashed the cached JIT while the interpreter and the direct JIT were correct:
1. Meter checkpoints load the pc into R11.
2. An impure scalar instruction's write-back sets R11 to the vector base, and then the instruction
   itself overwrites R11.

Both were fixed by invalidating R11 knowledge at checkpoints and after every non-vector instruction.
This is exactly the kind of complexity cost the cache brings. It had to be found from a macOS crash
report: the fault address was R11 (a pc value) plus the slot offset.

## BLAKE3 programs

`simd::blake3_program` generates SBPFv4 bytecode with a small macro assembler (`simd::asm`):

* **Scalar** (variant A): 32-bit ALU ops. The 16-word state lives in the heap (sBPF has 10 usable
  registers), so each G call does 6 loads and 4 stores. That's roughly what LLVM would produce for
  BPF.
* **Simd1**, single-state:
  - One state row per vector (lanes 0..3; the upper lanes compute deterministic garbage that is never
    observed).
  - Diagonalization uses `vshufd`.
  - Message words for each round come from `vpermd` + `vblendd` on the two message halves held in
    registers, so there are no scalar loads.
* **Simd8**, eight chunks in parallel:
  - `v0..v15` are the state words across 8 chunks.
  - Each 64-byte block of the 8 chunks is loaded with 16 `vload256`s and two `vtranspose8`s.
  - Blocks within a chunk are processed sequentially with CHUNK_START/CHUNK_END flags.
  - Chunk counters come from `vbcast32(c) + [0..7]`.
  - Parent nodes are also compressed 8 at a time. The CV array is the input, with a stride of 64.
  - The CVs are transposed back and stored.
* **Tree:** pair adjacent CVs left to right and carry the odd one up. That's equivalent to BLAKE3's
  left-complete tree. The final parent gets ROOT, or the single chunk does if there is only one.
  Leftover full chunks and the partial last chunk use Simd1. Partial blocks are zero-padded.

All variants match `blake3::hash` byte for byte at every test size, from 0 bytes to 1 MiB.

## Files

| file | content |
|---|---|
| `src/simd/mod.rs` | encoding, decoding, reference semantics, operand translation, `VectorMemory` |
| `src/simd_jit.rs` | AVX2 lowering, register cache, intrinsic constant tracking, statistics (child module of `jit`) |
| `src/simd/intrinsics.rs` | static call (SIMD-0645 style) form: identifiers, operand packing, builtins |
| `src/x86.rs` | `VexInstruction` encoder |
| `src/simd/asm.rs` | macro assembler for the generated programs |
| `src/simd/blake3_program.rs` | BLAKE3 hash / compression / microbench program generators |
| `src/simd/harness.rs` | self contained runner used by tests and benches |
| `tests/simd.rs` | correctness tests |
| `benches/simd_blake3.rs` | benchmark harness (writes JSON) |
| `docs/simd-benchmarks.md` | results |
