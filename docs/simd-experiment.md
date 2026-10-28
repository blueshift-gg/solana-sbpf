# Experimental SIMD support for sBPF

> Research prototype. Not a production API, not a proposed ISA extension. Everything lives behind
> the `simd-experiment` cargo feature **and** the runtime flag `Config::enable_simd_experiment`.

## Question

Can compiler/JIT-visible vector operations, backed by virtual wide registers, give sBPF programs
efficient native SIMD, as a general-purpose alternative to a syscall that interprets arithmetic
(`sol_math`)? BLAKE3 is the workload.

## Proposal shape

What this experiment would propose, separated from the research scaffolding:

* **Vehicle: static syscall intrinsics only** (SIMD-0645 style, see below). The custom opcode
  `VOP = 0xe7` is research scaffolding for comparing lowerings and needs an ISA change; the
  intrinsics express everything without one. The opcode form and `SimdJitStrategy::Direct` stay
  in the experiment only as baselines (the benchmarks and `intrinsic_calls_lower_to_inline_avx2` compare
  against them) and would not be part of a proposal.
* **The core set: AVX instructions as intrinsics.** The integer subset of AVX2 and AVX-512 (F, BW,
  VL, DQ, CD, IFMA, VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ) plus GFNI and VPCLMULQDQ, each operation
  with a 256 bit and (except moves / lane extraction) a 512 bit form on slot pairs. Excluded on
  purpose: floating point (sBPF has none, and float results raise determinism questions), AES
  (nothing on-chain is confidential, little verification demand, a wide surface of conventions to
  get wrong), gather / scatter (needs a separate bounded design), per operation merge masking
  (`blendm` instead) and the private vector file's lack of memory forms (bulk and masked
  load / store instead). The operations with reserved numbers (0x27, 0x4f..0x52) were removed.
* **The composite set, proposed separately with benchmarks.** Not x86 instructions but short native
  sequences priced as one operation, where the CU savings are large: `transpose8x8` (u32, BLAKE3
  8-way), `transpose4x4q` / `transpose8x8q` (Keccak: 30 instead of ~40 operations per round) and
  `mont16` (Falcon's NTT: -870 operations).
* **Pricing.** Every operation costs 2 CUs (`mov64` + `call`) at either width. A 512 bit operation
  is two native operations on AVX2-only validators; the proposal should state that explicitly, as
  well as the operations which are slow there (long templates or scalar loops: `madd52`, VBMI,
  GFNI, `multishift`, `conflict`, 8 / 16 bit `compress` / `expand`; 512 bit `conflict` is a host
  call).

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
dst = scalar destination register (vext64, vmovmsk8), otherwise 0
src = scalar source register (address of vload/vstore, value of vbcast / vins64)
off = memory offset (vload/vstore), 8 bit immediate (vror32x8, vshufd, vblendd, ...) or lane
      (vins64, vext64)
imm = bits  0.. 5  vd   destination slot
      bits  6..11  va   first source slot
      bits 12..17  vb   second source slot (slot count - 1 of vload/vstore)
      bits 18..23  sub-operation (low 6 bits)
      bit  24      512 bit operation on slot pairs
      bit  25      sub-operation (bit 6)
      bits 26..31  reserved (zero)
```

Each `VOP` is exactly one well-defined VM operation. There's no sequence interpreter.

### Operations

Lanes are little-endian. Every operation has a 256 bit form on one slot; most also have a 512 bit
form on a **slot pair** `(v2k, v2k + 1)` (all slot operands even, low half in the even slot), which
is the 256 bit operation on both halves unless noted. The intrinsic name (next section) is
`sol_<isa>_<op>`, the prefix naming the ISA level that has the operation as one instruction.

| sub-op | 256 bit name | 512 bit name | semantics | AVX2 / AVX-512 lowering |
|---|---|---|---|---|
| 0x00 | `sol_avx_load256` | | `v[vd + i] = mem[r + off + 32 i]`, `vb + 1` slots (at most 32) | one host call: translation + copy |
| 0x01 | `sol_avx_store256` | | `mem[..] = v[va + i]`, `vb + 1` slots | one host call |
| 0x02 | `sol_avx2_add32x8` | `sol_avx512_add32x16` | wrapping u32 add | `vpaddd` |
| 0x03 | `sol_avx2_xor256` | `sol_avx512_xor512` | xor | `vpxor` / `vpxord` |
| 0x04 | `sol_avx2_and256` | `sol_avx512_and512` | and | `vpand` / `vpandd` |
| 0x05 | `sol_avx2_or256` | `sol_avx512_or512` | or | `vpor` / `vpord` |
| 0x06 | `sol_avx512_ror32x8` | `sol_avx512_ror32x16` | u32 `rotate_right(off)` | `vpshufb` or 3 shifts / `vprord` |
| 0x07 | `sol_avx2_shufd256` | `sol_avx512_shufd512` | pshufd per 128 bit lane | `vpshufd` |
| 0x08 | `sol_avx2_transpose8x8` | | 8 x 8 u32 matrix in `vd..vd + 8` | 24 instructions |
| 0x09 | `sol_avx2_bcast32x8` | `sol_avx512_bcast32x16` | `[r as u32; n]` | `vmovd` + `vpbroadcastd` / `vpbroadcastd zmm, r32` |
| 0x0a | `sol_avx_mov256` | `sol_avx512_mov512` | copy | `vmovdqa` |
| 0x0b | `sol_avx2_permd256` | `sol_avx512_permd512` | `v[d][i] = v[a][v[b][i] & (L - 1)]` over all `L` lanes (cross half for 512) | `vpermd`; 512 on AVX2: 2 `vpermd` + `vblendvps` per half |
| 0x0c | `sol_avx2_blendd256` | `sol_avx512_blendd512` | u32 select by imm8 (per 256 bit half) | `vpblendd` / `vpblendmd` + opmask |
| 0x0d | `sol_avx2_sub32x8` | `sol_avx512_sub32x16` | wrapping u32 sub | `vpsubd` |
| 0x0e | `sol_avx2_mullo32x8` | `sol_avx512_mullo32x16` | low u32 product | `vpmulld` |
| 0x0f | `sol_avx2_minu32x8` | `sol_avx512_minu32x16` | u32 minimum | `vpminud` |
| 0x10, 0x11 | `sol_avx2_srl32x8`, `sll32x8` | `sol_avx512_srl32x16`, `sll32x16` | shift by imm8 | `vpsrld`, `vpslld` |
| 0x12, 0x13 | `sol_avx2_sllv32x8`, `srlv32x8` | `..x16` | per lane shift, 0 above 31 | `vpsllvd`, `vpsrlvd` |
| 0x14 | `sol_avx2_andn256` | `sol_avx512_andn512` | `!a & b` | `vpandn` |
| 0x15 | `sol_avx512_rolv32x8` | `sol_avx512_rolv32x16` | u32 `rotate_left(b % 32)` | 7 instructions / `vprolvd` |
| 0x16 | `sol_avx512_ternlog256` | `sol_avx512_ternlog512` | `d = f(d, a, b)`, truth table imm8 | 1 to 6 instructions / `vpternlogd` |
| 0x17..0x19 | `sol_avx2_add16x16`, `sub16x16`, `mullo16x16` | `..16x32` | u16 lanes, wrapping | `vpaddw`, `vpsubw`, `vpmullw` |
| 0x1a | `sol_avx2_mulhi16x16` | `sol_avx512_mulhi16x32` | high half of signed i16 products | `vpmulhw` |
| 0x1b | `sol_avx2_minu16x16` | `sol_avx512_minu16x32` | u16 minimum | `vpminuw` |
| 0x1c | `sol_avx2_madd16x16` | `sol_avx512_madd16x32` | i16 products summed in pairs into i32 | `vpmaddwd` |
| 0x1d | `sol_avx2_blendw256` | `sol_avx512_blendw512` | u16 select by imm8 (per 128 bit lane) | `vpblendw` / `vpblendmw` + opmask |
| 0x1e | `sol_avx_ins64` | | `v[vd].u64[off] = r` | `vmovq` + `vpbroadcastq` + `vpblendd` |
| 0x1f | `sol_avx_ext64` | | `r = v[va].u64[off]` | `vmovq` / `vpextrq` (+ `vextracti128`), or a scalar load |
| 0x20 | `sol_avx2_bcast64x4` | `sol_avx512_bcast64x8` | `[r; n]` | `vmovq` + `vpbroadcastq` / `vpbroadcastq zmm, r64` |
| 0x21 | `sol_avx2_movmsk8x32` | `sol_avx512_movmsk8x64` | byte sign bits | `vpmovmskb` / `vpmovb2m` + `kmovq` |
| 0x22 | `sol_avx512_permi2d256` | `sol_avx512_permi2d512` | `v[d][i] = (v[a] ++ v[b])[v[d][i] & (2 L - 1)]` | 4 / 9 instructions per half / `vpermi2d` |
| 0x23 | `sol_avx512_rolv64x4` | `sol_avx512_rolv64x8` | u64 `rotate_left(b % 64)` | 7 instructions / `vprolvq` |
| 0x24 | `sol_avx2_transpose4x4q` | `sol_avx512_transpose8x8q` | u64 matrix: rows `vd..vd + 4` / pairs `vd, vd + 2, .., vd + 14` | 8 / 32 instructions / 24 zmm instructions |
| 0x25 | `sol_avx2_shufb256` | `sol_avx512_shufb512` | byte shuffle per 128 bit lane (index bit 7 zeroes) | `vpshufb` |
| 0x26 | `sol_avx512_permt2d256` | `sol_avx512_permt2d512` | `v[d][i] = (v[d] ++ v[b])[v[a][i] & (2 L - 1)]` (first table overwritten, index kept) | 4 / 9 instructions per half / `vpermt2d` |
| 0x28, 0x29 | `sol_avx2_add64x4`, `sub64x4` | `sol_avx512_add64x8`, `sub64x8` | wrapping u64 add / sub | `vpaddq`, `vpsubq` |
| 0x2a..0x2c | `sol_avx2_sll64x4`, `srl64x4`, `sol_avx512_sra64x4` | `..64x8` | u64 shift by imm (0..64), arithmetic right | `vpsllq`, `vpsrlq` / `vpsraq` |
| 0x2d, 0x2e | `sol_avx2_sllv64x4`, `srlv64x4` | `..64x8` | per lane u64 shift, 0 above 63 | `vpsllvq`, `vpsrlvq` |
| 0x2f | `sol_avx2_mulu32x4` | `sol_avx512_mulu32x8` | `u32 x u32 -> u64` of the low halves | `vpmuludq` |
| 0x30 | `sol_avx512_mullo64x4` | `sol_avx512_mullo64x8` | low 64 bits of u64 products | `vpmullq` (DQ) |
| 0x31, 0x32 | `sol_avx512_madd52lo64x4`, `madd52hi64x4` | `..64x8` | `d += lo / hi 52 bits of lo52(a) * lo52(b)` | `vpmadd52luq` / `huq` (IFMA) |
| 0x33 | `sol_avx512_cmp256` | `sol_avx512_cmp512` | lane mask of `pred(a, b)` in `dst` / r0; `off`: predicate (vpcmp encoding), bit 3 unsigned, bits 4..5 lane size | `vpcmp*` + `kmovq` |
| 0x34 | `sol_avx512_blendm256` | `sol_avx512_blendm512` | `d = mask ? a : b`, mask in `src` / r2, lane size in `off` | `kmovq` + `vpblendm*` |
| 0x35 | `sol_avx512_compress256` | `sol_avx512_compress512` | masked lanes of `a` packed (rest 0), count in `dst` / r0 | `kmovq` + `vpcompress*` + `popcnt` (8 / 16 bit lanes: VBMI2) |
| 0x36 | `sol_avx512_expand256` | `sol_avx512_expand512` | low lanes of `a` spread to the mask bits (rest 0) | `kmovq` + `vpexpand*` |
| 0x37 | `sol_avx2_mont16x16` | `sol_avx512_mont16x32` | i16 Montgomery product `mulhi(a, w) - mulhi(mullo(a, w'), q)`, `w'` in slot `w + 2`, `q` in slot `off` | `vpmullw` + 2 `vpmulhw` + `vpsubw` |
| 0x38..0x3b | `sol_avx2_sll16x16`, `srl16x16`, `sra16x16`, `sra32x8` | `..x32`, `..x16` | shift by imm | `vpsllw`, `vpsrlw`, `vpsraw`, `vpsrad` |
| 0x3c..0x3e | `sol_avx512_sllv16x16`, `srlv16x16`, `srav16x16` | `..16x32` | per lane u16 shifts | `vpsllvw` / `vpsrlvw` / `vpsravw` (BW), reference call without |
| 0x3f, 0x40 | `sol_avx2_srav32x8`, `sol_avx512_srav64x4` | `..x16`, `..x8` | per lane arithmetic shift (sign fill when too large) | `vpsravd`; `vpsravq` / 4 AVX2 instructions |
| 0x41 | `sol_avx512_ror64x4` | `sol_avx512_ror64x8` | u64 rotate by imm | `vprorq` / 3 AVX2 instructions |
| 0x42, 0x43 | `sol_avx2_unpacklo256`, `unpackhi256` | `sol_avx512_unpacklo512`, `..hi512` | interleave per 128 bit lane, lane size in `off` | `vpunpckl*` / `vpunpckh*` |
| 0x44, 0x45 | `sol_avx2_packss256`, `packus256` | `..512` | saturating narrowing per 128 bit lane | `vpackss*` / `vpackus*` |
| 0x46, 0x47 | `sol_avx2_zext256`, `sext256` | `..512` | widen the low lanes (all six size pairs) | `vpmovzx*` / `vpmovsx*` |
| 0x48 | `sol_avx512_popcnt256` | `sol_avx512_popcnt512` | per lane population count | `vpopcnt*` (BITALG / VPOPCNTDQ) / nibble table + `vpmaddubsw` / `vpmaddwd` / `vpsadbw` |
| 0x49 | `sol_avx512_lzcnt256` | `sol_avx512_lzcnt512` | per lane leading zeros (32 / 64 bit) | `vplzcnt*` (CD), reference call without |
| 0x4a..0x4c | `sol_avx512_permb256`, `permi2b256`, `multishift256` | `..512` | byte permutations, bit field extraction | `vpermb`, `vpermi2b`, `vpmultishiftqb` (VBMI), reference call without |
| 0x4d, 0x4e | `sol_avx512_gf2p8affine256`, `gf2p8mul256` | `..512` | GF(2) affine map / GF(2^8) product per byte | `vgf2p8affineqb`, `vgf2p8mulb` (GFNI), reference call without |
| 0x4f..0x52 | *reserved* (removed AES rounds) | | | |
| 0x53 | `sol_avx512_clmul256` | `sol_avx512_clmul512` | carry-less 64 x 64 -> 128 per 128 bit lane | `vpclmulqdq` (VPCLMULQDQ, VEX forms too), reference call without |
| 0x54, 0x55 | `sol_avx2_min256`, `max256` | `..512` | per lane min / max; `off` = lane size \| 4 unsigned | `vpmin*` / `vpmax*`; u64 lanes: compare + `vpblendvb` without AVX-512 |
| 0x56 | `sol_avx2_abs256` | `sol_avx512_abs512` | per lane absolute value | `vpabs*`; u64: sign mask xor / sub without AVX-512 |
| 0x57, 0x58 | `sol_avx2_adds256`, `subs256` | `..512` | saturating add / sub, 8 / 16 bit, signed or unsigned | `vpadds*`, `vpaddus*`, `vpsubs*`, `vpsubus*` |
| 0x59 | `sol_avx2_avg256` | `sol_avx512_avg512` | unsigned rounding average, 8 / 16 bit | `vpavgb` / `vpavgw` |
| 0x5a, 0x5b | `sol_avx2_add8x32`, `sub8x32` | `..8x64` | wrapping u8 add / sub | `vpaddb`, `vpsubb` |
| 0x5c, 0x5d | `sol_avx2_mulhu16x16`, `mulhrs16x16` | `..16x32` | unsigned high product, rounded scaled product | `vpmulhuw`, `vpmulhrsw` |
| 0x5e | `sol_avx2_muls32x4` | `sol_avx512_muls32x8` | `i32 x i32 -> i64` of the low halves | `vpmuldq` |
| 0x5f | `sol_avx2_maddubs16x16` | `sol_avx512_maddubs16x32` | u8 x i8 pairs summed with saturation | `vpmaddubsw` |
| 0x60 | `sol_avx512_trunc256` | `sol_avx512_trunc512` | truncate the lanes into the low part (six size pairs) | `vpmov{wb,db,qb,dw,qw,qd}`; `vpshufb` + 128 bit unpack without AVX-512 (512 bit: reference call) |
| 0x61 | `sol_avx2_permq256` | `sol_avx512_permq512` | u64 permutation by imm per 256 bit half | `vpermq` |
| 0x62 | `sol_avx512_permw256` | `sol_avx512_permw512` | cross lane u16 permutation | `vpermw` (BW), reference call without |
| 0x63 | `sol_avx2_alignr256` | `sol_avx512_alignr512` | byte align per 128 bit lane | `vpalignr` |
| 0x64, 0x65 | `sol_avx2_shuflw256`, `shufhw256` | `..512` | low / high u16 shuffle per 128 bit lane | `vpshuflw` / `vpshufhw` |
| 0x66 | `sol_avx512_conflict256` | `sol_avx512_conflict512` | per lane mask of earlier equal lanes (32 / 64 bit) | `vpconflict*` (CD), reference call without |
| 0x67 | `sol_avx512_testm256` | `sol_avx512_testm512` | lane mask of `a & b != 0` (`== 0`) in `dst` / r0 | `vptestm*` / `vptestnm*` + `kmovq`; and + compare + movemask without AVX-512 |
| 0x68, 0x69 | `sol_avx512_dpbusd256`, `dpwssd256` | `..512` | u8 x i8 / i16 x i16 dot products accumulated into i32 | `vpdpbusd` / `vpdpwssd` (VNNI); `vpmaddwd` + `vpaddd` for dpwssd without |
| 0x6a | `sol_avx2_sadbw256` | `sol_avx512_sadbw512` | sum of absolute byte differences per u64 | `vpsadbw` |
| 0x6b, 0x6c | `sol_avx512_shld256`, `shrd256` | `..512` | funnel shift by imm, 16 / 32 / 64 bit | `vpshld*` / `vpshrd*` (VBMI2); two shifts + or without |
| 0x6d, 0x6e | `sol_avx512_shldv256`, `shrdv256` | `..512` | funnel shift by lane into the accumulator | `vpshldv*` / `vpshrdv*` (VBMI2); 5 instructions for 32 / 64 bit without |
| 0x6f, 0x70 | `sol_avx512_loadm256`, `storem256` | `..512` | masked load / store, lane size in `vb`, mask in the `dst` register / r3 | one call: translation of the span of the set lanes + copy |
| 0x27 | *reserved* (removed fused compare + compress, now `cmp` + `compress`) | | | |

The BLAKE3 work needed the first 13; the rest came with the Falcon verifier (16 bit NTT arithmetic,
Keccak on 64 bit lanes, vectorized rejection sampling), the private vector file (lane moves) and the
completion of the core set (below).

Masks are scalar bitmasks (bit `i` = lane `i`) passed in `src` / r2 and returned in `dst` / r0, so mask
logic is ordinary scalar ALU code and there is no mask register file in the VM.

**Lowering tiers.** The JIT picks per operation and per host feature (AVX-512 F / BW / VL, DQ,
IFMA and VBMI2 are detected separately):

1. AVX-512: one instruction or a short sequence (`kmovq` + `vpcompressb` + `popcnt`, ...).
2. AVX2 templates for the AVX-512 operations, written after LLVM's lowerings or better known idioms:
   `sra64` (`vpsrad` / `vpsrlq` / `vpshufd` + `vpblendd`, 3 to 4 instructions), `mullo64` (3
   `vpmuludq`, 8 instructions), `madd52lo/hi` (26 bit limbs, 4 `vpmuludq`, about 16 instructions),
   `cmp` (`vpcmpgt` / `vpcmpeq`, sign flips for unsigned, `vpmovmskb` / `vmovmskps` / `vmovmskpd`,
   `vpacksswb` + `vpermq` for 16 bit lanes), `blendm` (mask expanded to a vector, `vpblendvb`), u32 /
   u64 `compress` / `expand` (`vpermd` with indices from a table indexed by the mask, no BMI2 since
   `pext` is microcoded on Zen 1 / 2). Further templates for the remaining extensions:
   `lzcnt` (bit smearing + the `popcnt` template), `permb` / `permi2b` / `permw` (byte table lookups:
   every 128 bit table lane broadcast and indexed with `vpshufb`, selected by the higher index bits
   with a `vpblendvb` tree), u16 variable shifts and funnel shifts (u32 variable shifts on the even
   and odd words), exact `dpbusd` (bytes widened to words, two `vpmaddwd`), `conflict` of u32 / u64
   lanes (compares with the lane rotations), `gf2p8mul` (8 shift / xor steps), `gf2p8affine` (per
   result bit a matrix byte broadcast with `vpshufb`, parity from a nibble table), `multishift`
   (8 variable u64 rotations), 512 bit `trunc` (two truncated halves joined) and `testm` (two masks
   joined). 512 bit forms otherwise run as two halves.
3. A branchless scalar loop over the vector file for `compress` and `expand` without a native or
   table based lowering (8 / 16 bit lanes, 512 bit forms).
4. A host call to the reference implementation (`intrinsic_slow_paths` in the statistics): only
   512 bit `conflict` without AVX-512CD (`avx2_templates_cover_the_avx512_operations` checks this).

Every tier must give the reference results bit for bit (integer operations only, semantics fixed
per operation, never per host), and CUs never depend on the tier. `Config::simd_force_reference`
runs every operation through tier 4 and `Config::simd_avx512 = false` restricts the JIT to AVX2,
so the `tiers_agree` test compares all tiers of a host on random and edge value inputs (0, 1, -1,
the extremes of every lane width, shift counts around 31 / 32 / 63 / 64).

The `exhaustive_*`, `clmul_products` and `masked_load_store` tests run each operation over a stream of records in one loop
and check the interpreter against an independent model and every tier against the interpreter:
every byte pair (`gf2p8mul`), every byte / index / control value (`gf2p8affine`, `permb`,
`permi2b`, `multishift`, `popcnt` of bytes), every u16 (`popcnt`, the 16 -> 8 packs, the
widenings), every immediate shift count, variable counts 0..=70 and huge ones, every leading zero
count, every mask of u32 / u64 `compress` / `expand` / `blendm` (both widths), every `cmp` lane size x
predicate x signedness, every byte pair for the 8 bit min / max / abs / saturating / average / add /
sub / sadbw / maddubs operations, every immediate of `permq` / `shuflw` / `shufhw` / `alignr` / funnel
shifts, every u16 index of `permw`, and masked loads / stores at region ends. These tests found a
reference interpreter bug (signed `min` / `max` compared sign extended lanes as unsigned): the JIT
was right.

Rosetta 2 gets `vpsllvq` / `vpsrlvq` wrong for counts above 63 (it shifts by the low bits instead of
returning 0). x86-64 macOS builds of the JIT detect Rosetta and zero those lanes explicitly (three
more instructions per u64 variable shift: `sllv64`, `srlv64`, `srav64`, the u64 funnel shifts and
`multishift`), so the tests run every count there too. Other targets don't contain the workaround,
so the code for real hardware is unchanged.

**Robustness** (`tests/simd_robustness.rs`). Each test runs the interpreter and every JIT tier of the
host (cached, direct, reference, with heavy code diversification, without value sanitization, and on
AVX-512 hosts AVX2-only plus every CPU feature profile of `simd::features::PROFILES` through
`Config::simd_disabled_features`) and compares result, CUs, guest memory and the vector file:

* *Out of bounds:* every operation at the boundary slots (0, the limits of the 6 / 7 bit fields,
  each of the last 32 slots) in every operand role, with constant operands (lowered inline), run
  time operands (the generic path) and malformed ones (flipped and non-canonical bits). The vector
  file lies between guard pages (`harness::GuardedVectorFile`), so an access outside of it faults,
  and canaries catch writes into the slack of a page. Bulk / masked transfers with every slot count
  up to the end and beyond, transfers at the edges of guest memory regions with every mask pattern.
* *Divergence:* every operation with every aliasing of its operands, each with the operands in
  memory, cached and dirty, or evicted; every scalar register (including `dst == src`) for the
  operations with scalar operands; every valid scalar instruction between the operand setup of a
  call and the call (the JIT's constant tracking of r1) and between two vector operations (R11
  tracking, the cache flushes, jumps which start a block); faults (access violations, division by zero, the instruction
  meter at checkpoints and at `lddw`) in the middle of vector code leave the vector file written
  back.
* *Alignment:* misaligned vector files (offsets 1, 8, 16, 31; under Rosetta 2 an aligned load planted
  in the JIT went unnoticed, so this is only checked on real hardware), and in debug builds the JIT checks
  RSP before every call into Rust (`test rsp, 15; jz; ud2`), so run the suites in debug builds too.
* The tests must notice planted bugs: an off by one slot count, broken templates, a missing write
  back or flush, an aligned load from the vector file, the stack parity bug all make them fail.

## Intrinsic call form (SIMD-0645 style, no new opcode)

The operations are also available without extending the ISA, as static syscalls in the style of
`sol_multi3` (SIMD-0645). This is `VecMode::Call` in the assembler, `src/simd/intrinsics.rs`, and the
form used by compiled Rust programs (`simd-falcon/`).

```text
mov64 r1, <packed operands>       ; compile time constant
call  sol_avx2_add32x8            ; CALL_IMM, src = 0, imm = murmur32("sol_avx2_add32x8")
```

* **Encoding:** one static call identifier per operation and width (211 of them, the names in the
  table above). `dst`, `src` and `off` of the `CALL_IMM` stay zero, so the bytecode is plain eBPF.
  The names say which ISA level each operation maps to: `sol_avx_` (moves), `sol_avx2_` (AVX2
  integer operations) and `sol_avx512_` (AVX-512 operations, and every 512 bit form). Every
  operation runs on every host: the JIT lowers the AVX-512 ones to AVX2 sequences when the host
  lacks AVX-512, with identical results.
* **ABI:** `r1` holds the packed operands, `r2` the scalar operand (load/store address, bcast / ins64
  value), `r0` receives the scalar result (`ext64`, `movmsk8`) or 0. A Rust wrapper
  `sol_avx2_add32x8(ops(V0, V0, V1))` compiles to exactly `mov64 r1, imm` + `call`, and compilers
  reload `r1` before every call because it is caller saved.
* **Packing:** 8 bit slot numbers (256 slots): `vd | va << 8 | vb << 16`, an imm8 in bits 16..24 for
  one-source ops, a 16 bit offset in bits 8..24 and the slot count - 1 in bits 24..29 for
  load/store, the lane in bits 8..10 for ins64/ext64. `blendd`, `blendw` and `ternlog` use 7 bit
  slots (`vd | va << 7 | vb << 14 | imm8 << 21`). Every value is below `2^29`.
* **Results:** every other register unchanged, 1 CU for the `CALL_IMM`, so each vector op costs 2
  CUs including the `mov64`, at either width.
* **Execution:** every engine runs the reference semantics. The JIT's `codegen` hook checks a block
  local constant tracker. If `r1` holds a valid constant it emits the code of the opcode form,
  register caching included. Otherwise it emits a regular syscall to the reference implementation.
  Results and CUs are identical either way.
* **Loads from .rodata:** the tracker also follows `lddw` and `add64 imm`. A `load256` whose
  address is a known constant inside the read-only section (a twiddle table, say) is translated at
  compile time: the section is immutable and always mapped at its address, so the JIT emits plain
  `vmovdqu`s from the host address into cached registers, with no runtime translation and no
  register cache flush.
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

## Vector file

* 256 slots x 32 bytes = 8 KiB of **private VM state**, like the scalar registers. It is not
  mapped into the guest address space (the former region at `0x6 << 32` is gone): guest loads and
  stores can't reach it, and nothing about it is part of the memory ABI.
* Data moves in and out only through the vector operations: `load256` / `store256` copy up to 32
  consecutive slots from / to guest memory with **one** translation, `ins64` / `ext64` move 64 bit
  lanes from / to scalar registers, the broadcasts and `movmsk8` produce vectors and masks.
* The VM allocates the file the first time it runs a program with vector instructions
  (`SimdAnalysis::vector_instructions`) and zeroes it on every invocation. An embedder may attach
  its own buffer with `EbpfVm::set_vector_memory` instead (the test harness does, to inspect the
  slots).
* The opcode form addresses slots 0..64 (6 bit fields), the intrinsic call form all 256.

Why private: a mapped region costs an entry in every memory mapping, makes every guest pointer a
potential alias of the vector registers (the JIT had to write back and drop its register cache
around every scalar load and store), and fixes a host buffer layout in the ABI. The CPU itself
moves data between general purpose and vector registers with `vmovq` / `vpinsrq` / `vpextrq` /
`vpbroadcastq` for single values and through memory for bulk data, which is exactly the split here.
Measured on the same Falcon verifier (the algorithm before the 512 bit rework), mapped region vs
private file: CUs +0.8% (buffers instead of scalar stores into slots), interpreter 13% faster
(native arm64, 617 vs 539 µs), JIT about 6% slower (x86 under Rosetta, 110 vs 117 µs): every
load/store is a host call which flushes the register cache, where the region let scalar code write
into slots directly.

## Interpreter (reference semantics)

`simd::interpret` is the reference. Lanes are little-endian, with Rust wrapping arithmetic and
rotates.

**Translation:** load/store translate their guest range once through
`MemoryMapping::map_with_access_violation_handler`, then copy. The JIT calls the same function
(`jit_transfer`, an `extern "C"` shim), so errors are identical in both engines.

**Read-before-write:** all sources are read before the destination is written, so overlapping and
aliasing operands (including slot pairs) are well defined.

## Metering

Every vector instruction costs **1 CU**, the same as any other sBPF instruction, at either width
and for any slot count of load/store. It's provisional,
but explicit and identical in both engines. The tests assert that interpreter and JIT CU counts are
equal for every program, including runs that exhaust the meter in the middle of vector code. CU
counts aren't used as evidence of speed.

## JIT (AVX2 and AVX-512)

### Encoding

`x86::VexInstruction` (2/3-byte VEX) and `x86::EvexInstruction` (EVEX, registers up to 31, opmasks,
compressed disp8) encoders, with register, `[base+disp]` and RIP-relative operands. Their output is
checked against `llvm-mc` in a unit test. Constants (shuffle masks, the opmask tables of the 512 bit
blends) are emitted once into the text section after the subroutines and read RIP-relative.

### Host features

`JitCompiler::new` returns `EbpfError::SimdHostUnsupported` if the program contains vector ops and
the host has no AVX2. `Config::simd_assume_avx2` skips the check (only for Rosetta 2, which executes
AVX2 but hides it from CPUID). With `Config::simd_avx512` (default on) the JIT uses AVX-512 when
the host has AVX-512 F, BW and VL; `simd_force_avx512` generates AVX-512 code on any host for
inspection. The semantics never depend on the host, only the instructions do.

### Strategy A: direct (`SimdJitStrategy::Direct`)

The register cache below, written back and emptied after every vector instruction: every operand
comes from the vector file in memory and every result goes back to it.

### Strategy B: basic-block-local register cache (`SimdJitStrategy::RegisterCached`)

`SimdJitState` tracks `{reg, dirty}` per slot over 15 YMM registers (`ymm15` is scratch), with LRU
eviction. With AVX-512 a second cache holds **slot pairs** in `zmm16..zmm30` (`zmm31` is scratch)
for the 512 bit operations; a slot is cached at one width only, an operation of the other width
writes it back and drops it first. Without AVX-512, 512 bit operations run as two 256 bit halves in
the YMM cache (the cross half permutations with temporaries, so operands may alias).

* **Block boundaries.**
  - Block starts are jump targets, internal call targets and the entry point, from a pre-scan.
  - At a block start the caches are written back and emptied *before* the pc offset is recorded.
    The fall-through path pays for the write-back, and jumps arrive with empty caches.
  - Every branch, call, exit, syscall and scalar memory access (all of which call host code that
    clobbers vector registers) writes back dirty slots and invalidates first. The same goes for any
    other instruction that may throw, such as division.
  - Pure scalar ALU instructions leave the caches alone.
* **load / store** call the host once (translation + copy); the caches are flushed and invalidated
  before them. Loads from `.rodata` at known addresses go straight into cached registers.
* **Exceptional exits.**
  - Meter checkpoints write back dirty slots but keep the registers.
  - Memory faults and calls are preceded by a full flush.
  - So the vector file is up to date at every point where the program can stop.
* **Sources.** A source that isn't cached is used directly as a memory operand
  (`vpaddd ymm, ymm, [r11+disp]`, full vector EVEX memory operands for zmm) and doesn't take a
  register.
* **Transposes.** They flush, then transpose in registers and leave the results resident and dirty.
* **AVX-512 single instructions.** With AVX-512 the 256 bit `ternlog`, `rolv32`, `rolv64`, `ror32`
  and `permi2d` also become one EVEX instruction on the YMM registers.
* **`callx`.** It may not enter a run of vector code (see above), so the caches stay on for programs
  with `callx`.
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
| `src/simd/mod.rs` | encoding, decoding, reference semantics, load/store, `VectorMemory` |
| `src/simd_jit.rs` | AVX2 / AVX-512 lowering, register caches, intrinsic constant tracking, statistics (child module of `jit`) |
| `src/simd/intrinsics.rs` | static call (SIMD-0645 style) form: identifiers, operand packing, builtins |
| `src/x86.rs` | `VexInstruction` and `EvexInstruction` encoders |
| `src/simd/asm.rs` | macro assembler for the generated programs |
| `src/simd/blake3_program.rs` | BLAKE3 hash / compression / microbench program generators |
| `src/simd/harness.rs` | self contained runner used by tests and benches |
| `tests/simd.rs` | correctness tests |
| `benches/simd_blake3.rs` | benchmark harness (writes JSON) |
| `docs/simd-benchmarks.md` | results |
