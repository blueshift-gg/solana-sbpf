//! The vector operations as static calls (SIMD-0645 style JIT intrinsics)
//!
//! Instead of a new opcode, every vector operation is a static syscall: `CALL_IMM` with `src = 0`,
//! `dst = 0`, `off = 0` and `imm = murmur32("sol_<isa>_<op>")` (see [NAMES_256], [NAMES_512]).
//! The bytecode stays within the BPF ISA, and a runtime which doesn't know the names rejects the
//! program at verification (`InvalidSyscall`).
//!
//! ABI (chosen so that compiled code needs nothing but the call itself; a Rust wrapper
//! `sol_avx2_add32x8(ops(V0, V0, V1))` compiles to `mov64 r1, imm` + `call`):
//! - `r1` holds the packed operands (see [pack]), always a compile time constant in practice.
//! - `r2` holds the scalar operand: the address of `load256`/`store256`, the value of `bcast` /
//!   `ins64`.
//! - The operation writes its scalar result (`ext64`, `movmsk8`, `compresslt16`) or 0 to `r0` and leaves every
//!   other register unchanged.
//! - It costs the 1 CU of its `CALL_IMM` instruction and nothing else.
//!
//! Every execution backend runs the operation through [interpret]. The JIT lowers a call to inline
//! AVX2, exactly like the opcode form, when it can prove that `r1` holds a valid constant at the
//! call (block local constant tracking). Otherwise it emits an ordinary syscall to the same
//! implementation the interpreter uses.

use super::*;
use crate::{
    elf::ElfError,
    program::{BuiltinFunctionDefinition, BuiltinProgram, JitCompiler},
    vm::{ContextObject, EncryptedHostAddressToEbpfVm},
};

/// Function names of the 256 bit forms, indexed by sub-operation. The prefix names the ISA level
/// which has the operation as (essentially) one instruction: `sol_avx_` (AVX: 256 bit moves and 64
/// bit lane moves), `sol_avx2_` (AVX2 integer operations), `sol_avx512_` (AVX-512VL: rotates,
/// three input logic, two table permutation). Every engine runs every operation, the JIT lowers the
/// AVX-512 ones to short AVX2 sequences on hosts without AVX-512.
pub const NAMES_256: [Option<&str>; SUBOP_COUNT as usize] = [
    Some("sol_avx_load256"),
    Some("sol_avx_store256"),
    Some("sol_avx2_add32x8"),
    Some("sol_avx2_xor256"),
    Some("sol_avx2_and256"),
    Some("sol_avx2_or256"),
    Some("sol_avx512_ror32x8"),
    Some("sol_avx2_shufd256"),
    Some("sol_avx2_transpose8x8"),
    Some("sol_avx2_bcast32x8"),
    Some("sol_avx_mov256"),
    Some("sol_avx2_permd256"),
    Some("sol_avx2_blendd256"),
    Some("sol_avx2_sub32x8"),
    Some("sol_avx2_mullo32x8"),
    Some("sol_avx2_minu32x8"),
    Some("sol_avx2_srl32x8"),
    Some("sol_avx2_sll32x8"),
    Some("sol_avx2_sllv32x8"),
    Some("sol_avx2_srlv32x8"),
    Some("sol_avx2_andn256"),
    Some("sol_avx512_rolv32x8"),
    Some("sol_avx512_ternlog256"),
    Some("sol_avx2_add16x16"),
    Some("sol_avx2_sub16x16"),
    Some("sol_avx2_mullo16x16"),
    Some("sol_avx2_mulhi16x16"),
    Some("sol_avx2_minu16x16"),
    Some("sol_avx2_madd16x16"),
    Some("sol_avx2_blendw256"),
    Some("sol_avx_ins64"),
    Some("sol_avx_ext64"),
    Some("sol_avx2_bcast64x4"),
    Some("sol_avx2_movmsk8x32"),
    Some("sol_avx512_permi2d256"),
    Some("sol_avx512_rolv64x4"),
    Some("sol_avx2_transpose4x4q"),
    Some("sol_avx2_shufb256"),
    Some("sol_avx512_permt2d256"),
    None,
    Some("sol_avx2_add64x4"),
    Some("sol_avx2_sub64x4"),
    Some("sol_avx2_sll64x4"),
    Some("sol_avx2_srl64x4"),
    Some("sol_avx512_sra64x4"),
    Some("sol_avx2_sllv64x4"),
    Some("sol_avx2_srlv64x4"),
    Some("sol_avx2_mulu32x4"),
    Some("sol_avx512_mullo64x4"),
    Some("sol_avx512_madd52lo64x4"),
    Some("sol_avx512_madd52hi64x4"),
    Some("sol_avx512_cmp256"),
    Some("sol_avx512_blendm256"),
    Some("sol_avx512_compress256"),
    Some("sol_avx512_expand256"),
    Some("sol_avx2_mont16x16"),
    Some("sol_avx2_sll16x16"),
    Some("sol_avx2_srl16x16"),
    Some("sol_avx2_sra16x16"),
    Some("sol_avx2_sra32x8"),
    Some("sol_avx512_sllv16x16"),
    Some("sol_avx512_srlv16x16"),
    Some("sol_avx512_srav16x16"),
    Some("sol_avx2_srav32x8"),
    Some("sol_avx512_srav64x4"),
    Some("sol_avx512_ror64x4"),
    Some("sol_avx2_unpacklo256"),
    Some("sol_avx2_unpackhi256"),
    Some("sol_avx2_packss256"),
    Some("sol_avx2_packus256"),
    Some("sol_avx2_zext256"),
    Some("sol_avx2_sext256"),
    Some("sol_avx512_popcnt256"),
    Some("sol_avx512_lzcnt256"),
    Some("sol_avx512_permb256"),
    Some("sol_avx512_permi2b256"),
    Some("sol_avx512_multishift256"),
    Some("sol_avx512_gf2p8affine256"),
    Some("sol_avx512_gf2p8mul256"),
    None,
    None,
    None,
    None,
    Some("sol_avx512_clmul256"),
    Some("sol_avx2_min256"),
    Some("sol_avx2_max256"),
    Some("sol_avx2_abs256"),
    Some("sol_avx2_adds256"),
    Some("sol_avx2_subs256"),
    Some("sol_avx2_avg256"),
    Some("sol_avx2_add8x32"),
    Some("sol_avx2_sub8x32"),
    Some("sol_avx2_mulhu16x16"),
    Some("sol_avx2_mulhrs16x16"),
    Some("sol_avx2_muls32x4"),
    Some("sol_avx2_maddubs16x16"),
    Some("sol_avx512_trunc256"),
    Some("sol_avx2_permq256"),
    Some("sol_avx512_permw256"),
    Some("sol_avx2_alignr256"),
    Some("sol_avx2_shuflw256"),
    Some("sol_avx2_shufhw256"),
    Some("sol_avx512_conflict256"),
    Some("sol_avx512_testm256"),
    Some("sol_avx512_dpbusd256"),
    Some("sol_avx512_dpwssd256"),
    Some("sol_avx2_sadbw256"),
    Some("sol_avx512_shld256"),
    Some("sol_avx512_shrd256"),
    Some("sol_avx512_shldv256"),
    Some("sol_avx512_shrdv256"),
    Some("sol_avx512_loadm256"),
    Some("sol_avx512_storem256"),
];

/// Function names of the 512 bit forms (slot pairs, AVX-512F/BW), indexed by sub-operation
pub const NAMES_512: [Option<&str>; SUBOP_COUNT as usize] = [
    None,
    None,
    Some("sol_avx512_add32x16"),
    Some("sol_avx512_xor512"),
    Some("sol_avx512_and512"),
    Some("sol_avx512_or512"),
    Some("sol_avx512_ror32x16"),
    Some("sol_avx512_shufd512"),
    None,
    Some("sol_avx512_bcast32x16"),
    Some("sol_avx512_mov512"),
    Some("sol_avx512_permd512"),
    Some("sol_avx512_blendd512"),
    Some("sol_avx512_sub32x16"),
    Some("sol_avx512_mullo32x16"),
    Some("sol_avx512_minu32x16"),
    Some("sol_avx512_srl32x16"),
    Some("sol_avx512_sll32x16"),
    Some("sol_avx512_sllv32x16"),
    Some("sol_avx512_srlv32x16"),
    Some("sol_avx512_andn512"),
    Some("sol_avx512_rolv32x16"),
    Some("sol_avx512_ternlog512"),
    Some("sol_avx512_add16x32"),
    Some("sol_avx512_sub16x32"),
    Some("sol_avx512_mullo16x32"),
    Some("sol_avx512_mulhi16x32"),
    Some("sol_avx512_minu16x32"),
    Some("sol_avx512_madd16x32"),
    Some("sol_avx512_blendw512"),
    None,
    None,
    Some("sol_avx512_bcast64x8"),
    Some("sol_avx512_movmsk8x64"),
    Some("sol_avx512_permi2d512"),
    Some("sol_avx512_rolv64x8"),
    Some("sol_avx512_transpose8x8q"),
    Some("sol_avx512_shufb512"),
    Some("sol_avx512_permt2d512"),
    None,
    Some("sol_avx512_add64x8"),
    Some("sol_avx512_sub64x8"),
    Some("sol_avx512_sll64x8"),
    Some("sol_avx512_srl64x8"),
    Some("sol_avx512_sra64x8"),
    Some("sol_avx512_sllv64x8"),
    Some("sol_avx512_srlv64x8"),
    Some("sol_avx512_mulu32x8"),
    Some("sol_avx512_mullo64x8"),
    Some("sol_avx512_madd52lo64x8"),
    Some("sol_avx512_madd52hi64x8"),
    Some("sol_avx512_cmp512"),
    Some("sol_avx512_blendm512"),
    Some("sol_avx512_compress512"),
    Some("sol_avx512_expand512"),
    Some("sol_avx512_mont16x32"),
    Some("sol_avx512_sll16x32"),
    Some("sol_avx512_srl16x32"),
    Some("sol_avx512_sra16x32"),
    Some("sol_avx512_sra32x16"),
    Some("sol_avx512_sllv16x32"),
    Some("sol_avx512_srlv16x32"),
    Some("sol_avx512_srav16x32"),
    Some("sol_avx512_srav32x16"),
    Some("sol_avx512_srav64x8"),
    Some("sol_avx512_ror64x8"),
    Some("sol_avx512_unpacklo512"),
    Some("sol_avx512_unpackhi512"),
    Some("sol_avx512_packss512"),
    Some("sol_avx512_packus512"),
    Some("sol_avx512_zext512"),
    Some("sol_avx512_sext512"),
    Some("sol_avx512_popcnt512"),
    Some("sol_avx512_lzcnt512"),
    Some("sol_avx512_permb512"),
    Some("sol_avx512_permi2b512"),
    Some("sol_avx512_multishift512"),
    Some("sol_avx512_gf2p8affine512"),
    Some("sol_avx512_gf2p8mul512"),
    None,
    None,
    None,
    None,
    Some("sol_avx512_clmul512"),
    Some("sol_avx512_min512"),
    Some("sol_avx512_max512"),
    Some("sol_avx512_abs512"),
    Some("sol_avx512_adds512"),
    Some("sol_avx512_subs512"),
    Some("sol_avx512_avg512"),
    Some("sol_avx512_add8x64"),
    Some("sol_avx512_sub8x64"),
    Some("sol_avx512_mulhu16x32"),
    Some("sol_avx512_mulhrs16x32"),
    Some("sol_avx512_muls32x8"),
    Some("sol_avx512_maddubs16x32"),
    Some("sol_avx512_trunc512"),
    Some("sol_avx512_permq512"),
    Some("sol_avx512_permw512"),
    Some("sol_avx512_alignr512"),
    Some("sol_avx512_shuflw512"),
    Some("sol_avx512_shufhw512"),
    Some("sol_avx512_conflict512"),
    Some("sol_avx512_testm512"),
    Some("sol_avx512_dpbusd512"),
    Some("sol_avx512_dpwssd512"),
    Some("sol_avx512_sadbw512"),
    Some("sol_avx512_shld512"),
    Some("sol_avx512_shrd512"),
    Some("sol_avx512_shldv512"),
    Some("sol_avx512_shrdv512"),
    Some("sol_avx512_loadm512"),
    Some("sol_avx512_storem512"),
];

/// Register holding the packed operands
pub const OPERAND_REG: u8 = 1;
/// Register holding the scalar operand
pub const SCALAR_REG: u8 = 2;
/// Register holding the mask of masked load / store
pub const MASK_REG: u8 = 3;

/// Function name of a sub-operation
pub fn name(subop: u8, wide: bool) -> Option<&'static str> {
    if wide {
        NAMES_512.get(subop as usize).copied().flatten()
    } else {
        NAMES_256.get(subop as usize).copied().flatten()
    }
}

/// Static call identifier of a sub-operation
pub fn hash(subop: u8, wide: bool) -> u32 {
    ebpf::hash_symbol_name(name(subop, wide).expect("no such intrinsic").as_bytes())
}

/// `(identifier, sub-operation, wide)` of every intrinsic
pub fn hashes() -> Vec<(u32, u8, bool)> {
    let mut hashes = Vec::new();
    for subop in 0..SUBOP_COUNT {
        for wide in [false, true] {
            if name(subop, wide).is_some() {
                hashes.push((hash(subop, wide), subop, wide));
            }
        }
    }
    hashes
}

/// Packs the operands of `op` (scalar operand in `r2`, scalar result in `r0`) into the `r1` value.
///
/// Slots are 8 bit (256 slots). The layout depends on the operation, and every packed value is
/// below `2^29`, so it always fits a `mov64 r1, imm32`:
///
/// ```text
/// load256 / store256     bits 0..7 first slot   bits 8..23 signed 16 bit offset   bits 24..28 slots - 1
/// bcast / transpose      bits 0..7 vd
/// ins64                  bits 0..7 vd   bits 8..9 lane
/// ext64                  bits 0..7 va   bits 8..9 lane
/// movmsk8                bits 0..7 va
/// two sources            bits 0..7 vd   bits 8..15 va   bits 16..23 vb   (also permi2d)
/// one source + imm8      bits 0..7 vd   bits 8..15 va   bits 16..23 imm8   (ror, shufd, shifts)
/// cmp                    bits 0..7 va   bits 8..15 vb   bits 16..23 imm8 (predicate, signedness, lane size)
/// blendm                 bits 0..7 vd   bits 8..15 va   bits 16..23 vb   bits 24..25 lane size (mask in r2)
/// compress / expand      bits 0..7 vd   bits 8..15 va   bits 16..17 lane size (mask in r2)
/// loadm / storem         bits 0..7 slot   bits 8..23 offset   bits 24..25 lane size (address r2, mask r3)
/// testm                  bits 0..7 va   bits 8..15 vb   bits 16..23 imm8
/// min / max / adds / subs / avg / alignr / shld / shrd / shldv / shrdv: 7 bit slots + imm8 as blendd
/// mov                    bits 0..7 vd   bits 8..15 va
/// blendd / blendw /      bits 0..6 vd   bits 7..13 va   bits 14..20 vb   bits 21..28 imm8 (slots < 128)
/// ternlog
/// mont16                 bits 0..6 vd   bits 7..13 va   bits 14..20 w   bits 21..27 q   (slots < 128)
/// unpack / pack /        bits 0..6 vd   bits 7..13 va   bits 14..20 vb   bits 21..28 imm8 (slots < 128)
/// gf2p8affine / clmul
/// ```
pub fn pack(op: &VecInsn) -> u64 {
    let (vd, va, vb, imm) = (
        op.vd as u64,
        op.va as u64,
        op.vb as u64,
        op.off as u16 as u64,
    );
    match op.subop {
        VLOAD => vd | imm << 8 | vb << 24,
        VSTORE => va | imm << 8 | vb << 24,
        VBCAST32 | VBCAST64 | VTRANSPOSE8 | VTRANSPOSE64 => vd,
        VINS64 => vd | imm << 8,
        VEXT64 => va | imm << 8,
        VMOVMSK8 => va,
        VROR32 | VSHUFD | VSRL32 | VSLL32 | VSLL64 | VSRL64 | VSRA64 | VSLL16 | VSRL16 | VSRA16
        | VSRA32 | VROR64 | VPOPCNT | VLZCNT | VZEXT | VSEXT | VABS | VTRUNC | VPERMQ | VSHUFLW
        | VSHUFHW | VCONFLICT => vd | va << 8 | (imm & 0xff) << 16,
        VCMP | VTESTM => va | vb << 8 | (imm & 0xff) << 16,
        VLOADM => vd | imm << 8 | vb << 24,
        VSTOREM => va | imm << 8 | vb << 24,
        VUNPCKLO | VUNPCKHI | VPACKSS | VPACKUS | VGF2P8AFFINE | VCLMUL | VMIN | VMAX | VADDS
        | VSUBS | VAVG | VALIGNR | VSHLD | VSHRD | VSHLDV | VSHRDV => {
            (vd & 0x7f) | (va & 0x7f) << 7 | (vb & 0x7f) << 14 | (imm & 0xff) << 21
        }
        VBLENDM => vd | va << 8 | vb << 16 | (imm & 3) << 24,
        VCOMPRESS | VEXPAND => vd | va << 8 | (imm & 3) << 16,
        VBLENDD | VBLENDW | VTERNLOG => {
            (vd & 0x7f) | (va & 0x7f) << 7 | (vb & 0x7f) << 14 | (imm & 0xff) << 21
        }
        VMONT16 => (vd & 0x7f) | (va & 0x7f) << 7 | (vb & 0x7f) << 14 | (imm & 0x7f) << 21,
        VMOV => vd | va << 8,
        _ => vd | va << 8 | vb << 16,
    }
}

/// Decodes the packed operands, `None` if they are malformed (including any non canonical bits)
pub fn unpack(subop: u8, wide: bool, packed: u64) -> Option<VecInsn> {
    if is_reserved(subop) || packed >> 29 != 0 {
        return None;
    }
    let byte = |shift: u32| ((packed >> shift) & 0xff) as u8;
    let (mut vd, mut va, mut vb, mut off) = (0u8, 0u8, 0u8, 0i16);
    match subop {
        VLOAD => (vd, off, vb) = (byte(0), ((packed >> 8) & 0xffff) as u16 as i16, byte(24)),
        VSTORE => (va, off, vb) = (byte(0), ((packed >> 8) & 0xffff) as u16 as i16, byte(24)),
        VBCAST32 | VBCAST64 | VTRANSPOSE8 | VTRANSPOSE64 => vd = byte(0),
        VINS64 => (vd, off) = (byte(0), byte(8) as i16),
        VEXT64 => (va, off) = (byte(0), byte(8) as i16),
        VMOVMSK8 => va = byte(0),
        VROR32 | VSHUFD | VSRL32 | VSLL32 | VSLL64 | VSRL64 | VSRA64 | VSLL16 | VSRL16 | VSRA16
        | VSRA32 | VROR64 | VPOPCNT | VLZCNT | VZEXT | VSEXT | VABS | VTRUNC | VPERMQ | VSHUFLW
        | VSHUFHW | VCONFLICT => (vd, va, off) = (byte(0), byte(8), byte(16) as i16),
        VCMP | VTESTM => (va, vb, off) = (byte(0), byte(8), byte(16) as i16),
        VLOADM => (vd, off, vb) = (byte(0), ((packed >> 8) & 0xffff) as u16 as i16, byte(24)),
        VSTOREM => (va, off, vb) = (byte(0), ((packed >> 8) & 0xffff) as u16 as i16, byte(24)),
        VUNPCKLO | VUNPCKHI | VPACKSS | VPACKUS | VGF2P8AFFINE | VCLMUL | VMIN | VMAX | VADDS
        | VSUBS | VAVG | VALIGNR | VSHLD | VSHRD | VSHLDV | VSHRDV => {
            let field = |shift: u32| ((packed >> shift) & 0x7f) as u8;
            (vd, va, vb, off) = (field(0), field(7), field(14), byte(21) as i16);
        }
        VBLENDM => (vd, va, vb, off) = (byte(0), byte(8), byte(16), byte(24) as i16),
        VCOMPRESS | VEXPAND => (vd, va, off) = (byte(0), byte(8), byte(16) as i16),
        VBLENDD | VBLENDW | VTERNLOG => {
            let field = |shift: u32| ((packed >> shift) & 0x7f) as u8;
            (vd, va, vb, off) = (field(0), field(7), field(14), byte(21) as i16);
        }
        VMONT16 => {
            let field = |shift: u32| ((packed >> shift) & 0x7f) as u8;
            (vd, va, vb, off) = (field(0), field(7), field(14), field(21) as i16);
        }
        VMOV => (vd, va) = (byte(0), byte(8)),
        _ => (vd, va, vb) = (byte(0), byte(8), byte(16)),
    }
    let src = if uses_src(subop) { SCALAR_REG } else { 0 };
    // Masked load / store take the mask in r3
    let dst = if reads_mask_register(subop) {
        MASK_REG
    } else {
        0
    };
    let op = VecInsn {
        subop,
        vd,
        va,
        vb,
        wide,
        dst,
        src,
        off,
    };
    // The opcode form's validation (operand ranges), applied without its 6 bit slot encoding
    validate_slot_op(&op)?;
    (pack(&op) == packed).then_some(op)
}

/// Executes one call with the argument registers `r1..=r5`, returns `r0`
pub fn execute(
    subop: u8,
    wide: bool,
    args: [u64; 5],
    vector_base: *mut u8,
    mapping: &mut MemoryMapping,
) -> Result<u64, EbpfError> {
    let op = unpack(subop, wide, args[0]).ok_or(EbpfError::UnsupportedInstruction)?;
    let reg = [0, args[0], args[1], args[2], args[3], args[4]];
    interpret(&op, &reg, vector_base, mapping)
}

/// The built-in function of one sub-operation
pub struct Intrinsic<const SUBOP: u8, const WIDE: bool>;

impl<C: ContextObject, const SUBOP: u8, const WIDE: bool> BuiltinFunctionDefinition<C>
    for Intrinsic<SUBOP, WIDE>
{
    type Error = EbpfError;

    fn rust(_: &mut C, _: u64, _: u64, _: u64, _: u64, _: u64) -> Result<u64, EbpfError> {
        // Unused, `vm` below needs the vector region and the memory mapping
        Err(EbpfError::UnsupportedInstruction)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn vm(mut vm: EncryptedHostAddressToEbpfVm<C>, a: u64, b: u64, c: u64, d: u64, e: u64) {
        unsafe {
            vm.with_vm(|vm| {
                let enable_insn_meter = vm.loader.get_config().enable_instruction_meter;
                if enable_insn_meter {
                    let used_cus = vm.previous_instruction_meter - vm.due_insn_count;
                    vm.context().consume(used_cus);
                }
                let vector_base = vm.simd_vector_base;
                vm.program_result =
                    match execute(SUBOP, WIDE, [a, b, c, d, e], vector_base, vm.memory()) {
                        Ok(value) => ProgramResult::Ok(value),
                        Err(err) => ProgramResult::Err(err),
                    };
                if enable_insn_meter {
                    vm.previous_instruction_meter = vm.context().get_remaining();
                }
            })
        }
    }

    fn codegen(jit: &mut JitCompiler<C>) {
        #[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
        jit.emit_simd_intrinsic(SUBOP, WIDE, <Self as BuiltinFunctionDefinition<C>>::vm);
        #[cfg(not(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64")))]
        jit.emit_external_call(<Self as BuiltinFunctionDefinition<C>>::vm);
    }
}

/// Registers all vector operations with the loader
pub fn register<C: ContextObject>(loader: &mut BuiltinProgram<C>) -> Result<(), ElfError> {
    macro_rules! register {
        ($wide:literal: $($subop:literal),*) => {
            $(loader.register_definition::<Intrinsic<$subop, $wide>>(name($subop, $wide).unwrap())?;)*
        };
    }
    register!(false: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 83,
        84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112);
    register!(true: 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 83,
        84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112);
    Ok(())
}
