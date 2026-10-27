//! The vector operations as static calls (SIMD-0645 style JIT intrinsics)
//!
//! Instead of a new opcode, every vector operation is a static syscall: `CALL_IMM` with `src = 0`,
//! `dst = 0`, `off = 0` and `imm = murmur32("sol_<op>")`. The bytecode stays within the BPF ISA,
//! and a runtime which doesn't know the names rejects the program at verification
//! (`InvalidSyscall`).
//!
//! ABI (chosen so that compiled code needs nothing but the call itself; a Rust wrapper
//! `sol_vadd32x8(ops(V0, V0, V1))` compiles to `mov64 r1, imm` + `call`):
//! - `r1` holds the packed operands (see [pack]), always a compile time constant in practice.
//! - `r2` holds the scalar operand: the address of `vload`/`vstore`, the value of `vbcast32`.
//! - The operation writes `r0 = 0` and leaves every other register unchanged.
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

/// Function names, indexed by sub-operation
pub const NAMES: [&str; SUBOP_COUNT as usize] = [
    "sol_vload256",
    "sol_vstore256",
    "sol_vadd32x8",
    "sol_vxor256",
    "sol_vand256",
    "sol_vor256",
    "sol_vror32x8",
    "sol_vshufd",
    "sol_vtranspose8",
    "sol_vbcast32",
    "sol_vmov256",
    "sol_vpermd",
    "sol_vblendd",
    "sol_vsub32x8",
    "sol_vmullo32x8",
    "sol_vminu32x8",
    "sol_vsrl32x8",
    "sol_vsll32x8",
    "sol_vsllv32x8",
    "sol_vsrlv32x8",
    "sol_vandn256",
];

/// Register holding the packed operands
pub const OPERAND_REG: u8 = 1;
/// Register holding the scalar operand
pub const SCALAR_REG: u8 = 2;

/// Static call identifier of a sub-operation
pub fn hash(subop: u8) -> u32 {
    ebpf::hash_symbol_name(NAMES[subop as usize].as_bytes())
}

/// Identifiers of all sub-operations, indexed by sub-operation
pub fn hashes() -> [u32; SUBOP_COUNT as usize] {
    let mut hashes = [0; SUBOP_COUNT as usize];
    for (subop, hash) in hashes.iter_mut().enumerate() {
        *hash = self::hash(subop as u8);
    }
    hashes
}

/// Packs the operands of `op` (slot mode, scalar operand in `r2`) into the `r1` value.
///
/// Slots are 8 bit (256 slots). The layout depends on the operation, and every packed value is
/// below `2^29`, so it always fits a `mov64 r1, imm32`:
///
/// ```text
/// vload256 / vstore256   bits 0..7 slot (vd / va)   bits 8..23 signed 16 bit offset
/// vbcast32 / vtranspose8 bits 0..7 vd
/// two sources            bits 0..7 vd   bits 8..15 va   bits 16..23 vb
/// one source + imm8      bits 0..7 vd   bits 8..15 va   bits 16..23 imm8   (vror, vshufd, vsrl, vsll)
/// vmov256                bits 0..7 vd   bits 8..15 va
/// vblendd                bits 0..6 vd   bits 7..13 va   bits 14..20 vb   bits 21..28 imm8 (slots < 128)
/// ```
pub fn pack(op: &VecInsn) -> u64 {
    let (vd, va, vb, imm) = (op.vd as u64, op.va as u64, op.vb as u64, op.off as u16 as u64);
    match op.subop {
        VLOAD => vd | imm << 8,
        VSTORE => va | imm << 8,
        VBCAST32 | VTRANSPOSE8 => vd,
        VROR32 | VSHUFD | VSRL32 | VSLL32 => vd | va << 8 | (imm & 0xff) << 16,
        VBLENDD => (vd & 0x7f) | (va & 0x7f) << 7 | (vb & 0x7f) << 14 | (imm & 0xff) << 21,
        VMOV => vd | va << 8,
        _ => vd | va << 8 | vb << 16,
    }
}

/// Decodes the packed operands, `None` if they are malformed (including any non canonical bits)
pub fn unpack(subop: u8, packed: u64) -> Option<VecInsn> {
    if subop >= SUBOP_COUNT || packed >> 29 != 0 {
        return None;
    }
    let byte = |shift: u32| ((packed >> shift) & 0xff) as u8;
    let (mut vd, mut va, mut vb, mut off) = (0u8, 0u8, 0u8, 0i16);
    match subop {
        VLOAD => (vd, off) = (byte(0), ((packed >> 8) & 0xffff) as u16 as i16),
        VSTORE => (va, off) = (byte(0), ((packed >> 8) & 0xffff) as u16 as i16),
        VBCAST32 | VTRANSPOSE8 => vd = byte(0),
        VROR32 | VSHUFD | VSRL32 | VSLL32 => (vd, va, off) = (byte(0), byte(8), byte(16) as i16),
        VBLENDD => {
            let field = |shift: u32| ((packed >> shift) & 0x7f) as u8;
            (vd, va, vb, off) = (field(0), field(7), field(14), byte(21) as i16);
        }
        VMOV => (vd, va) = (byte(0), byte(8)),
        _ => (vd, va, vb) = (byte(0), byte(8), byte(16)),
    }
    let src = if matches!(subop, VLOAD | VSTORE | VBCAST32) { SCALAR_REG } else { 0 };
    let op = VecInsn { subop, vd, va, vb, memory_mode: false, base_reg: 0, src, off };
    // The opcode form's validation (operand ranges), applied without its 6 bit slot encoding
    validate_slot_op(&op)?;
    (pack(&op) == packed).then_some(op)
}

/// Executes one call with the argument registers `r1..=r5`
pub fn execute(
    subop: u8,
    args: [u64; 5],
    vector_base: *mut u8,
    mapping: &mut MemoryMapping,
) -> Result<(), EbpfError> {
    let op = unpack(subop, args[0]).ok_or(EbpfError::UnsupportedInstruction)?;
    let reg = [0, args[0], args[1], args[2], args[3], args[4]];
    interpret(&op, &reg, vector_base, mapping)
}

/// The built-in function of one sub-operation
pub struct Intrinsic<const SUBOP: u8>;

impl<C: ContextObject, const SUBOP: u8> BuiltinFunctionDefinition<C> for Intrinsic<SUBOP> {
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
                vm.program_result = match execute(SUBOP, [a, b, c, d, e], vector_base, vm.memory()) {
                    Ok(()) => ProgramResult::Ok(0),
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
        jit.emit_simd_intrinsic(SUBOP, <Self as BuiltinFunctionDefinition<C>>::vm);
        #[cfg(not(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64")))]
        jit.emit_external_call(<Self as BuiltinFunctionDefinition<C>>::vm);
    }
}

/// Registers all vector operations with the loader
pub fn register<C: ContextObject>(loader: &mut BuiltinProgram<C>) -> Result<(), ElfError> {
    macro_rules! register {
        ($($subop:literal),*) => {
            $(loader.register_definition::<Intrinsic<$subop>>(NAMES[$subop])?;)*
        };
    }
    register!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    Ok(())
}
