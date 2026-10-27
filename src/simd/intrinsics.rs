//! The vector operations as static calls (SIMD-0645 style JIT intrinsics)
//!
//! Instead of a new opcode, every vector operation is a static syscall: `CALL_IMM` with `src = 0`,
//! `dst = 0`, `off = 0` and `imm = murmur32("sol_<op>")`. The bytecode stays within the BPF ISA,
//! and a runtime which doesn't know the names rejects the program at verification
//! (`InvalidSyscall`).
//!
//! ABI:
//! - `r5` holds the packed operands (see [pack]).
//! - The scalar operand (address of `vload`/`vstore`, value of `vbcast32`) is one of `r1..=r4`,
//!   selected by a field of the packed operands.
//! - The operation writes `r0 = 0` and leaves every other register unchanged.
//! - It costs the 1 CU of its `CALL_IMM` instruction and nothing else.
//!
//! Every execution backend runs the operation through [interpret]. The JIT lowers a call to inline
//! AVX2, exactly like the opcode form, when it can prove that `r5` holds a valid constant at the
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
];

/// Register holding the packed operands
pub const OPERAND_REG: u8 = 5;

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

/// Packs the operands of `op` (slot mode, scalar register in `r1..=r4`) into the `r5` value.
///
/// ```text
/// bits  0.. 5  vd
/// bits  6..11  va
/// bits 12..17  vb                (bits 12..27: signed 16 bit offset for vload / vstore)
/// bits 18..25  imm8              (vror32x8, vshufd, vblendd)
/// bits 28..30  scalar register   (1..=4 for vload, vstore, vbcast32, else 0)
/// all other bits zero, so the value always fits a `mov64 r5, imm32`
/// ```
pub fn pack(op: &VecInsn) -> u64 {
    let mut packed = op.vd as u64 | (op.va as u64) << 6 | (op.src as u64) << 28;
    if matches!(op.subop, VLOAD | VSTORE) {
        packed |= (op.off as u16 as u64) << 12;
    } else {
        packed |= (op.vb as u64) << 12 | (op.off as u64 & 0xff) << 18;
    }
    packed
}

/// Decodes the packed operands, `None` if they are malformed
pub fn unpack(subop: u8, packed: u64) -> Option<VecInsn> {
    if subop >= SUBOP_COUNT || packed >> 31 != 0 {
        return None;
    }
    let src = ((packed >> 28) & 7) as u8;
    let (vb, off) = if matches!(subop, VLOAD | VSTORE) {
        (0, ((packed >> 12) & 0xffff) as u16 as i16)
    } else {
        (((packed >> 12) & 63) as u8, ((packed >> 18) & 0xff) as i16)
    };
    let op = VecInsn {
        subop,
        vd: (packed & 63) as u8,
        va: ((packed >> 6) & 63) as u8,
        vb,
        memory_mode: false,
        base_reg: 0,
        src,
        off,
    };
    let uses_src = matches!(subop, VLOAD | VSTORE | VBCAST32);
    if uses_src && !(1..=4).contains(&src) {
        return None;
    }
    let op = VecInsn::decode(&op.encode())?;
    (pack(&op) == packed).then_some(op)
}

/// Executes one call with the argument registers `r1..=r5`
pub fn execute(
    subop: u8,
    args: [u64; 5],
    vector_base: *mut u8,
    mapping: &mut MemoryMapping,
) -> Result<(), EbpfError> {
    let op = unpack(subop, args[4]).ok_or(EbpfError::UnsupportedInstruction)?;
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
    register!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    Ok(())
}
