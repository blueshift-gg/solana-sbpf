//! Machinecode emission for x86-64

#![allow(clippy::arithmetic_side_effects)]
use crate::{
    jit::{JitCompiler, OperandSize},
    vm::ContextObject,
};

macro_rules! exclude_operand_sizes {
    ($size:expr, $($to_exclude:path)|+ $(,)?) => {
        debug_assert!(match $size {
            $($to_exclude)|+ => false,
            _ => true,
        });
    }
}

/// Symbolic names for x86-64 registers
#[allow(dead_code, clippy::upper_case_acronyms)]
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum X86Register {
    /// eax / rax
    RAX = 0,
    /// ecx / rcx
    RCX = 1,
    /// edx / rdx
    RDX = 2,
    /// ebx / rbx
    RBX = 3,
    /// esp / rsp
    RSP = 4,
    /// ebp / rbp
    RBP = 5,
    /// esi / rsi
    RSI = 6,
    /// edi / rdi
    RDI = 7,
    /// r8d / r8
    R8 = 8,
    /// r9d / r9
    R9 = 9,
    /// r10d / r10
    R10 = 10,
    /// r11d / r11
    R11 = 11,
    /// r12d / r12
    R12 = 12,
    /// r13d / r13
    R13 = 13,
    /// r14d / r14
    R14 = 14,
    /// r15d / r15
    R15 = 15,
    /// mm0
    MM0 = 16,
    /// mm1
    MM1 = 17,
    /// mm2
    MM2 = 18,
    /// mm3
    MM3 = 19,
    /// mm4
    MM4 = 20,
    /// mm5
    MM5 = 21,
    /// mm6
    MM6 = 22,
    /// mm7
    MM7 = 23,
}
use X86Register::*;

// System V AMD64 ABI
// Works on: Linux, macOS, BSD and Solaris but not on Windows

/// Registers which can pass arguments at function calls
pub const ARGUMENT_REGISTERS: [X86Register; 6] = [RDI, RSI, RDX, RCX, R8, R9];
/// Registers which have to be saved by the caller
pub const CALLER_SAVED_REGISTERS: [X86Register; 9] = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11];
/// Registers which have to be saved by the callee
pub const CALLEE_SAVED_REGISTERS: [X86Register; 6] = [RBP, RBX, R12, R13, R14, R15];

struct X86Rex {
    w: bool,
    r: bool,
    x: bool,
    b: bool,
}

struct X86ModRm {
    mode: u8,
    r: u8,
    m: u8,
}

struct X86Sib {
    scale: u8,
    index: u8,
    base: u8,
}

/// Scale, index, base (SIB)
#[derive(Copy, Clone)]
pub enum X86IndirectAccess {
    /// [second_operand + offset]
    Offset(i32),
    /// [second_operand + offset + index << shift]
    OffsetIndexShift(i32, X86Register, u8),
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub(crate) enum FenceType {
    /// lfence
    Load = 5,
    /// mfence
    All = 6,
    /// sfence
    Store = 7,
}

/// Generic x86 instruction frame
#[derive(Copy, Clone)]
pub struct X86Instruction {
    size: OperandSize,
    opcode_escape_sequence: u8,
    opcode: u8,
    modrm: bool,
    indirect: Option<X86IndirectAccess>,
    first_operand: u8,
    second_operand: u8,
    immediate_size: OperandSize,
    immediate: i64,
}

impl X86Instruction {
    /// Can be used to fill out unspecified fields in the constructor
    pub const DEFAULT: X86Instruction = X86Instruction {
        size: OperandSize::S0,
        opcode_escape_sequence: 0,
        opcode: 0,
        modrm: true,
        indirect: None,
        first_operand: 0,
        second_operand: 0,
        immediate_size: OperandSize::S0,
        immediate: 0,
    };

    /// Emits this instruction into the text section
    #[inline(always)]
    pub fn emit<C: ContextObject>(&self, jit: &mut JitCompiler<C>) {
        debug_assert!(!matches!(self.size, OperandSize::S0));
        let mut rex = X86Rex {
            w: matches!(self.size, OperandSize::S64),
            r: self.first_operand & 0b1000 != 0,
            x: false,
            b: self.second_operand & 0b1000 != 0,
        };
        let mut modrm = X86ModRm {
            mode: 0,
            r: self.first_operand & 0b111,
            m: self.second_operand & 0b111,
        };
        let mut sib = X86Sib {
            scale: 0,
            index: 0,
            base: 0,
        };
        let mut displacement_size = OperandSize::S0;
        let mut displacement = 0;
        if self.modrm {
            match self.indirect {
                Some(X86IndirectAccess::Offset(offset)) => {
                    displacement = offset;
                    debug_assert_ne!(self.second_operand & 0b111, 4); // Reserved for SIB addressing
                    if (-128..=127).contains(&displacement)
                        || (displacement == 0 && self.second_operand & 0b111 == 5)
                    {
                        displacement_size = OperandSize::S8;
                        modrm.mode = 1;
                    } else {
                        displacement_size = OperandSize::S32;
                        modrm.mode = 2;
                    }
                }
                Some(X86IndirectAccess::OffsetIndexShift(offset, index, shift)) => {
                    displacement = offset;
                    if (-128..=127).contains(&displacement) {
                        displacement_size = OperandSize::S8;
                        modrm.mode = 1;
                    } else {
                        displacement_size = OperandSize::S32;
                        modrm.mode = 2;
                    }
                    modrm.m = 4;
                    rex.x = (index as u8) & 0b1000 != 0;
                    sib.scale = shift & 0b11;
                    sib.index = (index as u8) & 0b111;
                    sib.base = self.second_operand & 0b111;
                }
                None => {
                    modrm.mode = 3;
                }
            }
        }
        if matches!(self.size, OperandSize::S16) {
            jit.emit::<u8>(0x66);
        }
        let rex =
            ((rex.w as u8) << 3) | ((rex.r as u8) << 2) | ((rex.x as u8) << 1) | (rex.b as u8);
        if rex != 0 {
            jit.emit::<u8>(0x40 | rex);
        }
        match self.opcode_escape_sequence {
            1 => jit.emit::<u8>(0x0f),
            2 => jit.emit::<u16>(0x0f38),
            3 => jit.emit::<u16>(0x0f3a),
            _ => {}
        }
        jit.emit::<u8>(self.opcode);
        if self.modrm {
            jit.emit::<u8>((modrm.mode << 6) | (modrm.r << 3) | modrm.m);
            let sib = (sib.scale << 6) | (sib.index << 3) | sib.base;
            if sib != 0 {
                jit.emit::<u8>(sib);
            }
            jit.emit_variable_length(displacement_size, displacement as u64);
        }
        jit.emit_variable_length(self.immediate_size, self.immediate as u64);
    }

    /// Arithmetic or logic
    pub const fn alu_escaped(
        size: OperandSize,
        opcode_escape_sequence: u8,
        opcode: u8,
        source: X86Register,
        destination: X86Register,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0 | OperandSize::S8 | OperandSize::S16);
        Self {
            size,
            opcode_escape_sequence,
            opcode,
            first_operand: source as u8,
            second_operand: destination as u8,
            indirect,
            ..X86Instruction::DEFAULT
        }
    }

    /// Arithmetic or logic
    pub const fn alu_immediate_escaped(
        size: OperandSize,
        opcode_escape_sequence: u8,
        opcode: u8,
        opcode_extension: u8,
        destination: X86Register,
        immediate: i64,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0 | OperandSize::S8 | OperandSize::S16);
        Self {
            size,
            opcode_escape_sequence,
            opcode,
            first_operand: opcode_extension,
            second_operand: destination as u8,
            immediate_size: match opcode {
                0xc1 => OperandSize::S8,
                0x81 => OperandSize::S32,
                0xf7 if opcode_extension == 0 => OperandSize::S32,
                _ => OperandSize::S0,
            },
            immediate,
            indirect,
            ..X86Instruction::DEFAULT
        }
    }

    /// Arithmetic or logic
    pub const fn alu(
        size: OperandSize,
        opcode: u8,
        source: X86Register,
        destination: X86Register,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        Self::alu_escaped(size, 0, opcode, source, destination, indirect)
    }

    /// Arithmetic or logic
    pub const fn alu_immediate(
        size: OperandSize,
        opcode: u8,
        opcode_extension: u8,
        destination: X86Register,
        immediate: i64,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        Self::alu_immediate_escaped(
            size,
            0,
            opcode,
            opcode_extension,
            destination,
            immediate,
            indirect,
        )
    }

    /// Move source to destination
    pub const fn mov(size: OperandSize, source: X86Register, destination: X86Register) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0 | OperandSize::S8 | OperandSize::S16);
        Self {
            size,
            opcode: 0x89,
            first_operand: source as u8,
            second_operand: destination as u8,
            ..Self::DEFAULT
        }
    }

    /// Move source to destination
    pub const fn mov_with_sign_extension(
        size: OperandSize,
        source: X86Register,
        destination: X86Register,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0 | OperandSize::S8 | OperandSize::S16);
        Self {
            size,
            opcode: 0x63,
            first_operand: destination as u8,
            second_operand: source as u8,
            ..Self::DEFAULT
        }
    }

    /// Move to / from / between MMX (float mantissa)
    #[allow(dead_code)]
    pub const fn mov_mmx(size: OperandSize, source: X86Register, destination: X86Register) -> Self {
        exclude_operand_sizes!(
            size,
            OperandSize::S0 | OperandSize::S8 | OperandSize::S16 | OperandSize::S32
        );
        if (destination as u8) & 16 != 0 {
            // If the destination is a MMX register
            Self {
                size,
                opcode_escape_sequence: 1,
                opcode: if (source as u8) & 16 != 0 { 0x6F } else { 0x6E },
                first_operand: (destination as u8) & 0xF,
                second_operand: (source as u8) & 0xF,
                ..Self::DEFAULT
            }
        } else {
            Self {
                size,
                opcode_escape_sequence: 1,
                opcode: 0x7E,
                first_operand: (source as u8) & 0xF,
                second_operand: (destination as u8) & 0xF,
                ..Self::DEFAULT
            }
        }
    }

    /// Conditionally move source to destination
    pub const fn cmov(
        size: OperandSize,
        condition: u8,
        source: X86Register,
        destination: X86Register,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0 | OperandSize::S8 | OperandSize::S16);
        Self {
            size,
            opcode_escape_sequence: 1,
            opcode: condition,
            first_operand: destination as u8,
            second_operand: source as u8,
            ..Self::DEFAULT
        }
    }

    /// Swap source and destination
    pub const fn xchg(
        size: OperandSize,
        source: X86Register,
        destination: X86Register,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        exclude_operand_sizes!(
            size,
            OperandSize::S0 | OperandSize::S8 | OperandSize::S16 | OperandSize::S32,
        );
        Self {
            size,
            opcode: 0x87,
            first_operand: source as u8,
            second_operand: destination as u8,
            indirect,
            ..Self::DEFAULT
        }
    }

    /// Swap byte order of destination
    pub const fn bswap(size: OperandSize, destination: X86Register) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0 | OperandSize::S8);
        match size {
            OperandSize::S16 => Self {
                size,
                opcode: 0xc1,
                second_operand: destination as u8,
                immediate_size: OperandSize::S8,
                immediate: 8,
                ..Self::DEFAULT
            },
            OperandSize::S32 | OperandSize::S64 => Self {
                size,
                opcode_escape_sequence: 1,
                opcode: 0xc8 | ((destination as u8) & 0b111),
                modrm: false,
                second_operand: destination as u8,
                ..Self::DEFAULT
            },
            _ => unimplemented!(),
        }
    }

    /// Test source and destination
    pub const fn test(
        size: OperandSize,
        source: X86Register,
        destination: X86Register,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0);
        Self {
            size,
            opcode: if let OperandSize::S8 = size {
                0x84
            } else {
                0x85
            },
            first_operand: source as u8,
            second_operand: destination as u8,
            indirect,
            ..Self::DEFAULT
        }
    }

    /// Test immediate and destination
    pub const fn test_immediate(
        size: OperandSize,
        destination: X86Register,
        immediate: i64,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0);
        Self {
            size,
            opcode: if let OperandSize::S8 = size {
                0xf6
            } else {
                0xf7
            },
            first_operand: 0,
            second_operand: destination as u8,
            immediate_size: if let OperandSize::S64 = size {
                OperandSize::S32
            } else {
                size
            },
            immediate,
            indirect,
            ..Self::DEFAULT
        }
    }

    /// Compare source and destination
    pub const fn cmp(
        size: OperandSize,
        source: X86Register,
        destination: X86Register,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0);
        Self {
            size,
            opcode: if let OperandSize::S8 = size {
                0x38
            } else {
                0x39
            },
            first_operand: source as u8,
            second_operand: destination as u8,
            indirect,
            ..Self::DEFAULT
        }
    }

    /// Compare immediate and destination
    pub const fn cmp_immediate(
        size: OperandSize,
        destination: X86Register,
        immediate: i64,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0);
        Self {
            size,
            opcode: if let OperandSize::S8 = size {
                0x80
            } else {
                0x81
            },
            first_operand: 7,
            second_operand: destination as u8,
            immediate_size: if let OperandSize::S64 = size {
                OperandSize::S32
            } else {
                size
            },
            immediate,
            indirect,
            ..Self::DEFAULT
        }
    }

    /// Load effective address of source into destination
    pub const fn lea(
        size: OperandSize,
        source: X86Register,
        destination: X86Register,
        indirect: Option<X86IndirectAccess>,
    ) -> Self {
        exclude_operand_sizes!(
            size,
            OperandSize::S0 | OperandSize::S8 | OperandSize::S16 | OperandSize::S32,
        );
        Self {
            size,
            opcode: 0x8d,
            first_operand: destination as u8,
            second_operand: source as u8,
            indirect,
            ..Self::DEFAULT
        }
    }

    /// Convert word to doubleword or doubleword to quadword
    pub const fn sign_extend_rax_rdx(size: OperandSize) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0 | OperandSize::S8 | OperandSize::S16);
        Self {
            size,
            opcode: 0x99,
            modrm: false,
            ..X86Instruction::DEFAULT
        }
    }

    /// Load destination from [source + offset]
    pub const fn load(
        size: OperandSize,
        source: X86Register,
        destination: X86Register,
        indirect: X86IndirectAccess,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0);
        Self {
            size: if let OperandSize::S64 = size {
                OperandSize::S64
            } else {
                OperandSize::S32
            },
            opcode_escape_sequence: match size {
                OperandSize::S8 | OperandSize::S16 => 1,
                _ => 0,
            },
            opcode: match size {
                OperandSize::S8 => 0xb6,
                OperandSize::S16 => 0xb7,
                _ => 0x8b,
            },
            first_operand: destination as u8,
            second_operand: source as u8,
            indirect: Some(indirect),
            ..Self::DEFAULT
        }
    }

    /// Store source in [destination + offset]
    pub const fn store(
        size: OperandSize,
        source: X86Register,
        destination: X86Register,
        indirect: X86IndirectAccess,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0);
        Self {
            size,
            opcode: match size {
                OperandSize::S8 => 0x88,
                _ => 0x89,
            },
            first_operand: source as u8,
            second_operand: destination as u8,
            indirect: Some(indirect),
            ..Self::DEFAULT
        }
    }

    /// Load destination from immediate
    pub const fn load_immediate(destination: X86Register, immediate: i64) -> Self {
        let mut size = OperandSize::S64;
        if immediate >= 0 {
            if immediate <= u32::MAX as i64 {
                // Zero extend u32 imm to u64 reg
                size = OperandSize::S32;
            }
        } else if immediate >= i32::MIN as i64 {
            // Sign extend i32 imm to i64 reg
            return Self {
                size: OperandSize::S64,
                opcode: 0xc7,
                second_operand: destination as u8,
                immediate_size: OperandSize::S32,
                immediate,
                ..Self::DEFAULT
            };
        }
        // Load full u64 imm into u64 reg
        Self {
            size,
            opcode: (0xb8 | ((destination as u8) & 0b111)),
            modrm: false,
            second_operand: destination as u8,
            immediate_size: size,
            immediate,
            ..Self::DEFAULT
        }
    }

    /// Store sign-extended immediate in destination
    pub const fn store_immediate(
        size: OperandSize,
        destination: X86Register,
        indirect: X86IndirectAccess,
        immediate: i64,
    ) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0);
        Self {
            size,
            opcode: match size {
                OperandSize::S8 => 0xc6,
                _ => 0xc7,
            },
            second_operand: destination as u8,
            indirect: Some(indirect),
            immediate_size: if let OperandSize::S64 = size {
                OperandSize::S32
            } else {
                size
            },
            immediate,
            ..Self::DEFAULT
        }
    }

    /// Push source onto the stack
    #[allow(dead_code)]
    pub const fn push_immediate(size: OperandSize, immediate: i32) -> Self {
        exclude_operand_sizes!(size, OperandSize::S0 | OperandSize::S16);
        Self {
            size,
            opcode: match size {
                OperandSize::S8 => 0x6A,
                _ => 0x68,
            },
            modrm: false,
            immediate_size: if let OperandSize::S64 = size {
                OperandSize::S32
            } else {
                size
            },
            immediate: immediate as i64,
            ..Self::DEFAULT
        }
    }

    /// Push source onto the stack
    pub const fn push(source: X86Register, indirect: Option<X86IndirectAccess>) -> Self {
        if indirect.is_none() {
            Self {
                size: OperandSize::S32,
                opcode: 0x50 | ((source as u8) & 0b111),
                modrm: false,
                second_operand: source as u8,
                ..Self::DEFAULT
            }
        } else {
            Self {
                size: OperandSize::S64,
                opcode: 0xFF,
                modrm: true,
                first_operand: 6,
                second_operand: source as u8,
                indirect,
                ..Self::DEFAULT
            }
        }
    }

    /// Pop from the stack into destination
    pub const fn pop(destination: X86Register) -> Self {
        Self {
            size: OperandSize::S32,
            opcode: 0x58 | ((destination as u8) & 0b111),
            modrm: false,
            second_operand: destination as u8,
            ..Self::DEFAULT
        }
    }

    /// Jump to relative destination on condition
    pub const fn conditional_jump_immediate(opcode: u8, relative_destination: i32) -> Self {
        Self {
            size: OperandSize::S32,
            opcode_escape_sequence: 1,
            opcode,
            modrm: false,
            immediate_size: OperandSize::S32,
            immediate: relative_destination as i64,
            ..Self::DEFAULT
        }
    }

    /// Jump to relative destination
    pub const fn jump_immediate(relative_destination: i32) -> Self {
        Self {
            size: OperandSize::S32,
            opcode: 0xe9,
            modrm: false,
            immediate_size: OperandSize::S32,
            immediate: relative_destination as i64,
            ..Self::DEFAULT
        }
    }

    /// Jump to absolute destination
    #[allow(dead_code)]
    pub const fn jump_reg(destination: X86Register, indirect: Option<X86IndirectAccess>) -> Self {
        Self {
            size: OperandSize::S64,
            opcode: 0xff,
            first_operand: 4,
            second_operand: destination as u8,
            indirect,
            ..Self::DEFAULT
        }
    }

    /// Push RIP and jump to relative destination
    pub const fn call_immediate(relative_destination: i32) -> Self {
        Self {
            size: OperandSize::S32,
            opcode: 0xe8,
            modrm: false,
            immediate_size: OperandSize::S32,
            immediate: relative_destination as i64,
            ..Self::DEFAULT
        }
    }

    /// Push RIP and jump to absolute destination
    pub const fn call_reg(destination: X86Register, indirect: Option<X86IndirectAccess>) -> Self {
        Self {
            size: OperandSize::S64,
            opcode: 0xff,
            first_operand: 2,
            second_operand: destination as u8,
            indirect,
            ..Self::DEFAULT
        }
    }

    /// Pop RIP
    pub const fn return_near() -> Self {
        Self {
            size: OperandSize::S32,
            opcode: 0xc3,
            modrm: false,
            ..Self::DEFAULT
        }
    }

    /// No operation
    #[allow(dead_code)]
    pub const fn noop() -> Self {
        Self {
            size: OperandSize::S32,
            opcode: 0x90,
            modrm: false,
            ..Self::DEFAULT
        }
    }

    /// Trap / software interrupt
    #[allow(dead_code)]
    pub const fn interrupt(immediate: u8) -> Self {
        if immediate == 3 {
            Self {
                size: OperandSize::S32,
                opcode: 0xcc,
                modrm: false,
                ..Self::DEFAULT
            }
        } else {
            Self {
                size: OperandSize::S32,
                opcode: 0xcd,
                modrm: false,
                immediate_size: OperandSize::S8,
                immediate: immediate as i64,
                ..Self::DEFAULT
            }
        }
    }

    /// rdtsc
    pub const fn cycle_count() -> Self {
        Self {
            size: OperandSize::S32,
            opcode_escape_sequence: 1,
            opcode: 0x31,
            modrm: false,
            ..Self::DEFAULT
        }
    }

    /// lfence / sfence / mfence
    #[allow(dead_code)]
    pub(crate) const fn fence(fence_type: FenceType) -> Self {
        Self {
            size: OperandSize::S32,
            opcode_escape_sequence: 1,
            opcode: 0xae,
            first_operand: fence_type as u8,
            ..Self::DEFAULT
        }
    }
}

/// Register or memory operand of a VEX encoded instruction
#[cfg(feature = "simd-experiment")]
#[derive(Copy, Clone)]
pub enum VexRm {
    /// Register (YMM / XMM / GPR depending on the instruction)
    Reg(u8),
    /// `[base + displacement]`
    Mem(X86Register, i32),
    /// `[rip + (target - end_of_instruction)]`
    Rip(*const u8),
}

/// VEX encoded (AVX / AVX2) instruction
#[cfg(feature = "simd-experiment")]
#[derive(Copy, Clone)]
pub struct VexInstruction {
    /// Implied mandatory prefix: 0 = none, 1 = 66, 2 = F3, 3 = F2
    pub pp: u8,
    /// Opcode map: 1 = 0F, 2 = 0F38, 3 = 0F3A
    pub map: u8,
    /// VEX.W
    pub w: bool,
    /// VEX.L (256 bit)
    pub l: bool,
    /// Opcode byte
    pub opcode: u8,
    /// ModRM.reg (register number or opcode extension)
    pub reg: u8,
    /// Additional source register (VEX.vvvv), 0 if unused
    pub vvvv: u8,
    /// ModRM.rm operand
    pub rm: VexRm,
    /// Optional trailing 8 bit immediate
    pub imm: Option<u8>,
}

#[cfg(feature = "simd-experiment")]
impl VexInstruction {
    /// A 256 bit instruction: prefix (0 none, 1 66, 2 F3, 3 F2), map (1 0F, 2 0F38, 3 0F3A), opcode,
    /// ModRM.reg, VEX.vvvv, ModRM.rm, immediate
    pub const fn ymm(pp: u8, map: u8, opcode: u8, reg: u8, vvvv: u8, rm: VexRm, imm: Option<u8>) -> Self {
        Self { pp, map, w: false, l: true, opcode, reg, vvvv, rm, imm }
    }

    /// vmovdqu ymm, m256
    pub const fn load(dst: u8, base: X86Register, disp: i32) -> Self {
        Self::ymm(2, 1, 0x6f, dst, 0, VexRm::Mem(base, disp), None)
    }
    /// vmovdqu m256, ymm
    pub const fn store(src: u8, base: X86Register, disp: i32) -> Self {
        Self::ymm(2, 1, 0x7f, src, 0, VexRm::Mem(base, disp), None)
    }
    /// vmovdqu ymm, [rip + target]
    pub const fn load_rip(dst: u8, target: *const u8) -> Self {
        Self::ymm(2, 1, 0x6f, dst, 0, VexRm::Rip(target), None)
    }
    /// vmovdqa ymm, ymm
    pub const fn mov(dst: u8, src: u8) -> Self {
        Self::ymm(1, 1, 0x6f, dst, 0, VexRm::Reg(src), None)
    }
    /// Three operand packed integer op: vpaddd (0xfe), vpxor (0xef), vpand (0xdb), vpor (0xeb),
    /// vpunpckldq (0x62), vpunpckhdq (0x6a), vpunpcklqdq (0x6c), vpunpckhqdq (0x6d)
    pub const fn alu(opcode: u8, dst: u8, src1: u8, src2: VexRm) -> Self {
        Self::ymm(1, 1, opcode, dst, src1, src2, None)
    }
    /// Three operand packed integer op of the 0F38 map: vpmulld (0x40), vpminud (0x3b)
    pub const fn alu38(opcode: u8, dst: u8, src1: u8, src2: VexRm) -> Self {
        Self::ymm(1, 2, opcode, dst, src1, src2, None)
    }
    /// Three operand packed integer op of the 0F38 map with VEX.W1: vpsllvq (0x47), vpsrlvq (0x45)
    pub const fn alu38w(opcode: u8, dst: u8, src1: u8, src2: VexRm) -> Self {
        Self { w: true, ..Self::ymm(1, 2, opcode, dst, src1, src2, None) }
    }
    /// vpsrad ymm, ymm, imm8
    pub const fn psrad(dst: u8, src: u8, imm: u8) -> Self {
        Self::ymm(1, 1, 0x72, 4, dst, VexRm::Reg(src), Some(imm))
    }
    /// vmovmskps r32, ymm (`pd`: vmovmskpd)
    pub const fn movmskps(dst: X86Register, src: u8, pd: bool) -> Self {
        Self::ymm(pd as u8, 1, 0x50, dst as u8, 0, VexRm::Reg(src), None)
    }
    /// vpermq ymm, ymm, imm8
    pub const fn permq(dst: u8, src: u8, imm: u8) -> Self {
        Self { w: true, ..Self::ymm(1, 3, 0x00, dst, 0, VexRm::Reg(src), Some(imm)) }
    }
    /// vpblendvb ymm, ymm, ymm, ymm(mask): picks src2 where the mask byte's sign bit is set
    pub const fn pblendvb(dst: u8, src1: u8, src2: u8, mask: u8) -> Self {
        Self::ymm(1, 3, 0x4c, dst, src1, VexRm::Reg(src2), Some(mask << 4))
    }
    /// vpbroadcastw ymm, xmm
    pub const fn pbroadcastw(dst: u8, src: u8) -> Self {
        Self::ymm(1, 2, 0x79, dst, 0, VexRm::Reg(src), None)
    }
    /// vpmovzxbd ymm, m64
    pub const fn pmovzxbd(dst: u8, base: X86Register, disp: i32) -> Self {
        Self::ymm(1, 2, 0x31, dst, 0, VexRm::Mem(base, disp), None)
    }
    /// vpsllq ymm, ymm, imm8
    pub const fn psllq(dst: u8, src: u8, imm: u8) -> Self {
        Self::ymm(1, 1, 0x73, 6, dst, VexRm::Reg(src), Some(imm))
    }
    /// kmovd k, r32 / kmovq k, r64
    pub const fn kmov_from_gpr(k: u8, src: X86Register, q: bool) -> Self {
        Self { pp: 3, map: 1, w: q, l: false, opcode: 0x92, reg: k, vvvv: 0, rm: VexRm::Reg(src as u8), imm: None }
    }
    /// vpsrlq ymm, ymm, imm8
    pub const fn psrlq(dst: u8, src: u8, imm: u8) -> Self {
        Self::ymm(1, 1, 0x73, 2, dst, VexRm::Reg(src), Some(imm))
    }
    /// vpsrld ymm, ymm, imm8
    pub const fn psrld(dst: u8, src: u8, imm: u8) -> Self {
        Self::ymm(1, 1, 0x72, 2, dst, VexRm::Reg(src), Some(imm))
    }
    /// vpslld ymm, ymm, imm8
    pub const fn pslld(dst: u8, src: u8, imm: u8) -> Self {
        Self::ymm(1, 1, 0x72, 6, dst, VexRm::Reg(src), Some(imm))
    }
    /// vpshufb ymm, ymm, ymm/m256
    pub const fn pshufb(dst: u8, src: u8, mask: VexRm) -> Self {
        Self::ymm(1, 2, 0x00, dst, src, mask, None)
    }
    /// vpshufd ymm, ymm/m256, imm8
    pub const fn pshufd(dst: u8, src: VexRm, imm: u8) -> Self {
        Self::ymm(1, 1, 0x70, dst, 0, src, Some(imm))
    }
    /// vpermd ymm, ymm(indices), ymm/m256(data)
    pub const fn permd(dst: u8, indices: u8, data: VexRm) -> Self {
        Self::ymm(1, 2, 0x36, dst, indices, data, None)
    }
    /// vpblendd ymm, ymm, ymm/m256, imm8
    pub const fn pblendd(dst: u8, src1: u8, src2: VexRm, imm: u8) -> Self {
        Self::ymm(1, 3, 0x02, dst, src1, src2, Some(imm))
    }
    /// vpblendw ymm, ymm, ymm/m256, imm8
    pub const fn pblendw(dst: u8, src1: u8, src2: VexRm, imm: u8) -> Self {
        Self::ymm(1, 3, 0x0e, dst, src1, src2, Some(imm))
    }
    /// vpcmpeqd ymm, ymm, ymm/m256
    pub const fn pcmpeqd(dst: u8, src1: u8, src2: VexRm) -> Self {
        Self::ymm(1, 1, 0x76, dst, src1, src2, None)
    }
    /// vperm2i128 ymm, ymm, ymm/m256, imm8
    pub const fn perm2i128(dst: u8, src1: u8, src2: VexRm, imm: u8) -> Self {
        Self::ymm(1, 3, 0x46, dst, src1, src2, Some(imm))
    }
    /// vmovd xmm, r32
    pub const fn movd_from_gpr(dst: u8, src: X86Register) -> Self {
        Self { pp: 1, map: 1, w: false, l: false, opcode: 0x6e, reg: dst, vvvv: 0, rm: VexRm::Reg(src as u8), imm: None }
    }
    /// vpbroadcastd ymm, xmm
    pub const fn pbroadcastd(dst: u8, src: u8) -> Self {
        Self::ymm(1, 2, 0x58, dst, 0, VexRm::Reg(src), None)
    }
    /// vpbroadcastq ymm, xmm
    pub const fn pbroadcastq(dst: u8, src: u8) -> Self {
        Self::ymm(1, 2, 0x59, dst, 0, VexRm::Reg(src), None)
    }
    /// vmovq xmm, r64
    pub const fn movq_from_gpr(dst: u8, src: X86Register) -> Self {
        Self { pp: 1, map: 1, w: true, l: false, opcode: 0x6e, reg: dst, vvvv: 0, rm: VexRm::Reg(src as u8), imm: None }
    }
    /// vmovq r64, xmm
    pub const fn movq_to_gpr(dst: X86Register, src: u8) -> Self {
        Self { pp: 1, map: 1, w: true, l: false, opcode: 0x7e, reg: src, vvvv: 0, rm: VexRm::Reg(dst as u8), imm: None }
    }
    /// vpextrq r64, xmm, imm8
    pub const fn pextrq(dst: X86Register, src: u8, imm: u8) -> Self {
        Self { pp: 1, map: 3, w: true, l: false, opcode: 0x16, reg: src, vvvv: 0, rm: VexRm::Reg(dst as u8), imm: Some(imm) }
    }
    /// vextracti128 xmm, ymm, imm8
    pub const fn extracti128(dst: u8, src: u8, imm: u8) -> Self {
        Self::ymm(1, 3, 0x39, src, 0, VexRm::Reg(dst), Some(imm))
    }
    /// vpmovmskb r32, ymm
    pub const fn pmovmskb(dst: X86Register, src: u8) -> Self {
        Self::ymm(1, 1, 0xd7, dst as u8, 0, VexRm::Reg(src), None)
    }
    /// vblendvps ymm, ymm, ymm/m256, ymm(mask): picks src2 where the mask lane's sign bit is set
    pub const fn blendvps(dst: u8, src1: u8, src2: u8, mask: u8) -> Self {
        Self::ymm(1, 3, 0x4a, dst, src1, VexRm::Reg(src2), Some(mask << 4))
    }

    /// Encodes the instruction. `address` is the host address the instruction will be placed at
    /// (only needed for RIP relative operands).
    pub fn encode(&self, address: *const u8) -> ([u8; 16], usize) {
        let mut out = [0u8; 16];
        let mut len = 0;
        let mut push = |byte: u8, len: &mut usize| {
            out[*len] = byte;
            *len += 1;
        };
        let rm_ext = match self.rm {
            VexRm::Reg(r) => (r >> 3) & 1,
            VexRm::Mem(base, _) => ((base as u8) >> 3) & 1,
            VexRm::Rip(_) => 0,
        };
        let r_bar = (!(self.reg >> 3) & 1) << 7;
        let vvvv_bar = (!self.vvvv & 0xf) << 3;
        let l = (self.l as u8) << 2;
        if self.map == 1 && !self.w && rm_ext == 0 {
            push(0xc5, &mut len);
            push(r_bar | vvvv_bar | l | self.pp, &mut len);
        } else {
            push(0xc4, &mut len);
            push(r_bar | (1 << 6) | ((!rm_ext & 1) << 5) | self.map, &mut len);
            push(((self.w as u8) << 7) | vvvv_bar | l | self.pp, &mut len);
        }
        push(self.opcode, &mut len);
        let reg = (self.reg & 7) << 3;
        let imm_len = self.imm.is_some() as usize;
        match self.rm {
            VexRm::Reg(r) => push(0xc0 | reg | (r & 7), &mut len),
            VexRm::Mem(base, disp) => {
                let b = (base as u8) & 7;
                let mode = if disp == 0 && b != 5 {
                    0
                } else if (-128..=127).contains(&disp) {
                    1
                } else {
                    2
                };
                push((mode << 6) | reg | b, &mut len);
                if b == 4 {
                    push(0x24, &mut len); // SIB: no index, base = rsp / r12
                }
                match mode {
                    1 => push(disp as i8 as u8, &mut len),
                    2 => disp.to_le_bytes().iter().for_each(|byte| push(*byte, &mut len)),
                    _ => {}
                }
            }
            VexRm::Rip(target) => {
                push(reg | 5, &mut len);
                let end = (address as isize) + (len + 4 + imm_len) as isize;
                let disp = (target as isize - end) as i32;
                disp.to_le_bytes().iter().for_each(|byte| push(*byte, &mut len));
            }
        }
        if let Some(imm) = self.imm {
            push(imm, &mut len);
        }
        (out, len)
    }
}

#[cfg(feature = "simd-experiment")]
impl VexInstruction {
    /// kmovw k, m16 (pp 0, W0) / kmovd k, m32 (pp 1, W1)
    pub const fn kmov_load(k: u8, dwords: bool, src: VexRm) -> Self {
        Self { pp: dwords as u8, map: 1, w: dwords, l: false, opcode: 0x90, reg: k, vvvv: 0, rm: src, imm: None }
    }
    /// kmovd r32, k
    pub const fn kmovd_to_gpr(dst: X86Register, k: u8) -> Self {
        Self { pp: 3, map: 1, w: false, l: false, opcode: 0x93, reg: dst as u8, vvvv: 0, rm: VexRm::Reg(k), imm: None }
    }
    /// kmovq r64, k
    pub const fn kmovq_to_gpr(dst: X86Register, k: u8) -> Self {
        Self { pp: 3, map: 1, w: true, l: false, opcode: 0x93, reg: dst as u8, vvvv: 0, rm: VexRm::Reg(k), imm: None }
    }
}

/// EVEX encoded (AVX-512) instruction. Register numbers go up to 31, `vl` is 1 (256 bit) or 2
/// (512 bit). Memory operands must be full vectors (disp8 is scaled by the vector length).
#[cfg(feature = "simd-experiment")]
#[derive(Copy, Clone)]
pub struct EvexInstruction {
    /// Implied mandatory prefix: 0 = none, 1 = 66, 2 = F3, 3 = F2
    pub pp: u8,
    /// Opcode map: 1 = 0F, 2 = 0F38, 3 = 0F3A
    pub map: u8,
    /// EVEX.W
    pub w: bool,
    /// EVEX.L'L: 1 = 256 bit, 2 = 512 bit
    pub vl: u8,
    /// Opcode byte
    pub opcode: u8,
    /// ModRM.reg (register number or opcode extension)
    pub reg: u8,
    /// Additional source register (EVEX.V'vvvv), 0 if unused
    pub vvvv: u8,
    /// ModRM.rm operand
    pub rm: VexRm,
    /// Opmask register (EVEX.aaa), 0 for none
    pub mask: u8,
    /// Zeroing instead of merging masking (EVEX.z)
    pub zero: bool,
    /// Optional trailing 8 bit immediate
    pub imm: Option<u8>,
}

#[cfg(feature = "simd-experiment")]
impl EvexInstruction {
    /// An instruction with the 66 prefix family: prefix, map, EVEX.W, vector length (1 ymm, 2 zmm),
    /// opcode, ModRM.reg, EVEX.vvvv, ModRM.rm, immediate
    #[allow(clippy::too_many_arguments)]
    pub const fn op(pp: u8, map: u8, w: bool, vl: u8, opcode: u8, reg: u8, vvvv: u8, rm: VexRm, imm: Option<u8>) -> Self {
        Self { pp, map, w, vl, opcode, reg, vvvv, rm, mask: 0, zero: false, imm }
    }

    /// vmovdqu32 zmm, m512
    pub const fn load(dst: u8, base: X86Register, disp: i32) -> Self {
        Self::op(2, 1, false, 2, 0x6f, dst, 0, VexRm::Mem(base, disp), None)
    }
    /// vmovdqu32 m512, zmm
    pub const fn store(src: u8, base: X86Register, disp: i32) -> Self {
        Self::op(2, 1, false, 2, 0x7f, src, 0, VexRm::Mem(base, disp), None)
    }
    /// vmovdqa32 zmm, zmm
    pub const fn mov(dst: u8, src: u8) -> Self {
        Self::op(1, 1, false, 2, 0x6f, dst, 0, VexRm::Reg(src), None)
    }
    /// Three operand packed integer op of the 0F map (`vl` wide): vpaddd (0xfe), vpxord (0xef), ...
    pub const fn alu(vl: u8, opcode: u8, dst: u8, src1: u8, src2: VexRm) -> Self {
        Self::op(1, 1, false, vl, opcode, dst, src1, src2, None)
    }
    /// Three operand packed integer op of the 0F38 map: vpmulld (0x40), vprolvd (0x15), ...
    pub const fn alu38(vl: u8, opcode: u8, dst: u8, src1: u8, src2: VexRm) -> Self {
        Self::op(1, 2, false, vl, opcode, dst, src1, src2, None)
    }
    /// Three operand packed integer op with EVEX.W1 (quadword elements): vpunpcklqdq (map 1, 0x6c),
    /// vpunpckhqdq (map 1, 0x6d), vprolvq (map 2, 0x15)
    pub const fn alu_q(vl: u8, map: u8, opcode: u8, dst: u8, src1: u8, src2: VexRm) -> Self {
        Self::op(1, map, true, vl, opcode, dst, src1, src2, None)
    }
    /// vshufi64x2 zmm, zmm, zmm, imm8 (128 bit lanes: two from src1, two from src2)
    pub const fn shufi64x2(dst: u8, src1: u8, src2: u8, imm: u8) -> Self {
        Self::op(1, 3, true, 2, 0x43, dst, src1, VexRm::Reg(src2), Some(imm))
    }
    /// Shift / rotate by an immediate (opcode 0x72): /0 vprord, /1 vprold, /2 vpsrld, /6 vpslld
    pub const fn shift_imm(vl: u8, ext: u8, dst: u8, src: u8, imm: u8) -> Self {
        Self::op(1, 1, false, vl, 0x72, ext, dst, VexRm::Reg(src), Some(imm))
    }
    /// vpshufd zmm, zmm/m512, imm8
    pub const fn pshufd(dst: u8, src: VexRm, imm: u8) -> Self {
        Self::op(1, 1, false, 2, 0x70, dst, 0, src, Some(imm))
    }
    /// vpermd zmm, zmm(indices), zmm/m512(data)
    pub const fn permd(dst: u8, indices: u8, data: VexRm) -> Self {
        Self::op(1, 2, false, 2, 0x36, dst, indices, data, None)
    }
    /// vpermi2d: `dst` holds the indices and receives `(table1 ++ table2)[index]`
    pub const fn permi2d(vl: u8, dst: u8, table1: u8, table2: VexRm) -> Self {
        Self::op(1, 2, false, vl, 0x76, dst, table1, table2, None)
    }
    /// vpternlogd: `dst = f(dst, src1, src2)` for the truth table imm8 (index dst << 2 | src1 << 1 | src2)
    pub const fn pternlogd(vl: u8, dst: u8, src1: u8, src2: VexRm, imm: u8) -> Self {
        Self::op(1, 3, false, vl, 0x25, dst, src1, src2, Some(imm))
    }
    /// vpblendmd zmm {k}, zmm, zmm/m512: `k ? src2 : src1` per dword
    pub const fn pblendmd(dst: u8, src1: u8, src2: VexRm, k: u8) -> Self {
        Self { mask: k, ..Self::op(1, 2, false, 2, 0x64, dst, src1, src2, None) }
    }
    /// vpblendmw zmm {k}, zmm, zmm/m512: `k ? src2 : src1` per word
    pub const fn pblendmw(dst: u8, src1: u8, src2: VexRm, k: u8) -> Self {
        Self { mask: k, ..Self::op(1, 2, true, 2, 0x66, dst, src1, src2, None) }
    }
    /// vpbroadcastd zmm, r32
    pub const fn pbroadcastd_gpr(dst: u8, src: X86Register) -> Self {
        Self::op(1, 2, false, 2, 0x7c, dst, 0, VexRm::Reg(src as u8), None)
    }
    /// vpbroadcastq zmm, r64
    pub const fn pbroadcastq_gpr(dst: u8, src: X86Register) -> Self {
        Self::op(1, 2, true, 2, 0x7c, dst, 0, VexRm::Reg(src as u8), None)
    }
    /// vpermt2d: `dst` (the first table) receives `(dst ++ table2)[index]`
    pub const fn permt2d(vl: u8, dst: u8, index: u8, table2: VexRm) -> Self {
        Self::op(1, 2, false, vl, 0x7e, dst, index, table2, None)
    }
    /// vpshufb (byte shuffle per 128 bit lane)
    pub const fn pshufb(vl: u8, dst: u8, src: u8, mask: VexRm) -> Self {
        Self::op(1, 2, false, vl, 0x00, dst, src, mask, None)
    }
    /// vpcmpuw k, src1, src2, predicate (1: less than)
    pub const fn pcmpuw(vl: u8, k: u8, src1: u8, src2: VexRm, predicate: u8) -> Self {
        Self::op(1, 3, true, vl, 0x3e, k, src1, src2, Some(predicate))
    }
    /// vpcompressw dst {k}{z}, src (AVX512_VBMI2)
    pub const fn pcompressw(vl: u8, dst: u8, src: u8, k: u8) -> Self {
        Self { mask: k, zero: true, ..Self::op(1, 2, true, vl, 0x63, src, 0, VexRm::Reg(dst), None) }
    }
    /// Quadword shift by an immediate: opcode 0x73 /6 vpsllq, 0x73 /2 vpsrlq, 0x72 /4 vpsraq
    pub const fn shift_imm_q(vl: u8, opcode: u8, ext: u8, dst: u8, src: u8, imm: u8) -> Self {
        Self::op(1, 1, true, vl, opcode, ext, dst, VexRm::Reg(src), Some(imm))
    }
    /// vpcmp{b,w,d,q} / vpcmpu* k, src1, src2, predicate for lanes of `1 << size` bytes
    pub const fn pcmp(vl: u8, size: u8, unsigned: bool, k: u8, src1: u8, src2: VexRm, predicate: u8) -> Self {
        let opcode = if size < 2 { 0x3e } else { 0x1e } | !unsigned as u8;
        Self::op(1, 3, size & 1 != 0, vl, opcode, k, src1, src2, Some(predicate))
    }
    /// vpblendm{b,w,d,q} dst {k}, src1, src2: `k ? src2 : src1`
    pub const fn pblendm(vl: u8, size: u8, dst: u8, src1: u8, src2: VexRm, k: u8) -> Self {
        let opcode = if size < 2 { 0x66 } else { 0x64 };
        Self { mask: k, ..Self::op(1, 2, size & 1 != 0, vl, opcode, dst, src1, src2, None) }
    }
    /// vpcompress{b,w,d,q} dst {k}{z}, src (b / w: VBMI2)
    pub const fn pcompress(vl: u8, size: u8, dst: u8, src: u8, k: u8) -> Self {
        let opcode = if size < 2 { 0x63 } else { 0x8b };
        Self { mask: k, zero: true, ..Self::op(1, 2, size & 1 != 0, vl, opcode, src, 0, VexRm::Reg(dst), None) }
    }
    /// vpexpand{b,w,d,q} dst {k}{z}, src (b / w: VBMI2)
    pub const fn pexpand(vl: u8, size: u8, dst: u8, src: u8, k: u8) -> Self {
        let opcode = if size < 2 { 0x62 } else { 0x89 };
        Self { mask: k, zero: true, ..Self::op(1, 2, size & 1 != 0, vl, opcode, dst, 0, VexRm::Reg(src), None) }
    }
    /// vpmovb2m k, zmm (byte sign bits)
    pub const fn pmovb2m(k: u8, src: u8) -> Self {
        Self::op(2, 2, false, 2, 0x29, k, 0, VexRm::Reg(src), None)
    }

    /// Encodes the instruction
    pub fn encode(&self) -> ([u8; 16], usize) {
        let mut out = [0u8; 16];
        let mut len = 0;
        let mut push = |byte: u8, len: &mut usize| {
            out[*len] = byte;
            *len += 1;
        };
        let (b, x) = match self.rm {
            VexRm::Reg(r) => ((r >> 3) & 1, (r >> 4) & 1),
            VexRm::Mem(base, _) => (((base as u8) >> 3) & 1, 0),
            VexRm::Rip(_) => unreachable!(),
        };
        let not = |bit: u8| (!bit) & 1;
        push(0x62, &mut len);
        push(not(self.reg >> 3) << 7 | not(x) << 6 | not(b) << 5 | not(self.reg >> 4) << 4 | self.map, &mut len);
        push((self.w as u8) << 7 | ((!self.vvvv) & 0xf) << 3 | 1 << 2 | self.pp, &mut len);
        push((self.zero as u8) << 7 | self.vl << 5 | not(self.vvvv >> 4) << 3 | self.mask, &mut len);
        push(self.opcode, &mut len);
        let reg = (self.reg & 7) << 3;
        match self.rm {
            VexRm::Reg(r) => push(0xc0 | reg | (r & 7), &mut len),
            VexRm::Mem(base, disp) => {
                let base = (base as u8) & 7;
                let n = 16i32 << self.vl;
                let mode = if disp == 0 && base != 5 {
                    0
                } else if disp % n == 0 && (-128..=127).contains(&(disp / n)) {
                    1
                } else {
                    2
                };
                push((mode << 6) | reg | base, &mut len);
                if base == 4 {
                    push(0x24, &mut len); // SIB: no index, base = rsp / r12
                }
                match mode {
                    1 => push((disp / n) as i8 as u8, &mut len),
                    2 => disp.to_le_bytes().iter().for_each(|byte| push(*byte, &mut len)),
                    _ => {}
                }
            }
            VexRm::Rip(_) => unreachable!(),
        }
        if let Some(imm) = self.imm {
            push(imm, &mut len);
        }
        (out, len)
    }
}

#[cfg(all(test, feature = "simd-experiment"))]
mod evex_tests {
    use super::{EvexInstruction as E, VexInstruction as V, VexRm, X86Register::*};

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
    }

    /// Expected bytes from `llvm-mc --triple=x86_64 --show-encoding` (Intel syntax given in comments)
    #[test]
    fn evex_encodings_match_llvm_mc() {
        let r = VexRm::Reg;
        let cases: Vec<(E, &str)> = vec![
            (E::load(16, R11, 0), "62 c1 7e 48 6f 03"),                 // vmovdqu32 zmm16, [r11]
            (E::load(30, R11, 64 * 5), "62 41 7e 48 6f 73 05"),         // vmovdqu32 zmm30, [r11 + 320]
            (E::load(17, R11, 32), "62 c1 7e 48 6f 8b 20 00 00 00"),    // vmovdqu32 zmm17, [r11 + 32]
            (E::store(31, R11, 64 * 127), "62 41 7e 48 7f 7b 7f"),      // vmovdqu32 [r11 + 8128], zmm31
            (E::store(16, R11, 64 * 128), "62 c1 7e 48 7f 83 00 20 00 00"), // vmovdqu32 [r11 + 8192], zmm16
            (E::mov(20, 31), "62 81 7d 48 6f e7"),                      // vmovdqa32 zmm20, zmm31
            (E::alu(2, 0xfe, 16, 17, r(30)), "62 81 75 40 fe c6"),     // vpaddd zmm16, zmm17, zmm30
            (E::alu(2, 0xef, 31, 16, VexRm::Mem(R11, 128)), "62 41 7d 40 ef 7b 02"), // vpxord zmm31, zmm16, [r11 + 128]
            (E::alu(2, 0xfd, 18, 19, r(20)), "62 a1 65 40 fd d4"),     // vpaddw zmm18, zmm19, zmm20
            (E::alu38(2, 0x40, 21, 22, r(23)), "62 a2 4d 40 40 ef"),   // vpmulld zmm21, zmm22, zmm23
            (E::alu38(2, 0x15, 16, 17, r(18)), "62 a2 75 40 15 c2"),   // vprolvd zmm16, zmm17, zmm18
            (E::alu38(1, 0x15, 1, 2, r(3)), "62 f2 6d 28 15 cb"),      // vprolvd ymm1, ymm2, ymm3
            (E::alu38(2, 0x3a, 24, 25, r(26)), "62 02 35 40 3a c2"),   // vpminuw zmm24, zmm25, zmm26
            (E::shift_imm(2, 0, 16, 17, 7), "62 b1 7d 40 72 c1 07"),   // vprord zmm16, zmm17, 7
            (E::shift_imm(1, 0, 3, 14, 31), "62 d1 65 28 72 c6 1f"),   // vprord ymm3, ymm14, 31
            (E::shift_imm(2, 2, 16, 17, 16), "62 b1 7d 40 72 d1 10"),  // vpsrld zmm16, zmm17, 16
            (E::shift_imm(2, 6, 16, 17, 3), "62 b1 7d 40 72 f1 03"),   // vpslld zmm16, zmm17, 3
            (E::pshufd(16, r(17), 0x1b), "62 a1 7d 48 70 c1 1b"),      // vpshufd zmm16, zmm17, 0x1b
            (E::permd(16, 17, r(18)), "62 a2 75 40 36 c2"),            // vpermd zmm16, zmm17, zmm18
            (E::permi2d(2, 16, 17, r(18)), "62 a2 75 40 76 c2"),       // vpermi2d zmm16, zmm17, zmm18
            (E::permi2d(1, 0, 1, r(2)), "62 f2 75 28 76 c2"),          // vpermi2d ymm0, ymm1, ymm2
            (E::pternlogd(2, 16, 17, r(18), 0x96), "62 a3 75 40 25 c2 96"), // vpternlogd zmm16, zmm17, zmm18, 0x96
            (E::pternlogd(1, 5, 14, r(9), 0xd2), "62 d3 0d 28 25 e9 d2"),   // vpternlogd ymm5, ymm14, ymm9, 0xd2
            (E::pblendmd(16, 17, r(18), 1), "62 a2 75 41 64 c2"),      // vpblendmd zmm16 {k1}, zmm17, zmm18
            (E::pblendmw(16, 17, r(18), 1), "62 a2 f5 41 66 c2"),      // vpblendmw zmm16 {k1}, zmm17, zmm18
            (E::pbroadcastd_gpr(16, RAX), "62 e2 7d 48 7c c0"),        // vpbroadcastd zmm16, eax
            (E::pbroadcastq_gpr(30, R9), "62 42 fd 48 7c f1"),         // vpbroadcastq zmm30, r9
            (E::pmovb2m(1, 30), "62 92 7e 48 29 ce"),                  // vpmovb2m k1, zmm30
            (E::alu_q(2, 1, 0x6c, 24, 16, r(17)), "62 21 fd 40 6c c1"), // vpunpcklqdq zmm24, zmm16, zmm17
            (E::alu_q(2, 1, 0x6d, 31, 22, r(23)), "62 21 cd 40 6d ff"), // vpunpckhqdq zmm31, zmm22, zmm23
            (E::alu_q(2, 2, 0x15, 16, 17, VexRm::Mem(R11, 64)), "62 c2 f5 40 15 43 01"), // vprolvq zmm16, zmm17, [r11 + 64]
            (E::alu_q(1, 2, 0x15, 1, 2, r(3)), "62 f2 ed 28 15 cb"),   // vprolvq ymm1, ymm2, ymm3
            (E::shufi64x2(16, 24, 26, 0x88), "62 83 bd 40 43 c2 88"),  // vshufi64x2 zmm16, zmm24, zmm26, 0x88
            (E::permt2d(2, 16, 17, r(18)), "62 a2 75 40 7e c2"),       // vpermt2d zmm16, zmm17, zmm18
            (E::permt2d(1, 3, 4, r(5)), "62 f2 5d 28 7e dd"),          // vpermt2d ymm3, ymm4, ymm5
            (E::pshufb(2, 16, 17, VexRm::Mem(R11, 128)), "62 c2 75 40 00 43 02"), // vpshufb zmm16, zmm17, [r11 + 128]
            (E::pcmpuw(2, 1, 16, r(17), 1), "62 b3 fd 40 3e c9 01"),   // vpcmpltuw k1, zmm16, zmm17
            (E::pcmpuw(1, 1, 3, r(4), 1), "62 f3 e5 28 3e cc 01"),     // vpcmpltuw k1, ymm3, ymm4
            (E::pcompressw(2, 18, 16, 1), "62 a2 fd c9 63 c2"),        // vpcompressw zmm18 {k1} {z}, zmm16
            (E::pcompressw(1, 2, 3, 1), "62 f2 fd a9 63 da"),          // vpcompressw ymm2 {k1} {z}, ymm3
            (E::alu_q(2, 1, 0xd4, 16, 17, r(18)), "62 a1 f5 40 d4 c2"), // vpaddq zmm16, zmm17, zmm18
            (E::alu_q(2, 1, 0xf4, 16, 17, r(18)), "62 a1 f5 40 f4 c2"), // vpmuludq zmm16, zmm17, zmm18
            (E::alu_q(2, 2, 0x40, 16, 17, r(18)), "62 a2 f5 40 40 c2"), // vpmullq zmm16, zmm17, zmm18
            (E::alu_q(1, 2, 0x40, 1, 2, r(3)), "62 f2 ed 28 40 cb"),   // vpmullq ymm1, ymm2, ymm3
            (E::alu_q(2, 2, 0xb4, 16, 17, r(18)), "62 a2 f5 40 b4 c2"), // vpmadd52luq zmm16, zmm17, zmm18
            (E::alu_q(2, 2, 0xb5, 16, 17, r(18)), "62 a2 f5 40 b5 c2"), // vpmadd52huq zmm16, zmm17, zmm18
            (E::shift_imm_q(2, 0x73, 6, 16, 17, 5), "62 b1 fd 40 73 f1 05"), // vpsllq zmm16, zmm17, 5
            (E::shift_imm_q(2, 0x73, 2, 16, 17, 5), "62 b1 fd 40 73 d1 05"), // vpsrlq zmm16, zmm17, 5
            (E::shift_imm_q(2, 0x72, 4, 16, 17, 5), "62 b1 fd 40 72 e1 05"), // vpsraq zmm16, zmm17, 5
            (E::shift_imm_q(1, 0x72, 4, 1, 2, 63), "62 f1 f5 28 72 e2 3f"),  // vpsraq ymm1, ymm2, 63
            (E::pcmp(2, 0, false, 1, 16, r(17), 6), "62 b3 7d 40 3f c9 06"), // vpcmpnleb k1, zmm16, zmm17
            (E::pcmp(2, 0, true, 1, 16, r(17), 1), "62 b3 7d 40 3e c9 01"),  // vpcmpltub k1, zmm16, zmm17
            (E::pcmp(2, 1, false, 1, 16, r(17), 0), "62 b3 fd 40 3f c9 00"), // vpcmpeqw k1, zmm16, zmm17
            (E::pcmp(2, 2, false, 1, 16, r(17), 2), "62 b3 7d 40 1f c9 02"), // vpcmpled k1, zmm16, zmm17
            (E::pcmp(1, 3, true, 1, 3, r(4), 4), "62 f3 e5 28 1e cc 04"),    // vpcmpnequq k1, ymm3, ymm4
            (E::pblendm(2, 0, 16, 17, r(18), 1), "62 a2 75 41 66 c2"),  // vpblendmb zmm16 {k1}, zmm17, zmm18
            (E::pblendm(2, 3, 16, 17, r(18), 1), "62 a2 f5 41 64 c2"),  // vpblendmq zmm16 {k1}, zmm17, zmm18
            (E::pcompress(2, 0, 18, 16, 1), "62 a2 7d c9 63 c2"),      // vpcompressb zmm18 {k1} {z}, zmm16
            (E::pcompress(2, 2, 18, 16, 1), "62 a2 7d c9 8b c2"),      // vpcompressd zmm18 {k1} {z}, zmm16
            (E::pcompress(1, 3, 2, 3, 1), "62 f2 fd a9 8b da"),        // vpcompressq ymm2 {k1} {z}, ymm3
            (E::pexpand(2, 1, 18, 16, 1), "62 a2 fd c9 62 d0"),        // vpexpandw zmm18 {k1} {z}, zmm16
            (E::pexpand(2, 2, 18, 16, 1), "62 a2 7d c9 89 d0"),        // vpexpandd zmm18 {k1} {z}, zmm16
            (E::op(1, 1, false, 2, 0x71, 6, 16, r(17), Some(3)), "62 b1 7d 40 71 f1 03"), // vpsllw zmm16, zmm17, 3
            (E::op(1, 1, false, 2, 0x71, 4, 16, r(17), Some(3)), "62 b1 7d 40 71 e1 03"), // vpsraw zmm16, zmm17, 3
            (E::op(1, 2, true, 2, 0x12, 16, 17, r(18), None), "62 a2 f5 40 12 c2"),     // vpsllvw zmm16, zmm17, zmm18
            (E::op(1, 2, true, 2, 0x11, 16, 17, r(18), None), "62 a2 f5 40 11 c2"),     // vpsravw zmm16, zmm17, zmm18
            (E::op(1, 2, true, 2, 0x46, 16, 17, r(18), None), "62 a2 f5 40 46 c2"),     // vpsravq zmm16, zmm17, zmm18
            (E::op(1, 1, true, 2, 0x72, 0, 16, r(17), Some(9)), "62 b1 fd 40 72 c1 09"), // vprorq zmm16, zmm17, 9
            (E::op(1, 1, false, 2, 0x60, 16, 17, r(18), None), "62 a1 75 40 60 c2"),    // vpunpcklbw zmm16, zmm17, zmm18
            (E::op(1, 1, true, 2, 0x6d, 16, 17, r(18), None), "62 a1 f5 40 6d c2"),     // vpunpckhqdq zmm16, zmm17, zmm18
            (E::op(1, 2, false, 2, 0x2b, 16, 17, r(18), None), "62 a2 75 40 2b c2"),    // vpackusdw zmm16, zmm17, zmm18
            (E::op(1, 2, false, 2, 0x30, 16, 0, r(17), None), "62 a2 7d 48 30 c1"),     // vpmovzxbw zmm16, ymm17
            (E::op(1, 2, false, 2, 0x25, 16, 0, r(17), None), "62 a2 7d 48 25 c1"),     // vpmovsxdq zmm16, ymm17
            (E::op(1, 2, false, 2, 0x54, 16, 0, r(17), None), "62 a2 7d 48 54 c1"),     // vpopcntb zmm16, zmm17
            (E::op(1, 2, true, 2, 0x55, 16, 0, r(17), None), "62 a2 fd 48 55 c1"),      // vpopcntq zmm16, zmm17
            (E::op(1, 2, false, 2, 0x44, 16, 0, r(17), None), "62 a2 7d 48 44 c1"),     // vplzcntd zmm16, zmm17
            (E::op(1, 2, false, 2, 0x8d, 16, 17, r(18), None), "62 a2 75 40 8d c2"),    // vpermb zmm16, zmm17, zmm18
            (E::op(1, 2, false, 2, 0x75, 16, 17, r(18), None), "62 a2 75 40 75 c2"),    // vpermi2b zmm16, zmm17, zmm18
            (E::op(1, 2, true, 2, 0x83, 16, 17, r(18), None), "62 a2 f5 40 83 c2"),     // vpmultishiftqb zmm16, zmm17, zmm18
            (E::op(1, 3, true, 2, 0xce, 16, 17, r(18), Some(0x63)), "62 a3 f5 40 ce c2 63"), // vgf2p8affineqb zmm16, zmm17, zmm18, 0x63
            (E::op(1, 2, false, 2, 0xcf, 16, 17, r(18), None), "62 a2 75 40 cf c2"),    // vgf2p8mulb zmm16, zmm17, zmm18
            (E::op(1, 2, false, 2, 0xdc, 16, 17, r(18), None), "62 a2 75 40 dc c2"),    // vaesenc zmm16, zmm17, zmm18
            (E::op(1, 2, false, 2, 0xdf, 16, 17, r(18), None), "62 a2 75 40 df c2"),    // vaesdeclast zmm16, zmm17, zmm18
            (E::op(1, 3, false, 2, 0x44, 16, 17, r(18), Some(0x11)), "62 a3 75 40 44 c2 11"), // vpclmulqdq zmm16, zmm17, zmm18, 0x11
            (E::op(1, 2, true, 2, 0x39, 16, 17, r(18), None), "62 a2 f5 40 39 c2"),     // vpminsq zmm16, zmm17, zmm18
            (E::op(1, 2, true, 2, 0x3f, 16, 17, r(18), None), "62 a2 f5 40 3f c2"),     // vpmaxuq zmm16, zmm17, zmm18
            (E::op(1, 1, false, 2, 0xea, 16, 17, r(18), None), "62 a1 75 40 ea c2"),    // vpminsw zmm16, zmm17, zmm18
            (E::op(1, 2, true, 2, 0x1f, 16, 0, r(17), None), "62 a2 fd 48 1f c1"),      // vpabsq zmm16, zmm17
            (E::op(1, 1, false, 2, 0xdc, 16, 17, r(18), None), "62 a1 75 40 dc c2"),    // vpaddusb zmm16, zmm17, zmm18
            (E::op(1, 2, true, 2, 0x28, 16, 17, r(18), None), "62 a2 f5 40 28 c2"),     // vpmuldq zmm16, zmm17, zmm18
            (E::op(1, 2, false, 2, 0x0b, 16, 17, r(18), None), "62 a2 75 40 0b c2"),    // vpmulhrsw zmm16, zmm17, zmm18
            (E::op(2, 2, false, 2, 0x30, 17, 0, r(16), None), "62 a2 7e 48 30 c8"),     // vpmovwb ymm16, zmm17
            (E::op(2, 2, false, 2, 0x35, 17, 0, r(16), None), "62 a2 7e 48 35 c8"),     // vpmovqd ymm16, zmm17
            (E::op(2, 2, false, 1, 0x32, 3, 0, r(4), None), "62 f2 7e 28 32 dc"),       // vpmovqb xmm4, ymm3
            (E::op(1, 3, true, 2, 0x00, 16, 0, r(17), Some(0x1b)), "62 a3 fd 48 00 c1 1b"), // vpermq zmm16, zmm17, 0x1b
            (E::op(1, 2, true, 2, 0x8d, 16, 17, r(18), None), "62 a2 f5 40 8d c2"),     // vpermw zmm16, zmm17, zmm18
            (E::op(1, 3, false, 2, 0x0f, 16, 17, r(18), Some(5)), "62 a3 75 40 0f c2 05"), // vpalignr zmm16, zmm17, zmm18, 5
            (E::op(3, 1, false, 2, 0x70, 16, 0, r(17), Some(0x1b)), "62 a1 7f 48 70 c1 1b"), // vpshuflw zmm16, zmm17, 0x1b
            (E::op(2, 1, false, 2, 0x70, 16, 0, r(17), Some(0x1b)), "62 a1 7e 48 70 c1 1b"), // vpshufhw zmm16, zmm17, 0x1b
            (E::op(1, 2, true, 2, 0xc4, 16, 0, r(17), None), "62 a2 fd 48 c4 c1"),      // vpconflictq zmm16, zmm17
            (E::op(1, 2, false, 2, 0x26, 1, 16, r(17), None), "62 b2 7d 40 26 c9"),     // vptestmb k1, zmm16, zmm17
            (E::op(2, 2, true, 2, 0x27, 1, 16, r(17), None), "62 b2 fe 40 27 c9"),      // vptestnmq k1, zmm16, zmm17
            (E::op(1, 2, false, 2, 0x50, 16, 17, r(18), None), "62 a2 75 40 50 c2"),    // vpdpbusd zmm16, zmm17, zmm18
            (E::op(1, 2, false, 2, 0x52, 16, 17, r(18), None), "62 a2 75 40 52 c2"),    // vpdpwssd zmm16, zmm17, zmm18
            (E::op(1, 1, false, 2, 0xf6, 16, 17, r(18), None), "62 a1 75 40 f6 c2"),    // vpsadbw zmm16, zmm17, zmm18
            (E::op(1, 3, true, 2, 0x71, 16, 17, r(18), Some(9)), "62 a3 f5 40 71 c2 09"), // vpshldq zmm16, zmm17, zmm18, 9
            (E::op(1, 3, true, 2, 0x72, 16, 17, r(18), Some(3)), "62 a3 f5 40 72 c2 03"), // vpshrdw zmm16, zmm17, zmm18, 3
            (E::op(1, 2, false, 2, 0x71, 16, 17, r(18), None), "62 a2 75 40 71 c2"),    // vpshldvd zmm16, zmm17, zmm18
            (E::op(1, 2, true, 2, 0x73, 16, 17, r(18), None), "62 a2 f5 40 73 c2"),     // vpshrdvq zmm16, zmm17, zmm18
        ];
        for (insn, expected) in cases {
            let (bytes, len) = insn.encode();
            assert_eq!(hex(&bytes[..len]), expected);
        }
        let vex: Vec<(V, &str)> = vec![
            (V::kmovq_to_gpr(R8, 1), "c4 61 fb 93 c1"),                // kmovq r8, k1
            (V::kmovd_to_gpr(RAX, 1), "c5 fb 93 c1"),                  // kmovd eax, k1
            (V::kmovd_to_gpr(R14, 1), "c5 7b 93 f1"),                  // kmovd r14d, k1
            (V::kmov_from_gpr(1, RDX, false), "c5 fb 92 ca"),          // kmovd k1, edx
            (V::kmov_from_gpr(1, R9, true), "c4 c1 fb 92 c9"),         // kmovq k1, r9
            (V::psllq(1, 2, 7), "c5 f5 73 f2 07"),                     // vpsllq ymm1, ymm2, 7
            (V::alu(0xd4, 1, 2, VexRm::Reg(3)), "c5 ed d4 cb"),        // vpaddq ymm1, ymm2, ymm3
            (V::alu(0xf4, 1, 2, VexRm::Reg(3)), "c5 ed f4 cb"),        // vpmuludq ymm1, ymm2, ymm3
            (V::psrad(1, 2, 31), "c5 f5 72 e2 1f"),                    // vpsrad ymm1, ymm2, 31
            (V::movmskps(RAX, 3, false), "c5 fc 50 c3"),               // vmovmskps eax, ymm3
            (V::movmskps(R9, 14, true), "c4 41 7d 50 ce"),             // vmovmskpd r9d, ymm14
            (V::permq(1, 2, 0xd8), "c4 e3 fd 00 ca d8"),               // vpermq ymm1, ymm2, 0xd8
            (V::pblendvb(1, 2, 3, 15), "c4 e3 6d 4c cb f0"),           // vpblendvb ymm1, ymm2, ymm3, ymm15
            (V::pbroadcastw(15, 15), "c4 42 7d 79 ff"),                // vpbroadcastw ymm15, xmm15
            (V::pmovzxbd(15, R11, 0), "c4 42 7d 31 3b"),               // vpmovzxbd ymm15, qword ptr [r11]
            (V::alu38(0x37, 1, 2, VexRm::Reg(3)), "c4 e2 6d 37 cb"),   // vpcmpgtq ymm1, ymm2, ymm3
            (V::alu38(0x29, 1, 2, VexRm::Reg(3)), "c4 e2 6d 29 cb"),   // vpcmpeqq ymm1, ymm2, ymm3
            (V::alu(0x63, 1, 1, VexRm::Reg(1)), "c5 f5 63 c9"),        // vpacksswb ymm1, ymm1, ymm1
            (V::ymm(1, 1, 0x71, 2, 1, VexRm::Reg(2), Some(5)), "c5 f5 71 d2 05"), // vpsrlw ymm1, ymm2, 5
            (V::ymm(1, 2, 0x46, 1, 2, VexRm::Reg(3), None), "c4 e2 6d 46 cb"),  // vpsravd ymm1, ymm2, ymm3
            (V::ymm(1, 2, 0x2b, 1, 2, VexRm::Reg(3), None), "c4 e2 6d 2b cb"),  // vpackusdw ymm1, ymm2, ymm3
            (V::ymm(1, 2, 0x33, 1, 0, VexRm::Reg(2), None), "c4 e2 7d 33 ca"),  // vpmovzxwd ymm1, xmm2
            (V::ymm(1, 1, 0x73, 3, 15, VexRm::Reg(2), Some(8)), "c5 85 73 da 08"), // vpsrldq ymm15, ymm2, 8
            (V::ymm(1, 2, 0x04, 1, 2, VexRm::Reg(3), None), "c4 e2 6d 04 cb"),  // vpmaddubsw ymm1, ymm2, ymm3
            (V::ymm(1, 1, 0xf6, 1, 2, VexRm::Reg(3), None), "c5 ed f6 cb"),     // vpsadbw ymm1, ymm2, ymm3
            (V::ymm(1, 2, 0xdc, 1, 2, VexRm::Reg(3), None), "c4 e2 6d dc cb"),  // vaesenc ymm1, ymm2, ymm3
            (V::ymm(1, 3, 0x44, 1, 2, VexRm::Reg(3), Some(0x10)), "c4 e3 6d 44 cb 10"), // vpclmulqdq ymm1, ymm2, ymm3, 0x10
            (V::ymm(1, 2, 0x38, 1, 2, VexRm::Reg(3), None), "c4 e2 6d 38 cb"),  // vpminsb ymm1, ymm2, ymm3
            (V::ymm(1, 2, 0x1e, 1, 0, VexRm::Reg(2), None), "c4 e2 7d 1e ca"),  // vpabsd ymm1, ymm2
            (V::ymm(3, 1, 0x70, 1, 0, VexRm::Reg(2), Some(0x1b)), "c5 ff 70 ca 1b"), // vpshuflw ymm1, ymm2, 0x1b
            (V::ymm(2, 1, 0x70, 1, 0, VexRm::Reg(2), Some(0x1b)), "c5 fe 70 ca 1b"), // vpshufhw ymm1, ymm2, 0x1b
            (V::ymm(1, 3, 0x0f, 1, 2, VexRm::Reg(3), Some(7)), "c4 e3 6d 0f cb 07"), // vpalignr ymm1, ymm2, ymm3, 7
            (V { l: false, ..V::alu(0x61, 1, 2, VexRm::Reg(15)) }, "c4 c1 69 61 cf"), // vpunpcklwd xmm1, xmm2, xmm15
            (V::movq_from_gpr(15, R9), "c4 41 f9 6e f9"),              // vmovq xmm15, r9
            (V::movq_to_gpr(RAX, 3), "c4 e1 f9 7e d8"),                // vmovq rax, xmm3
            (V::pextrq(R10, 15, 1), "c4 43 f9 16 fa 01"),              // vpextrq r10, xmm15, 1
            (V::extracti128(15, 3, 1), "c4 c3 7d 39 df 01"),           // vextracti128 xmm15, ymm3, 1
            (V::pbroadcastq(2, 15), "c4 c2 7d 59 d7"),                 // vpbroadcastq ymm2, xmm15
            (V::pmovmskb(RCX, 14), "c4 c1 7d d7 ce"),                  // vpmovmskb ecx, ymm14
            (V::blendvps(7, 4, 5, 6), "c4 e3 5d 4a fd 60"),            // vblendvps ymm7, ymm4, ymm5, ymm6
            (V::alu38w(0x47, 1, 2, VexRm::Reg(3)), "c4 e2 ed 47 cb"),  // vpsllvq ymm1, ymm2, ymm3
            (V::alu38w(0x45, 12, 12, VexRm::Reg(15)), "c4 42 9d 45 e7"), // vpsrlvq ymm12, ymm12, ymm15
            (V::psrlq(12, 3, 1), "c5 9d 73 d3 01"),                    // vpsrlq ymm12, ymm3, 1
        ];
        for (insn, expected) in vex {
            let (bytes, len) = insn.encode(std::ptr::null());
            assert_eq!(hex(&bytes[..len]), expected);
        }
    }
}
