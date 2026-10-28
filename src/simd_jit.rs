//! AVX2 / AVX-512 lowering of the experimental vector instructions (child module of the JIT)
//!
//! Slots are cached in YMM registers (ymm0..ymm14). With AVX-512, slot pairs used by 512 bit
//! operations are cached in a second set (zmm16..zmm30); a slot is never cached in both: an
//! operation of the other width writes it back and drops it first. Without AVX-512, 512 bit
//! operations run as two 256 bit halves.

#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::{
    simd::{self, VecInsn},
    vm::SimdJitStrategy,
    x86::{EvexInstruction as E, VexInstruction as V, VexRm},
};

/// ZMM registers of the slot pair cache (zmm16..zmm30), zmm31 is scratch
const ZREGS: usize = 15;
const ZBASE: u8 = 16;
const SCRATCH_ZMM: u8 = 31;
/// Opmask register used by blends and vmovmsk8
const KMASK: u8 = 1;

// Layout of the constant tables (ANCHOR_SIMD_TABLES) of the AVX2 lowerings of the AVX-512 operations
/// Sign bit of every lane, 32 bytes per lane size (8, 16, 32, 64 bit)
const TAB_SIGN: usize = 0;
/// Bit `i` in lane `i` (lane `i % 8` for bytes), 32 bytes per lane size
const TAB_BITS: usize = 128;
/// vpshufb indices spreading mask byte `i / 8` to byte lane `i`
const TAB_BSHUF: usize = 256;
/// Lane indices as u32: 0..8, then 0, 0, 1, 1, .. (u64 lanes)
const TAB_IOTA: usize = 288;
/// 2^26 - 1 and 2^52 - 1 in every u64 lane
const TAB_M26: usize = 352;
const TAB_M52: usize = 384;
/// vpermd indices (u8) per mask: compress / expand of u32 lanes (256 entries), u64 lanes (16)
const TAB_C32: usize = 416;
const TAB_E32: usize = TAB_C32 + 2048;
const TAB_C64: usize = TAB_E32 + 2048;
const TAB_E64: usize = TAB_C64 + 128;
/// popcnt: nibble counts (vpshufb table), 0x0f bytes, 1 in every byte, 1 in every u16
const TAB_NIB: usize = TAB_E64 + 128;
const TAB_0F: usize = TAB_NIB + 32;
const TAB_ONE8: usize = TAB_0F + 32;
const TAB_ONE16: usize = TAB_ONE8 + 32;
/// Truncation: vpshufb indices packing the low bytes of every lane into the low bytes of its 128 bit
/// lane, per (from, to) in the order of TRUNC_PAIRS
const TAB_TRUNC: usize = TAB_ONE16 + 32;
const TRUNC_PAIRS: [(u8, u8); 6] = [(1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2)];
/// 64 in every u64 lane
const TAB_C64Q: usize = TAB_TRUNC + 6 * 32;
/// Index masks 0x1f, 0x3f, 0x7f in every byte (byte table lookups of 2, 4, 8 lanes)
const TAB_BMASK: usize = TAB_C64Q + 32;
/// u16 constants: 0x0100, 15, 31, 16, 0x00ff
const TAB_W0100: usize = TAB_BMASK + 96;
const TAB_W15: usize = TAB_W0100 + 32;
const TAB_W31: usize = TAB_W15 + 32;
const TAB_C16W: usize = TAB_W31 + 32;
const TAB_W00FF: usize = TAB_C16W + 32;
/// Parity of the nibble (vpshufb table), 0x1b in every byte, 0xff in every u64
const TAB_PARITY: usize = TAB_W00FF + 32;
const TAB_1B: usize = TAB_PARITY + 32;
const TAB_FFQ: usize = TAB_1B + 32;
/// Bit k in every byte, 8 rows
const TAB_BYTEBIT: usize = TAB_FFQ + 32;
/// gf2p8affine: vpshufb indices broadcasting byte 7 - i of every u64 to its bytes, 8 rows
const TAB_AFFIDX: usize = TAB_BYTEBIT + 8 * 32;
/// conflict: u32 lane rotations by k (vpermd indices) and the bits 1 << (i - k), k = 1..8; the
/// bits for u64 lanes, k = 1..4
const TAB_CONF_IDX32: usize = TAB_AFFIDX + 8 * 32;
const TAB_CONF_BIT32: usize = TAB_CONF_IDX32 + 7 * 32;
const TAB_CONF_BIT64: usize = TAB_CONF_BIT32 + 7 * 32;
/// 0x0000ffff / 0xffff0000 in every u32
const TAB_LO16: usize = TAB_CONF_BIT64 + 3 * 32;
const TAB_HI16: usize = TAB_LO16 + 32;
/// !63 in every u64, zero (Rosetta 2 workaround)
const TAB_NOT63Q: usize = TAB_HI16 + 32;
const TAB_ZERO: usize = TAB_NOT63Q + 32;
const TAB_END: usize = TAB_ZERO + 32;
/// Upper bound of all constants emitted after the subroutines (shuffle masks, opmask tables, tables)
pub(super) const CONSTANTS_SIZE: usize = TAB_END + 4096;

/// Host YMM registers available to the register cache (ymm15 is reserved as scratch)
const CACHE_REGS: usize = 15;
const SCRATCH_YMM: u8 = 15;

const VPADDD: u8 = 0xfe;
const VPSUBD: u8 = 0xfa;
const VPMULLD: u8 = 0x40;
const VPMINUD: u8 = 0x3b;
const VPSLLVD: u8 = 0x47;
const VPSRLVD: u8 = 0x45;
const VPANDN: u8 = 0xdf;
const VPXOR: u8 = 0xef;
const VPADDW: u8 = 0xfd;
const VPSUBW: u8 = 0xf9;
const VPMULLW: u8 = 0xd5;
const VPMULHW: u8 = 0xe5;
const VPMADDWD: u8 = 0xf5;
const VPMINUW: u8 = 0x3a;
const VPAND: u8 = 0xdb;
const VPOR: u8 = 0xeb;
const VPUNPCKLDQ: u8 = 0x62;
const VPUNPCKHDQ: u8 = 0x6a;
const VPUNPCKLQDQ: u8 = 0x6c;
const VPUNPCKHQDQ: u8 = 0x6d;

/// Static statistics of the generated vector code
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimdJitStats {
    /// Guest vector instructions compiled
    pub vector_instructions: usize,
    /// Native instructions emitted (whole program)
    pub native_instructions: usize,
    /// Native 256 bit loads (vmovdqu ymm, m256), excluding memory source operands
    pub vector_loads: usize,
    /// Native 256 bit stores (vmovdqu m256, ymm)
    pub vector_stores: usize,
    /// ALU instructions which read a slot directly from memory (memory source operand)
    pub memory_operands: usize,
    /// Dirty slots written back because their register was evicted
    pub spills: usize,
    /// Slots loaded again after having been evicted within the same basic block
    pub reloads: usize,
    /// Operand accesses served from a cached register
    pub cache_hits: usize,
    /// Registers evicted to make room
    pub evictions: usize,
    /// Dirty slots written back at block boundaries, calls, memory accesses, meter checkpoints
    pub flush_stores: usize,
    /// Runtime vload / vstore transfers (calls into the memory mapping)
    pub translations: usize,
    /// Intrinsic calls whose operands weren't known at compile time (regular syscall)
    pub intrinsic_slow_paths: usize,
    /// vload256 from a compile time known read-only address (translated at compile time)
    pub static_loads: usize,
}

#[derive(Clone, Copy, Default)]
struct SlotState {
    reg: Option<u8>,
    dirty: bool,
    evicted: bool,
}

#[derive(Clone, Copy, Default)]
struct PairState {
    /// ZMM register (number)
    reg: Option<u8>,
    dirty: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum R11 {
    Unknown,
    VectorBase,
}

/// Compile time state of the vector lowering
pub struct SimdJitState {
    pub(super) enabled: bool,
    cached: bool,
    /// Basic block starts (jump and call targets)
    jump_targets: Vec<bool>,
    /// R11 contents can be tracked across instructions (no callx into the middle of blocks)
    static_flow: bool,
    /// Translated by Rosetta 2, which shifts by the low bits of counts above 63 in vpsllvq /
    /// vpsrlvq instead of giving 0 (x86-64 macOS builds only)
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    rosetta: bool,
    slots: [SlotState; ebpf::VECTOR_SLOTS],
    regs: [Option<u8>; CACHE_REGS],
    last_use: [u64; CACHE_REGS],
    clock: u64,
    pinned: [bool; CACHE_REGS],
    /// AVX-512 lowering (512 bit operations in ZMM registers, AVX-512VL single instructions)
    pub(super) avx512: bool,
    /// Every operation through the reference implementation (Config::simd_force_reference)
    force_reference: bool,
    /// AVX512_VBMI2 (vpcompressw) in addition
    vbmi2: bool,
    /// AVX512DQ (vpmullq) and AVX512_IFMA (vpmadd52*) in addition
    dq: bool,
    ifma: bool,
    /// AVX-512 CD, VBMI, BITALG, VPOPCNTDQ, GFNI (EVEX forms)
    cd: bool,
    vbmi: bool,
    bitalg: bool,
    vpopcntdq: bool,
    gfni: bool,
    /// VPCLMULQDQ: 256 bit VEX form on any host which has it, 512 bit form with AVX-512
    vpclmul: bool,
    /// AVX512_VNNI
    vnni: bool,
    pairs: [PairState; ebpf::VECTOR_SLOTS / 2],
    zregs: [Option<u8>; ZREGS],
    zlast_use: [u64; ZREGS],
    zpinned: [bool; ZREGS],
    r11: R11,
    /// The previous instruction left R11 intact
    previous_keeps_r11: bool,
    /// `(static call identifier, sub-operation, wide)` of the vector intrinsics
    intrinsics: Vec<(u32, u8, bool)>,
    /// Block local constant tracking of the guest registers (for the intrinsic operands)
    known: [Option<u64>; 11],
    /// The previously compiled instruction, its effect on `known` is applied lazily
    previous: Option<ebpf::Insn>,
    pub(super) stats: SimdJitStats,
}

impl SimdJitState {
    pub(super) fn disabled() -> Self {
        Self {
            enabled: false,
            cached: false,
            jump_targets: Vec::new(),
            static_flow: false,
            #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
            rosetta: false,
            slots: [SlotState::default(); ebpf::VECTOR_SLOTS],
            regs: [None; CACHE_REGS],
            last_use: [0; CACHE_REGS],
            clock: 0,
            pinned: [false; CACHE_REGS],
            avx512: false,
            force_reference: false,
            vbmi2: false,
            dq: false,
            ifma: false,
            cd: false,
            vbmi: false,
            bitalg: false,
            vpopcntdq: false,
            gfni: false,
            vpclmul: false,
            vnni: false,
            pairs: [PairState::default(); ebpf::VECTOR_SLOTS / 2],
            zregs: [None; ZREGS],
            zlast_use: [0; ZREGS],
            zpinned: [false; ZREGS],
            r11: R11::Unknown,
            previous_keeps_r11: false,
            intrinsics: Vec::new(),
            known: [None; 11],
            previous: None,
            stats: SimdJitStats::default(),
        }
    }

    /// Scans the program, returns the state and the number of vector instructions
    pub(super) fn new<C: ContextObject>(
        executable: &Executable<C>,
        program: &[u8],
        config: &Config,
    ) -> Result<(Self, usize), EbpfError> {
        let mut state = Self::disabled();
        if !config.enable_simd_experiment {
            return Ok((state, 0));
        }
        let _ = program;
        let analysis = executable.simd_analysis();
        let vector_instructions = analysis.vector_instructions;
        if vector_instructions > 0
            && !config.simd_assume_avx2
            && !std::arch::is_x86_feature_detected!("avx2")
        {
            return Err(EbpfError::SimdHostUnsupported);
        }
        if vector_instructions == 0 {
            // Programs without vector instructions compile exactly like without the experiment
            return Ok((state, 0));
        }
        state.enabled = true;
        state.avx512 = config.simd_force_avx512
            || (config.simd_avx512
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl"));
        state.force_reference = config.simd_force_reference;
        let (avx512, force) = (state.avx512, config.simd_force_avx512);
        let enabled = |bit: u32| config.simd_disabled_features & bit == 0;
        let feature = |bit: u32, detected: bool| avx512 && enabled(bit) && (force || detected);
        use simd::features::*;
        state.vbmi2 = feature(VBMI2, std::arch::is_x86_feature_detected!("avx512vbmi2"));
        state.dq = feature(DQ, std::arch::is_x86_feature_detected!("avx512dq"));
        state.ifma = feature(IFMA, std::arch::is_x86_feature_detected!("avx512ifma"));
        state.cd = feature(CD, std::arch::is_x86_feature_detected!("avx512cd"));
        state.vbmi = feature(VBMI, std::arch::is_x86_feature_detected!("avx512vbmi"));
        state.bitalg = feature(BITALG, std::arch::is_x86_feature_detected!("avx512bitalg"));
        state.vpopcntdq = feature(
            VPOPCNTDQ,
            std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
        );
        state.gfni = feature(GFNI, std::arch::is_x86_feature_detected!("gfni"));
        state.vnni = feature(VNNI, std::arch::is_x86_feature_detected!("avx512vnni"));
        // VEX forms: also without AVX-512 (e.g. Zen 3)
        state.vpclmul = !config.simd_force_reference
            && enabled(VPCLMULQDQ)
            && (state.avx512 && force || std::arch::is_x86_feature_detected!("vpclmulqdq"));
        // callx may only enter instructions outside runs of vector code (see simd::SimdAnalysis),
        // so block local state is sound even in programs which contain callx
        state.static_flow = true;
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        {
            state.rosetta = simd::running_under_rosetta();
        }
        state.cached = config.simd_jit_strategy == SimdJitStrategy::RegisterCached;
        state.jump_targets = analysis.jump_targets.clone();
        state.intrinsics = simd::intrinsics::hashes();
        // Code size units of 1024 bytes: one per vector instruction, a scalar loop over up to 64 lanes
        // (about 45 bytes each) four more
        Ok((state, vector_instructions + 4 * analysis.lane_loops))
    }
}

use simd::is_pure_alu;

impl<'a, C: ContextObject> JitCompiler<'a, C> {
    /// Emits a VEX encoded instruction
    fn emit_vex(&mut self, instruction: V) {
        let address = unsafe {
            self.result
                .text_section()
                .as_ptr()
                .add(self.offset_in_text_section)
        };
        let (bytes, len) = instruction.encode(address);
        for byte in &bytes[..len] {
            self.emit::<u8>(*byte);
        }
        self.simd.stats.native_instructions += 1;
        match instruction.opcode {
            0x6f if matches!(instruction.rm, VexRm::Mem(..)) => self.simd.stats.vector_loads += 1,
            0x7f => self.simd.stats.vector_stores += 1,
            _ if matches!(instruction.rm, VexRm::Mem(..)) => self.simd.stats.memory_operands += 1,
            _ => {}
        }
    }

    /// Emits an EVEX encoded instruction
    fn emit_evex(&mut self, instruction: E) {
        let (bytes, len) = instruction.encode();
        for byte in &bytes[..len] {
            self.emit::<u8>(*byte);
        }
        self.simd.stats.native_instructions += 1;
        match instruction.opcode {
            0x6f if matches!(instruction.rm, VexRm::Mem(..)) => self.simd.stats.vector_loads += 1,
            0x7f => self.simd.stats.vector_stores += 1,
            _ if matches!(instruction.rm, VexRm::Mem(..)) => self.simd.stats.memory_operands += 1,
            _ => {}
        }
    }

    /// Emits vzeroupper (before calling host code, which may use legacy SSE)
    pub(super) fn emit_vzeroupper(&mut self) {
        if self.simd.enabled {
            self.emit::<u8>(0xc5);
            self.emit::<u8>(0xf8);
            self.emit::<u8>(0x77);
            self.simd.stats.native_instructions += 1;
        }
    }

    /// Byte shuffle masks for the 8 and 16 bit rotations
    pub(super) fn emit_simd_constants(&mut self) {
        if !self.simd.enabled {
            return;
        }
        for (anchor, rotation) in [(ANCHOR_SIMD_ROT16, 2u8), (ANCHOR_SIMD_ROT8, 1u8)] {
            self.set_anchor(anchor);
            for i in 0..32u8 {
                let dword = i & !3;
                self.emit::<u8>(dword | ((i + rotation) & 3));
            }
        }
        for (anchor, value) in [
            (ANCHOR_SIMD_C32, 32u32),
            (ANCHOR_SIMD_ONES, u32::MAX),
            (ANCHOR_SIMD_C31, 31),
        ] {
            self.set_anchor(anchor);
            for _ in 0..8 {
                self.emit::<u32>(value);
            }
        }
        self.set_anchor(ANCHOR_SIMD_C63Q);
        for _ in 0..4 {
            self.emit::<u64>(63);
        }
        self.set_anchor(ANCHOR_SIMD_TABLES);
        let start = self.offset_in_text_section;
        let lane = |jit: &mut Self, size: u32, v: u64| match size {
            0 => jit.emit::<u8>(v as u8),
            1 => jit.emit::<u16>(v as u16),
            2 => jit.emit::<u32>(v as u32),
            _ => jit.emit::<u64>(v),
        };
        for size in 0..4u32 {
            for _ in 0..32 >> size {
                lane(self, size, 1u64 << (8 * (1 << size) - 1));
            }
        }
        for size in 0..4u32 {
            for i in 0..32 >> size {
                lane(
                    self,
                    size,
                    if size == 0 {
                        1u64 << (i % 8)
                    } else {
                        1u64 << i
                    },
                );
            }
        }
        for i in 0..32u8 {
            self.emit::<u8>(i / 8 % 2 + 2 * (i / 16));
        }
        for i in 0..8u32 {
            self.emit::<u32>(i);
        }
        for i in 0..8u32 {
            self.emit::<u32>(i / 2);
        }
        for m in [(1u64 << 26) - 1, (1u64 << 52) - 1] {
            for _ in 0..4 {
                self.emit::<u64>(m);
            }
        }
        // u32 lanes: compress (indices of the set bits) and expand (rank of each set bit)
        for expand in [false, true] {
            for m in 0..256u32 {
                let idx: Vec<u8> = if expand {
                    (0..8)
                        .map(|i| {
                            if (m >> i) & 1 != 0 {
                                (m & ((1 << i) - 1)).count_ones() as u8
                            } else {
                                0
                            }
                        })
                        .collect()
                } else {
                    let set: Vec<u8> = (0..8).filter(|i| (m >> i) & 1 != 0).collect();
                    (0..8).map(|i| set.get(i).copied().unwrap_or(0)).collect()
                };
                idx.iter().for_each(|b| self.emit::<u8>(*b));
            }
        }
        // u64 lanes as pairs of u32 indices
        for expand in [false, true] {
            for m in 0..16u32 {
                let lanes: Vec<u8> = if expand {
                    (0..4)
                        .map(|i| {
                            if (m >> i) & 1 != 0 {
                                (m & ((1 << i) - 1)).count_ones() as u8
                            } else {
                                0
                            }
                        })
                        .collect()
                } else {
                    let set: Vec<u8> = (0..4).filter(|i| (m >> i) & 1 != 0).collect();
                    (0..4).map(|i| set.get(i).copied().unwrap_or(0)).collect()
                };
                lanes.iter().for_each(|l| {
                    self.emit::<u8>(2 * l);
                    self.emit::<u8>(2 * l + 1);
                });
            }
        }
        for i in 0..32u32 {
            self.emit::<u8>((i % 16).count_ones() as u8);
        }
        for _ in 0..32 {
            self.emit::<u8>(0x0f);
        }
        for _ in 0..32 {
            self.emit::<u8>(1);
        }
        for _ in 0..16 {
            self.emit::<u16>(1);
        }
        for (from, to) in TRUNC_PAIRS {
            let (fs, ts) = (1usize << from, 1usize << to);
            for _ in 0..2 {
                for j in 0..16 {
                    let kept = 16 / fs * ts;
                    self.emit::<u8>(if j < kept {
                        ((j / ts) * fs + j % ts) as u8
                    } else {
                        0x80
                    });
                }
            }
        }
        for _ in 0..4 {
            self.emit::<u64>(64);
        }
        for m in [0x1fu8, 0x3f, 0x7f] {
            for _ in 0..32 {
                self.emit::<u8>(m);
            }
        }
        for v in [0x0100u16, 15, 31, 16, 0x00ff] {
            for _ in 0..16 {
                self.emit::<u16>(v);
            }
        }
        for i in 0..32u32 {
            self.emit::<u8>(((i % 16).count_ones() & 1) as u8);
        }
        for _ in 0..32 {
            self.emit::<u8>(0x1b);
        }
        for _ in 0..4 {
            self.emit::<u64>(0xff);
        }
        for k in 0..8 {
            for _ in 0..32 {
                self.emit::<u8>(1 << k);
            }
        }
        for i in 0..8u8 {
            for j in 0..32u8 {
                self.emit::<u8>(((j % 16) & !7) + 7 - i);
            }
        }
        for k in 1..8u32 {
            for i in 0..8u32 {
                self.emit::<u32>((i + 8 - k) % 8);
            }
        }
        for k in 1..8u32 {
            for i in 0..8u32 {
                self.emit::<u32>(if i >= k { 1 << (i - k) } else { 0 });
            }
        }
        for k in 1..4u32 {
            for i in 0..4u32 {
                self.emit::<u64>(if i >= k { 1 << (i - k) } else { 0 });
            }
        }
        for v in [0x0000_ffffu32, 0xffff_0000] {
            for _ in 0..8 {
                self.emit::<u32>(v);
            }
        }
        for v in [!63u64, 0] {
            for _ in 0..4 {
                self.emit::<u64>(v);
            }
        }
        debug_assert_eq!(self.offset_in_text_section - start, TAB_END);
        if self.simd.avx512 {
            // Opmasks of the 512 bit blends: the imm8 repeated for every 8 lanes
            self.set_anchor(ANCHOR_SIMD_BLENDD_MASKS);
            for imm in 0..256u32 {
                self.emit::<u16>((imm | imm << 8) as u16);
            }
            self.set_anchor(ANCHOR_SIMD_BLENDW_MASKS);
            for imm in 0..256u32 {
                self.emit::<u32>(imm * 0x0101_0101);
            }
        }
    }

    /// Called before every guest instruction is compiled (before its pc offset is recorded)
    pub(super) fn simd_before_instruction(&mut self, insn: &ebpf::Insn) {
        if !self.simd.enabled {
            return;
        }
        let is_target = self.simd.jump_targets.get(self.pc).copied().unwrap_or(true);
        if let Some(previous) = self.simd.previous.take() {
            self.simd_track_constants(&previous);
        }
        if !self.simd.previous_keeps_r11 {
            // The previous scalar instruction may have used R11
            self.simd.r11 = R11::Unknown;
        }
        if is_target {
            // Fall-through into a block start, jumps arrive with an empty cache
            self.simd_flush(true);
            self.simd.r11 = R11::Unknown;
            self.simd.known = [None; 11];
        }
        let vector = insn.opc == ebpf::VOP || self.simd_is_intrinsic(insn);
        // `mov64 dst, imm32` only writes `dst`, even when the immediate is sanitized
        self.simd.previous_keeps_r11 = vector || insn.opc == ebpf::MOV64_IMM;
        if !vector && !is_pure_alu(insn.opc) {
            self.simd_flush(true);
        } else if insn.opc == ebpf::LD_DW_IMM && !self.executable.get_sbpf_version().disable_lddw()
        {
            // lddw validates the instruction meter, which may throw: write back like a checkpoint
            self.simd_flush(false);
        }
        self.simd.previous = Some(insn.clone());
    }

    /// `(sub-operation, wide)` if the instruction is a vector intrinsic call
    fn simd_intrinsic(&self, insn: &ebpf::Insn) -> Option<(u8, bool)> {
        if insn.opc != ebpf::CALL_IMM || insn.src != 0 {
            return None;
        }
        self.simd
            .intrinsics
            .iter()
            .find(|(hash, _, _)| *hash == insn.imm as u32)
            .map(|(_, subop, wide)| (*subop, *wide))
    }

    fn simd_is_intrinsic(&self, insn: &ebpf::Insn) -> bool {
        self.simd_intrinsic(insn).is_some()
    }

    /// Applies the effect of an instruction on the known register constants
    fn simd_track_constants(&mut self, insn: &ebpf::Insn) {
        let class = insn.opc & ebpf::BPF_CLS_MASK;
        let dst = insn.dst as usize;
        match insn.opc {
            ebpf::MOV64_IMM if dst < 11 => self.simd.known[dst] = Some(insn.imm as u64),
            ebpf::LD_DW_IMM if dst < 11 && !self.executable.get_sbpf_version().disable_lddw() => {
                let mut full = insn.clone();
                ebpf::augment_lddw_unchecked(self.program, &mut full);
                self.simd.known[dst] = Some(full.imm as u64);
            }
            ebpf::ADD64_IMM if dst < 11 => {
                self.simd.known[dst] = self.simd.known[dst].map(|v| v.wrapping_add(insn.imm as u64))
            }
            ebpf::MOV64_REG if dst < 11 => {
                self.simd.known[dst] = self.simd.known.get(insn.src as usize).copied().flatten()
            }
            ebpf::CALL_IMM if self.simd_is_intrinsic(insn) => {
                let (subop, _) = self.simd_intrinsic(insn).unwrap();
                self.simd.known[0] = if simd::writes_dst(subop) {
                    None
                } else {
                    Some(0)
                };
            }
            // Syscalls write r0, internal calls may change anything
            ebpf::CALL_IMM | ebpf::CALL_REG => self.simd.known = [None; 11],
            ebpf::VOP => {
                if let Some(op) = VecInsn::decode(insn) {
                    if simd::writes_dst(op.subop) {
                        self.simd.known[op.dst as usize] = None;
                    }
                }
            }
            ebpf::EXIT => {}
            _ if class == ebpf::BPF_JMP64 || class == ebpf::BPF_JMP32 => {}
            _ => {
                // Conservatively treat everything else as writing dst (stores only read it)
                if dst < 11 {
                    self.simd.known[dst] = None;
                }
            }
        }
    }

    /// Codegen of a vector intrinsic call (SIMD-0645 style): inline AVX2 if the packed operands
    /// in r1 are a known valid constant, otherwise a regular syscall to `slow_path`
    pub(crate) fn emit_simd_intrinsic(
        &mut self,
        subop: u8,
        wide: bool,
        slow_path: BuiltinFunction<C>,
    ) {
        let op = if self.simd.enabled {
            self.simd.known[simd::intrinsics::OPERAND_REG as usize]
                .and_then(|packed| simd::intrinsics::unpack(subop, wide, packed))
        } else {
            None
        };
        if let Some(op) = op {
            if self.emit_simd_op(&op).is_ok() {
                // The intrinsic returns its scalar result (written by the op) or 0 in r0
                if !simd::writes_dst(op.subop) {
                    self.emit_ins(X86Instruction::load_immediate(REGISTER_MAP[0], 0));
                }
                return;
            }
        }
        if self.simd.enabled {
            // Host calls clobber all YMM registers and read or write the vector file in memory
            self.simd_flush(true);
            self.simd.stats.intrinsic_slow_paths += 1;
        }
        self.emit_external_call(slow_path);
        if self.simd.enabled {
            self.simd.r11 = R11::Unknown;
        }
    }

    /// `vload` from a compile time known address in the read-only section (e.g. a table in
    /// .rodata): the section is immutable and always mapped at its address, so the translation is
    /// done here instead of at runtime, and the load leaves the register cache intact. Returns false
    /// (nothing emitted) if the address isn't known or the bytes aren't all in the section.
    fn emit_simd_static_load(&mut self, op: &VecInsn) -> bool {
        let Some(base) = self.simd.known[op.src as usize] else {
            return false;
        };
        let addr = base.wrapping_add(op.off as i64 as u64);
        let slots = op.transfer_slots();
        let ro = self.executable.get_ro_region();
        let Some(host) = ro.vm_to_host_buffer(addr, (slots * ebpf::VECTOR_WIDTH) as u64) else {
            return false;
        };
        let host = host.ptr().cast::<u8>() as i64;
        self.simd.stats.static_loads += 1;
        // Validates the vector base like every vector instruction
        self.simd_ensure_base();
        for i in 0..slots {
            self.simd_drop_pair((op.vd as usize + i) as u8 / 2);
        }
        self.simd.pinned = [false; CACHE_REGS];
        for i in 0..slots {
            // The destination is allocated first: an eviction stores relative to the vector base
            let rd = self.simd_destination(op.vd + i as u8);
            self.emit_ins(X86Instruction::load_immediate(
                REGISTER_SCRATCH,
                host + (i * ebpf::VECTOR_WIDTH) as i64,
            ));
            self.simd.r11 = R11::Unknown;
            self.emit_vex(V::load(rd, REGISTER_SCRATCH, 0));
        }
        true
    }

    /// Called before an instruction meter checkpoint, which may throw
    pub(super) fn simd_before_checkpoint(&mut self) {
        if self.simd.enabled {
            self.simd_flush(false);
            // The checkpoint loads the pc into R11
            self.simd.r11 = R11::Unknown;
        }
    }

    /// Called before the per instruction trace call (a Rust call: clobbers the vector registers and
    /// R11)
    #[cfg(feature = "tracer")]
    pub(super) fn simd_before_trace(&mut self) {
        if self.simd.enabled {
            self.simd_flush(true);
            self.simd.r11 = R11::Unknown;
        }
    }

    /// Called before the final bumper which throws ExecutionOverrun
    pub(super) fn simd_before_end(&mut self) {
        if self.simd.enabled {
            self.simd_flush(true);
        }
    }

    fn simd_ensure_base(&mut self) {
        if self.simd.r11 == R11::VectorBase {
            return;
        }
        self.emit_ins(X86Instruction::load(
            OperandSize::S64,
            REGISTER_PTR_TO_VM,
            REGISTER_SCRATCH,
            X86IndirectAccess::Offset(self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase)),
        ));
        // if vector_base == 0 { throw UnsupportedInstruction }
        self.emit_ins(X86Instruction::test(
            OperandSize::S64,
            REGISTER_SCRATCH,
            REGISTER_SCRATCH,
            None,
        ));
        // jnz rel32, emitted raw so that no diversification noop can sneak in before the patch
        self.emit::<u8>(0x0f);
        self.emit::<u8>(0x85);
        self.emit::<i32>(0);
        let patch = self.offset_in_text_section;
        self.emit_ins(X86Instruction::load_immediate(
            REGISTER_SCRATCH,
            self.pc as i64,
        ));
        self.emit_ins(X86Instruction::jump_immediate(
            self.relative_to_anchor(ANCHOR_CALL_UNSUPPORTED_INSTRUCTION, 5),
        ));
        let distance = (self.offset_in_text_section - patch) as i32;
        unsafe {
            let ptr = self.result.text_section_mut().as_mut_ptr().add(patch - 4);
            std::ptr::write_unaligned(ptr.cast::<i32>(), distance);
        }
        self.emit_ins(X86Instruction::load(
            OperandSize::S64,
            REGISTER_PTR_TO_VM,
            REGISTER_SCRATCH,
            X86IndirectAccess::Offset(self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase)),
        ));
        // Reset at the end of the instruction if R11 can not be tracked across instructions
        self.simd.r11 = R11::VectorBase;
    }

    fn slot_disp(slot: u8) -> i32 {
        slot as i32 * ebpf::VECTOR_WIDTH as i32
    }

    /// Writes back all dirty cached slots, optionally forgetting all cached registers
    fn simd_flush(&mut self, invalidate: bool) {
        if self.simd.pairs.iter().any(|pair| pair.dirty) {
            self.simd_ensure_base();
            for pair in 0..ebpf::VECTOR_SLOTS / 2 {
                let state = self.simd.pairs[pair];
                if state.dirty {
                    self.emit_evex(E::store(
                        state.reg.unwrap(),
                        REGISTER_SCRATCH,
                        Self::slot_disp(2 * pair as u8),
                    ));
                    self.simd.stats.flush_stores += 1;
                    self.simd.pairs[pair].dirty = false;
                }
            }
        }
        if invalidate {
            self.simd.pairs = [PairState::default(); ebpf::VECTOR_SLOTS / 2];
            self.simd.zregs = [None; ZREGS];
        }
        let dirty = self.simd.slots.iter().any(|slot| slot.dirty);
        if dirty {
            self.simd_ensure_base();
            for slot in 0..ebpf::VECTOR_SLOTS {
                let state = self.simd.slots[slot];
                if state.dirty {
                    let reg = state.reg.unwrap();
                    self.emit_vex(V::store(reg, REGISTER_SCRATCH, Self::slot_disp(slot as u8)));
                    self.simd.stats.flush_stores += 1;
                    self.simd.slots[slot].dirty = false;
                }
            }
        }
        if invalidate {
            for slot in self.simd.slots.iter_mut() {
                *slot = SlotState::default();
            }
            self.simd.regs = [None; CACHE_REGS];
        }
    }

    fn simd_touch(&mut self, reg: u8) {
        self.simd.clock += 1;
        self.simd.last_use[reg as usize] = self.simd.clock;
    }

    /// Picks a free or the least recently used unpinned register, writing back its slot if dirty
    fn simd_allocate(&mut self) -> u8 {
        let mut best = None;
        for reg in 0..CACHE_REGS {
            if self.simd.pinned[reg] {
                continue;
            }
            if self.simd.regs[reg].is_none() {
                best = Some(reg);
                break;
            }
            if best.is_none_or(|b: usize| self.simd.last_use[reg] < self.simd.last_use[b]) {
                best = Some(reg);
            }
        }
        let reg = best.expect("all vector registers pinned");
        if let Some(victim) = self.simd.regs[reg] {
            let state = self.simd.slots[victim as usize];
            if state.dirty {
                self.simd_ensure_base();
                self.emit_vex(V::store(
                    reg as u8,
                    REGISTER_SCRATCH,
                    Self::slot_disp(victim),
                ));
                self.simd.stats.spills += 1;
            }
            self.simd.slots[victim as usize] = SlotState {
                reg: None,
                dirty: false,
                evicted: true,
            };
            self.simd.stats.evictions += 1;
            self.simd.regs[reg] = None;
        }
        self.simd_touch(reg as u8);
        reg as u8
    }

    fn simd_pin(&mut self, reg: u8) {
        self.simd.pinned[reg as usize] = true;
    }

    /// Register holding `slot`, loading it if necessary
    fn simd_ensure_reg(&mut self, slot: u8) -> u8 {
        if let Some(reg) = self.simd.slots[slot as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_touch(reg);
            return reg;
        }
        let reg = self.simd_allocate();
        self.simd_ensure_base();
        self.emit_vex(V::load(reg, REGISTER_SCRATCH, Self::slot_disp(slot)));
        if self.simd.slots[slot as usize].evicted {
            self.simd.stats.reloads += 1;
        }
        self.simd.slots[slot as usize] = SlotState {
            reg: Some(reg),
            dirty: false,
            evicted: false,
        };
        self.simd.regs[reg as usize] = Some(slot);
        reg
    }

    /// Register or memory operand for a read-only source
    fn simd_source(&mut self, slot: u8) -> VexRm {
        if let Some(reg) = self.simd.slots[slot as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_touch(reg);
            self.simd_pin(reg);
            VexRm::Reg(reg)
        } else {
            self.simd_ensure_base();
            VexRm::Mem(REGISTER_SCRATCH, Self::slot_disp(slot))
        }
    }

    /// Register which will receive the new value of `slot`
    fn simd_destination(&mut self, slot: u8) -> u8 {
        let reg = match self.simd.slots[slot as usize].reg {
            Some(reg) => reg,
            None => {
                let reg = self.simd_allocate();
                self.simd.regs[reg as usize] = Some(slot);
                reg
            }
        };
        self.simd.slots[slot as usize] = SlotState {
            reg: Some(reg),
            dirty: true,
            evicted: false,
        };
        self.simd_touch(reg);
        reg
    }

    /// Emits the rotation `dst = src.rotate_right(amount)` (uses ymm15 as temporary)
    fn simd_emit_rotate(&mut self, dst: u8, src: u8, amount: u8) {
        match amount {
            0 => {
                if dst != src {
                    self.emit_vex(V::mov(dst, src));
                }
            }
            16 | 8 => {
                let anchor = if amount == 16 {
                    ANCHOR_SIMD_ROT16
                } else {
                    ANCHOR_SIMD_ROT8
                };
                self.emit_vex(V::pshufb(dst, src, VexRm::Rip(self.anchors[anchor])));
            }
            _ => {
                self.emit_vex(V::psrld(SCRATCH_YMM, src, amount));
                self.emit_vex(V::pslld(dst, src, 32 - amount));
                self.emit_vex(V::alu(VPOR, dst, dst, VexRm::Reg(SCRATCH_YMM)));
            }
        }
    }

    /// Lane-wise binary instruction of a sub-operation
    fn simd_binary(subop: u8, rd: u8, ra: u8, b: VexRm, imm: u8) -> V {
        use crate::simd::*;
        match subop {
            VADD32 => V::alu(VPADDD, rd, ra, b),
            VSUB32 => V::alu(VPSUBD, rd, ra, b),
            VMULLO32 => V::alu38(VPMULLD, rd, ra, b),
            VMINU32 => V::alu38(VPMINUD, rd, ra, b),
            VSLLV32 => V::alu38(VPSLLVD, rd, ra, b),
            VSRLV32 => V::alu38(VPSRLVD, rd, ra, b),
            VANDN => V::alu(VPANDN, rd, ra, b),
            VXOR => V::alu(VPXOR, rd, ra, b),
            VAND => V::alu(VPAND, rd, ra, b),
            VOR => V::alu(VPOR, rd, ra, b),
            VADD16 => V::alu(VPADDW, rd, ra, b),
            VSUB16 => V::alu(VPSUBW, rd, ra, b),
            VMULLO16 => V::alu(VPMULLW, rd, ra, b),
            VMULHI16 => V::alu(VPMULHW, rd, ra, b),
            VMADD16 => V::alu(VPMADDWD, rd, ra, b),
            VMINU16 => V::alu38(VPMINUW, rd, ra, b),
            VBLENDW => V::pblendw(rd, ra, b, imm),
            VBLENDD => V::pblendd(rd, ra, b, imm),
            _ => unreachable!(),
        }
    }

    /// `rd = ra.rotate_left(rb & 31)` per lane, using `ra >> (32 - c) = (ra >> 1) >> (c ^ 31)`. `rd`
    /// may alias `ra` or `rb`, `t` is a free temporary (uses ymm15).
    fn simd_emit_rolv(&mut self, rd: u8, ra: u8, rb: u8, t: u8) {
        let c31 = VexRm::Rip(self.anchors[ANCHOR_SIMD_C31]);
        self.emit_vex(V::load_rip(SCRATCH_YMM, self.anchors[ANCHOR_SIMD_C31]));
        self.emit_vex(V::alu(VPAND, SCRATCH_YMM, SCRATCH_YMM, VexRm::Reg(rb)));
        self.emit_vex(V::alu38(VPSLLVD, t, ra, VexRm::Reg(SCRATCH_YMM)));
        self.emit_vex(V::alu(VPXOR, SCRATCH_YMM, SCRATCH_YMM, c31));
        self.emit_vex(V::psrld(rd, ra, 1));
        self.emit_vex(V::alu38(VPSRLVD, rd, rd, VexRm::Reg(SCRATCH_YMM)));
        self.emit_vex(V::alu(VPOR, rd, rd, VexRm::Reg(t)));
    }

    /// ymm15 = g(x, y) for the two input truth table `tt` (indexed by `x << 1 | y`)
    fn simd_emit_tt2(&mut self, tt: u8, x: u8, y: u8) {
        let s = SCRATCH_YMM;
        let ones = VexRm::Rip(self.anchors[ANCHOR_SIMD_ONES]);
        let (op, invert) = match tt & 0xf {
            0x0 => {
                self.emit_vex(V::alu(VPXOR, s, s, VexRm::Reg(s)));
                return;
            }
            0xf => {
                self.emit_vex(V::pcmpeqd(s, s, VexRm::Reg(s)));
                return;
            }
            0xc | 0x3 => {
                self.emit_vex(V::mov(s, x));
                if tt == 0x3 {
                    self.emit_vex(V::alu(VPXOR, s, s, ones));
                }
                return;
            }
            0xa | 0x5 => {
                self.emit_vex(V::mov(s, y));
                if tt == 0x5 {
                    self.emit_vex(V::alu(VPXOR, s, s, ones));
                }
                return;
            }
            0x8 => ((VPAND, x, y), false),
            0x7 => ((VPAND, x, y), true),
            0xe => ((VPOR, x, y), false),
            0x1 => ((VPOR, x, y), true),
            0x6 => ((VPXOR, x, y), false),
            0x9 => ((VPXOR, x, y), true),
            0x2 => ((VPANDN, x, y), false),
            0xd => ((VPANDN, x, y), true),
            0x4 => ((VPANDN, y, x), false),
            _ => ((VPANDN, y, x), true), // 0xb
        };
        self.emit_vex(V::alu(op.0, s, op.1, VexRm::Reg(op.2)));
        if invert {
            self.emit_vex(V::alu(VPXOR, s, s, ones));
        }
    }

    /// `rd = f(rd, ra, rb)` for the truth table `imm` (vpternlogd semantics, uses ymm15). Aliased
    /// operands are folded into a two input function.
    fn simd_emit_ternlog(&mut self, rd: u8, ra: u8, rb: u8, imm: u8) {
        let f = |d: u8, a: u8, b: u8| (imm >> (d << 2 | a << 1 | b)) & 1;
        let tt2 = |g: &dyn Fn(u8, u8) -> u8| (0..4u8).fold(0, |t, i| t | g(i >> 1, i & 1) << i);
        if rd == ra || rd == rb {
            let (tt, other) = if rd == ra {
                (tt2(&|x, y| f(x, x, y)), rb)
            } else {
                (tt2(&|x, y| f(x, y, x)), ra)
            };
            self.simd_emit_tt2(tt, rd, other);
            self.emit_vex(V::mov(rd, SCRATCH_YMM));
            return;
        }
        // f = g_lo(a, b) ^ (d & (g_lo ^ g_hi)(a, b))
        let (lo, hi) = (imm & 0xf, imm >> 4);
        let h = lo ^ hi;
        if h == 0 {
            self.simd_emit_tt2(lo, ra, rb);
            self.emit_vex(V::mov(rd, SCRATCH_YMM));
            return;
        }
        if h != 0xf {
            self.simd_emit_tt2(h, ra, rb);
            self.emit_vex(V::alu(VPAND, rd, rd, VexRm::Reg(SCRATCH_YMM)));
        }
        if lo != 0 {
            self.simd_emit_tt2(lo, ra, rb);
            self.emit_vex(V::alu(VPXOR, rd, rd, VexRm::Reg(SCRATCH_YMM)));
        }
    }

    /// Emits the 8x8 transpose of ymm0..ymm7, results in [ymm8..ymm14, ymm0]
    fn simd_emit_transpose_regs(&mut self) -> [u8; 8] {
        let r = |n: u8| VexRm::Reg(n);
        for (i, (lo, hi)) in [(0u8, 1u8), (2, 3), (4, 5), (6, 7)].iter().enumerate() {
            let i = i as u8;
            self.emit_vex(V::alu(VPUNPCKLDQ, 8 + 2 * i, *lo, r(*hi)));
            self.emit_vex(V::alu(VPUNPCKHDQ, 9 + 2 * i, *lo, r(*hi)));
        }
        // abcd_04, abcd_15, abcd_26, abcd_37, efgh_04, efgh_15, efgh_26, efgh_37 in ymm0..ymm7
        self.emit_vex(V::alu(VPUNPCKLQDQ, 0, 8, r(10)));
        self.emit_vex(V::alu(VPUNPCKHQDQ, 1, 8, r(10)));
        self.emit_vex(V::alu(VPUNPCKLQDQ, 2, 9, r(11)));
        self.emit_vex(V::alu(VPUNPCKHQDQ, 3, 9, r(11)));
        self.emit_vex(V::alu(VPUNPCKLQDQ, 4, 12, r(14)));
        self.emit_vex(V::alu(VPUNPCKHQDQ, 5, 12, r(14)));
        self.emit_vex(V::alu(VPUNPCKLQDQ, 6, 13, r(15)));
        self.emit_vex(V::alu(VPUNPCKHQDQ, 7, 13, r(15)));
        self.emit_vex(V::perm2i128(8, 0, r(4), 0x20));
        self.emit_vex(V::perm2i128(12, 0, r(4), 0x31));
        self.emit_vex(V::perm2i128(9, 1, r(5), 0x20));
        self.emit_vex(V::perm2i128(13, 1, r(5), 0x31));
        self.emit_vex(V::perm2i128(10, 2, r(6), 0x20));
        self.emit_vex(V::perm2i128(14, 2, r(6), 0x31));
        self.emit_vex(V::perm2i128(11, 3, r(7), 0x20));
        self.emit_vex(V::perm2i128(0, 3, r(7), 0x31));
        [8, 9, 10, 11, 12, 13, 14, 0]
    }

    /// Compiles one vector instruction
    pub(super) fn emit_simd_instruction(&mut self, insn: &ebpf::Insn) -> Result<(), EbpfError> {
        let Some(op) = VecInsn::decode(insn) else {
            return Err(EbpfError::UnsupportedInstruction);
        };
        self.emit_simd_op(&op)
    }

    /// Compiles one decoded vector operation (opcode or intrinsic call form)
    fn emit_simd_op(&mut self, op: &VecInsn) -> Result<(), EbpfError> {
        use crate::simd::*;
        if self.offset_in_text_section + 4096 >= self.result.text_section().len() {
            return Err(EbpfError::ExhaustedTextSegment(self.pc));
        }
        let op = *op;
        self.simd.stats.vector_instructions += 1;
        if matches!(op.subop, VLOADM | VSTOREM) {
            self.emit_simd_transfer_masked(&op);
            return Ok(());
        }
        if self.simd.force_reference && !matches!(op.subop, VLOAD | VSTORE) {
            self.emit_simd_fallback(&op);
            return Ok(());
        }
        if self.simd.avx512
            && !self.simd_avx512_native(&op)
            && !matches!(op.subop, VLOAD | VSTORE | VTRANSPOSE8 | VTRANSPOSE64)
        {
            // The AVX2 lowering on an AVX-512 host (an extension is missing): YMM cache only
            let (slots, count) = Self::simd_operand_slots(&op);
            for &slot in &slots[..count] {
                // (in usize: slot 255 + 1 must not wrap to an empty range)
                for s in (slot as usize..slot as usize + op.halves()).map(|s| s as u8) {
                    self.simd_drop_pair(s / 2);
                }
            }
        }
        if self.simd.avx512
            && self.simd_avx512_native(&op)
            && !matches!(op.subop, VLOAD | VSTORE | VTRANSPOSE8 | VTRANSPOSE64)
        {
            // A slot is cached at one width only
            let (slots, count) = Self::simd_operand_slots(&op);
            for &slot in &slots[..count] {
                if op.wide {
                    self.simd_drop_slot(slot);
                    self.simd_drop_slot(slot + 1);
                } else {
                    self.simd_drop_pair(slot / 2);
                }
            }
            if op.wide {
                self.emit_simd_zmm(&op);
            } else {
                self.emit_simd_cached(&op);
            }
            if !self.simd.cached {
                self.simd_flush(true);
            }
            if !self.simd.static_flow {
                self.simd.r11 = R11::Unknown;
            }
            return Ok(());
        }
        match op.subop {
            VLOAD if self.emit_simd_static_load(&op) => {}
            VLOAD | VSTORE => self.emit_simd_transfer(&op),
            VTRANSPOSE8 | VTRANSPOSE64 => self.emit_simd_fixed(&op),
            VPERMI2D | VPERMT2D => self.emit_simd_cross(&op),
            // AVX2 templates for 32 / 64 bit lanes; otherwise compress runs as a branchless scalar loop
            // and expand through the reference implementation
            VCOMPRESS if op.wide || op.off < 2 => self.emit_simd_compress_scalar(&op),
            VEXPAND if op.wide || op.off < 2 => self.emit_simd_expand_scalar(&op),
            VCMP | VBLENDM if op.wide => self.emit_simd_mask_split(&op),
            VZEXT | VSEXT if op.wide => self.emit_simd_widen_split(&op),
            VCONFLICT if op.wide => self.emit_simd_fallback(&op),
            VTRUNC if op.wide => self.emit_simd_trunc_split(&op),
            VTESTM if op.wide => self.emit_simd_testm_split(&op),
            VCLMUL if !self.simd.vpclmul => self.emit_simd_fallback(&op),
            VPERMB | VPERMI2B | VPERMW => self.emit_simd_byte_table(&op),
            VPERMD if op.wide => self.emit_simd_cross(&op),
            VMOVMSK8 if op.wide => self.emit_simd_movmsk_wide(&op),
            _ if op.wide => {
                // Lane-wise on the two halves. All slots are even, so the low half's result never
                // overwrites a source of the high half.
                for h in 0..2u8 {
                    // mont16 also takes q from a slot (in `off`)
                    let off = if op.subop == VMONT16 {
                        op.off + h as i16
                    } else {
                        op.off
                    };
                    let half = VecInsn {
                        wide: false,
                        vd: op.vd + h,
                        va: op.va + h,
                        vb: op.vb + h,
                        off,
                        ..op
                    };
                    self.emit_simd_cached(&half);
                }
            }
            _ => self.emit_simd_cached(&op),
        }
        if !self.simd.cached {
            // Strategy A: nothing stays in registers across vector instructions
            self.simd_flush(true);
        }
        if !self.simd.static_flow {
            self.simd.r11 = R11::Unknown;
        }
        Ok(())
    }

    /// vload / vstore: one call which translates the guest range and copies all slots
    fn emit_simd_transfer(&mut self, op: &VecInsn) {
        // Host calls clobber all YMM registers, and vstore reads the vector file in memory
        self.simd_flush(true);
        let store = op.subop == simd::VSTORE;
        let slot = if store { op.va } else { op.vd };
        let spec = store as u64 | (slot as u64) << 8 | (op.transfer_slots() as u64) << 16;
        self.emit_ins(X86Instruction::load_immediate(
            REGISTER_SCRATCH,
            op.off as i64,
        ));
        self.emit_ins(X86Instruction::alu(
            OperandSize::S64,
            0x01,
            REGISTER_MAP[op.src as usize],
            REGISTER_SCRATCH,
            None,
        ));
        self.simd.stats.translations += 1;
        self.emit_rust_call(
            Value::Constant64(simd::jit_transfer as *const u8 as i64, false),
            &[
                // First, the other arguments may overwrite it
                Argument {
                    index: 3,
                    value: Value::Register(REGISTER_SCRATCH),
                },
                Argument {
                    index: 4,
                    value: Value::Constant64(spec as i64, false),
                },
                Argument {
                    index: 2,
                    value: Value::RegisterIndirect(
                        REGISTER_PTR_TO_VM,
                        self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase),
                        false,
                    ),
                },
                Argument {
                    index: 1,
                    value: Value::RegisterIndirect(
                        REGISTER_PTR_TO_VM,
                        self.slot_in_vm(RuntimeEnvironmentSlot::MemoryMapping),
                        false,
                    ),
                },
                Argument {
                    index: 0,
                    value: Value::RegisterPlusConstant32(
                        REGISTER_PTR_TO_VM,
                        self.slot_in_vm(RuntimeEnvironmentSlot::ProgramResult),
                        false,
                    ),
                },
            ],
            None,
        );
        self.emit_check_result_is_ok();
        self.emit_ins(X86Instruction::load_immediate(
            REGISTER_SCRATCH,
            self.pc as i64,
        ));
        self.emit_ins(X86Instruction::conditional_jump_immediate(
            0x85,
            self.relative_to_anchor(ANCHOR_THROW_EXCEPTION, 6),
        ));
        self.simd.r11 = R11::Unknown;
    }

    /// Makes `slot` resident in `reg` (dirty), after a flush emptied the cache
    fn simd_assign(&mut self, slot: u8, reg: u8) {
        self.simd.slots[slot as usize] = SlotState {
            reg: Some(reg),
            dirty: true,
            evicted: false,
        };
        self.simd.regs[reg as usize] = Some(slot);
        self.simd_touch(reg);
    }

    /// Transposes: flush, load the matrix into fixed registers, transpose, leave the results resident
    /// and dirty
    fn emit_simd_fixed(&mut self, op: &VecInsn) {
        use crate::simd::*;
        self.simd_flush(true);
        self.simd_ensure_base();
        let load = |jit: &mut Self, reg: u8, slot: u8| {
            jit.emit_vex(V::load(reg, REGISTER_SCRATCH, Self::slot_disp(slot)))
        };
        match (op.subop, op.wide) {
            (VTRANSPOSE8, _) => {
                for row in 0..8u8 {
                    load(self, row, op.vd + row);
                }
                let out = self.simd_emit_transpose_regs();
                for (row, reg) in out.iter().enumerate() {
                    self.simd_assign(op.vd + row as u8, *reg);
                }
            }
            (VTRANSPOSE64, false) => {
                for row in 0..4u8 {
                    load(self, row, op.vd + row);
                }
                self.simd_emit_transpose4q([0, 1, 2, 3], [4, 5, 6, 7]);
                for row in 0..4u8 {
                    self.simd_assign(op.vd + row, row);
                }
            }
            (VTRANSPOSE64, true) if self.simd.avx512 => {
                // r0..r7 in zmm16..23, t in zmm24..31, u in zmm16..23, columns in zmm24..31
                for row in 0..8u8 {
                    self.emit_evex(E::load(
                        16 + row,
                        REGISTER_SCRATCH,
                        Self::slot_disp(op.vd + 2 * row),
                    ));
                }
                for i in 0..4u8 {
                    self.emit_evex(E::alu_q(
                        2,
                        1,
                        VPUNPCKLQDQ,
                        24 + 2 * i,
                        16 + 2 * i,
                        VexRm::Reg(17 + 2 * i),
                    ));
                    self.emit_evex(E::alu_q(
                        2,
                        1,
                        VPUNPCKHQDQ,
                        25 + 2 * i,
                        16 + 2 * i,
                        VexRm::Reg(17 + 2 * i),
                    ));
                }
                for (base, t) in [(16u8, 24u8), (20, 28)] {
                    self.emit_evex(E::shufi64x2(base, t, t + 2, 0x88));
                    self.emit_evex(E::shufi64x2(base + 1, t + 1, t + 3, 0x88));
                    self.emit_evex(E::shufi64x2(base + 2, t, t + 2, 0xdd));
                    self.emit_evex(E::shufi64x2(base + 3, t + 1, t + 3, 0xdd));
                }
                for k in 0..4u8 {
                    self.emit_evex(E::shufi64x2(24 + k, 16 + k, 20 + k, 0x88));
                    self.emit_evex(E::shufi64x2(28 + k, 16 + k, 20 + k, 0xdd));
                }
                // Column k is the new row k; zmm31 is the scratch register, so that one is stored
                for k in 0..8u8 {
                    let (reg, pair) = (24 + k, op.vd / 2 + k);
                    if reg == SCRATCH_ZMM {
                        self.emit_evex(E::store(reg, REGISTER_SCRATCH, Self::slot_disp(2 * pair)));
                    } else {
                        self.simd.pairs[pair as usize] = PairState {
                            reg: Some(reg),
                            dirty: true,
                        };
                        self.simd.zregs[(reg - ZBASE) as usize] = Some(pair);
                        self.simd_ztouch(reg);
                    }
                }
            }
            _ => {
                // 8 x 8 as four 4 x 4 blocks of 256 bit halves: [A B; C D] -> [A' C'; B' D']
                let row = |r: u8, hi: u8| op.vd + 2 * r + hi;
                for (src, dst) in [((0u8, 0u8), (0u8, 0u8)), ((4, 1), (4, 1))] {
                    // A and D in place
                    for r in 0..4u8 {
                        load(self, r, row(src.0 + r, src.1));
                    }
                    self.simd_emit_transpose4q([0, 1, 2, 3], [4, 5, 6, 7]);
                    for r in 0..4u8 {
                        self.emit_vex(V::store(
                            r,
                            REGISTER_SCRATCH,
                            Self::slot_disp(row(dst.0 + r, dst.1)),
                        ));
                    }
                }
                // B (rows 0..4, high halves) and C (rows 4..8, low halves) swap places
                for r in 0..4u8 {
                    load(self, r, row(r, 1));
                    load(self, 8 + r, row(4 + r, 0));
                }
                self.simd_emit_transpose4q([0, 1, 2, 3], [4, 5, 6, 7]);
                self.simd_emit_transpose4q([8, 9, 10, 11], [4, 5, 6, 7]);
                for r in 0..4u8 {
                    self.simd_assign(row(4 + r, 0), r);
                    self.simd_assign(row(r, 1), 8 + r);
                }
            }
        }
    }

    /// 4 x 4 transpose of the u64 lanes of the registers `m` in place, `t` are temporaries
    fn simd_emit_transpose4q(&mut self, m: [u8; 4], t: [u8; 4]) {
        let r = VexRm::Reg;
        self.emit_vex(V::alu(VPUNPCKLQDQ, t[0], m[0], r(m[1])));
        self.emit_vex(V::alu(VPUNPCKHQDQ, t[1], m[0], r(m[1])));
        self.emit_vex(V::alu(VPUNPCKLQDQ, t[2], m[2], r(m[3])));
        self.emit_vex(V::alu(VPUNPCKHQDQ, t[3], m[2], r(m[3])));
        self.emit_vex(V::perm2i128(m[0], t[0], r(t[2]), 0x20));
        self.emit_vex(V::perm2i128(m[1], t[1], r(t[3]), 0x20));
        self.emit_vex(V::perm2i128(m[2], t[0], r(t[2]), 0x31));
        self.emit_vex(V::perm2i128(m[3], t[1], r(t[3]), 0x31));
    }

    /// Makes `reg` (a temporary) the new, dirty register of `slot`
    fn simd_rebind(&mut self, slot: u8, reg: u8) {
        if let Some(old) = self.simd.slots[slot as usize].reg {
            self.simd.regs[old as usize] = None;
        }
        self.simd_assign(slot, reg);
    }

    /// Cross half permutations without AVX-512, in the register cache: both halves are computed into
    /// temporaries before any destination is written (the operands may alias)
    fn emit_simd_cross(&mut self, op: &VecInsn) {
        use crate::simd::*;
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        let ensure = |jit: &mut Self, slot: u8| {
            let reg = jit.simd_ensure_reg(slot);
            jit.simd_pin(reg);
            reg
        };
        let temp = |jit: &mut Self| {
            let reg = jit.simd_allocate();
            jit.simd_pin(reg);
            reg
        };
        let s = SCRATCH_YMM;
        match (op.subop, op.wide) {
            (VPERMD, _) => {
                // out_h = idx_h bit 3 ? a_hi[idx_h & 7] : a_lo[idx_h & 7]
                let (a0, a1) = (ensure(self, op.va), ensure(self, op.va + 1));
                let idx = [ensure(self, op.vb), ensure(self, op.vb + 1)];
                let out = [temp(self), temp(self)];
                let m = temp(self);
                for h in 0..2 {
                    self.emit_vex(V::permd(out[h], idx[h], VexRm::Reg(a0)));
                    self.emit_vex(V::permd(s, idx[h], VexRm::Reg(a1)));
                    self.emit_vex(V::pslld(m, idx[h], 28));
                    self.emit_vex(V::blendvps(out[h], out[h], s, m));
                }
                self.simd_rebind(op.vd, out[0]);
                self.simd_rebind(op.vd + 1, out[1]);
            }
            (VPERMI2D | VPERMT2D, false) => {
                // out = idx bit 3 ? b[idx & 7] : a[idx & 7]
                let (ta, ti) = if op.subop == VPERMI2D {
                    (op.va, op.vd)
                } else {
                    (op.vd, op.va)
                };
                let (a, b, idx) = (ensure(self, ta), ensure(self, op.vb), ensure(self, ti));
                let (out, m) = (temp(self), temp(self));
                self.emit_vex(V::permd(out, idx, VexRm::Reg(a)));
                self.emit_vex(V::permd(s, idx, VexRm::Reg(b)));
                self.emit_vex(V::pslld(m, idx, 28));
                self.emit_vex(V::blendvps(out, out, s, m));
                self.simd_rebind(op.vd, out);
            }
            _ => {
                // 512 bit: bit 3 of the index picks the half, bit 4 the table
                let (ta, ti) = if op.subop == VPERMI2D {
                    (op.va, op.vd)
                } else {
                    (op.vd, op.va)
                };
                let (a0, a1) = (ensure(self, ta), ensure(self, ta + 1));
                let (b0, b1) = (ensure(self, op.vb), ensure(self, op.vb + 1));
                let idx = [ensure(self, ti), ensure(self, ti + 1)];
                let out = [temp(self), temp(self)];
                let (m, t) = (temp(self), temp(self));
                for h in 0..2 {
                    self.emit_vex(V::pslld(m, idx[h], 28));
                    self.emit_vex(V::permd(out[h], idx[h], VexRm::Reg(a0)));
                    self.emit_vex(V::permd(s, idx[h], VexRm::Reg(a1)));
                    self.emit_vex(V::blendvps(out[h], out[h], s, m));
                    self.emit_vex(V::permd(t, idx[h], VexRm::Reg(b0)));
                    self.emit_vex(V::permd(s, idx[h], VexRm::Reg(b1)));
                    self.emit_vex(V::blendvps(t, t, s, m));
                    self.emit_vex(V::pslld(m, idx[h], 27));
                    self.emit_vex(V::blendvps(out[h], out[h], t, m));
                }
                self.simd_rebind(op.vd, out[0]);
                self.simd_rebind(op.vd + 1, out[1]);
            }
        }
        self.simd.pinned = [false; CACHE_REGS];
    }

    /// Register holding `slot` for a read, ymm15 loaded from memory if it isn't cached
    fn simd_read_reg(&mut self, slot: u8) -> u8 {
        if let Some(reg) = self.simd.slots[slot as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_touch(reg);
            return reg;
        }
        self.simd_ensure_base();
        self.emit_vex(V::load(
            SCRATCH_YMM,
            REGISTER_SCRATCH,
            Self::slot_disp(slot),
        ));
        SCRATCH_YMM
    }

    /// vmovmsk8, 512 bit: the masks of both halves combined in one 64 bit register
    fn emit_simd_movmsk_wide(&mut self, op: &VecInsn) {
        let dst = REGISTER_MAP[op.dst as usize];
        let lo = self.simd_read_reg(op.va);
        self.emit_vex(V::pmovmskb(dst, lo));
        let hi = self.simd_read_reg(op.va + 1);
        self.emit_vex(V::pmovmskb(REGISTER_SCRATCH, hi));
        self.simd.r11 = R11::Unknown;
        self.emit_ins(X86Instruction::alu_immediate(
            OperandSize::S64,
            0xc1,
            4,
            REGISTER_SCRATCH,
            32,
            None,
        ));
        self.emit_ins(X86Instruction::alu(
            OperandSize::S64,
            0x09,
            REGISTER_SCRATCH,
            dst,
            None,
        ));
    }

    /// Strategy B: slots of the vector file are cached in YMM registers (strategy A flushes after
    /// every instruction)
    fn emit_simd_cached(&mut self, op: &VecInsn) {
        use crate::simd::*;
        // Every vector instruction validates the vector base, like the interpreter
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        let native = self.simd.avx512 && self.simd_avx512_native(op);
        match op.subop {
            VSLLV16 | VSRLV16 | VSRAV16 if !native => {
                let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
                let out = self.simd_temp();
                self.emit_simd_var16(op.subop, out, ra, rb);
                self.simd_rebind(op.vd, out);
            }
            VSHLDV | VSHRDV if !native && op.off == 1 => self.emit_simd_funnel16_avx2(op),
            VLZCNT if !native => self.emit_simd_lzcnt_avx2(op),
            VMULTISHIFT if !native => self.emit_simd_multishift_avx2(op),
            VGF2P8MUL if !native => self.emit_simd_gf2p8mul_avx2(op),
            VGF2P8AFFINE if !native => self.emit_simd_gf2p8affine_avx2(op),
            VCONFLICT if !native => self.emit_simd_conflict_avx2(op),
            VDPBUSD if !native => self.emit_simd_dpbusd_avx2(op),
            VBCAST32 => {
                self.emit_vex(V::movd_from_gpr(SCRATCH_YMM, REGISTER_MAP[op.src as usize]));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pbroadcastd(rd, SCRATCH_YMM));
            }
            VBCAST64 => {
                self.emit_vex(V::movq_from_gpr(SCRATCH_YMM, REGISTER_MAP[op.src as usize]));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pbroadcastq(rd, SCRATCH_YMM));
            }
            VINS64 => {
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                self.emit_vex(V::movq_from_gpr(SCRATCH_YMM, REGISTER_MAP[op.src as usize]));
                self.emit_vex(V::pbroadcastq(SCRATCH_YMM, SCRATCH_YMM));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pblendd(
                    rd,
                    rd,
                    VexRm::Reg(SCRATCH_YMM),
                    3 << (2 * op.off),
                ));
            }
            VEXT64 => {
                let dst = REGISTER_MAP[op.dst as usize];
                if let Some(reg) = self.simd.slots[op.va as usize].reg {
                    self.simd.stats.cache_hits += 1;
                    self.simd_touch(reg);
                    let lane = op.off as u8;
                    let src = if lane >= 2 {
                        self.emit_vex(V::extracti128(SCRATCH_YMM, reg, 1));
                        SCRATCH_YMM
                    } else {
                        reg
                    };
                    self.emit_vex(if lane & 1 == 0 {
                        V::movq_to_gpr(dst, src)
                    } else {
                        V::pextrq(dst, src, 1)
                    });
                } else {
                    let disp = Self::slot_disp(op.va) + 8 * op.off as i32;
                    self.emit_ins(X86Instruction::load(
                        OperandSize::S64,
                        REGISTER_SCRATCH,
                        dst,
                        X86IndirectAccess::Offset(disp),
                    ));
                }
            }
            VMOVMSK8 => {
                let ra = self.simd_read_reg(op.va);
                self.emit_vex(V::pmovmskb(REGISTER_MAP[op.dst as usize], ra));
            }
            VADD32 | VXOR | VAND | VOR | VBLENDD | VSUB32 | VMULLO32 | VMINU32 | VSLLV32
            | VSRLV32 | VANDN | VADD16 | VSUB16 | VMULLO16 | VMULHI16 | VMINU16 | VMADD16
            | VBLENDW => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_vex(Self::simd_binary(op.subop, rd, ra, b, op.off as u8));
            }
            VROLV32 if self.simd.avx512 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu38(1, 0x15, rd, ra, b));
            }
            VTERNLOG if self.simd.avx512 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::pternlogd(1, rd, ra, b, op.off as u8));
            }
            VPERMI2D => {
                // AVX-512 only (without it, the fixed register lowering is used)
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::permi2d(1, rd, ra, b));
            }
            VMONT16 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let w = self.simd_source(op.vb);
                let wp = self.simd_source(op.vb + 2);
                let q = self.simd_source(op.off as u8);
                let t = self.simd_allocate();
                self.simd_pin(t);
                self.emit_vex(V::alu(VPMULLW, SCRATCH_YMM, ra, wp));
                self.emit_vex(V::alu(VPMULHW, SCRATCH_YMM, SCRATCH_YMM, q));
                self.emit_vex(V::alu(VPMULHW, t, ra, w));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::alu(VPSUBW, rd, t, VexRm::Reg(SCRATCH_YMM)));
            }
            VADD64 | VSUB64 | VMULU32 | VSLLV64 | VSRLV64 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                match op.subop {
                    VADD64 => self.emit_vex(V::alu(0xd4, rd, ra, b)),
                    VSUB64 => self.emit_vex(V::alu(0xfb, rd, ra, b)),
                    VMULU32 => self.emit_vex(V::alu(0xf4, rd, ra, b)),
                    VSLLV64 => self.emit_simd_shiftv64(true, rd, ra, b),
                    _ => self.emit_simd_shiftv64(false, rd, ra, b),
                }
            }
            VSLL16 | VSRL16 | VSRA16 | VSRA32 | VROR64 | VPOPCNT | VLZCNT | VZEXT | VSEXT => {
                let ra = self.simd_pinned(op.va);
                let n = op.off as u8;
                let ext = |sub: u8| match sub {
                    VSLL16 => 6,
                    VSRL16 => 2,
                    _ => 4,
                };
                match op.subop {
                    VSLL16 | VSRL16 | VSRA16 => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_vex(V::ymm(
                            1,
                            1,
                            0x71,
                            ext(op.subop),
                            rd,
                            VexRm::Reg(ra),
                            Some(n),
                        ));
                    }
                    VSRA32 => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_vex(V::psrad(rd, ra, n));
                    }
                    VROR64 if self.simd.avx512 => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_evex(E::op(1, 1, true, 1, 0x72, 0, rd, VexRm::Reg(ra), Some(n)));
                    }
                    VROR64 => {
                        let rd = self.simd_destination(op.vd);
                        if n == 0 {
                            if rd != ra {
                                self.emit_vex(V::mov(rd, ra));
                            }
                        } else {
                            self.emit_vex(V::psrlq(SCRATCH_YMM, ra, n));
                            self.emit_vex(V::psllq(rd, ra, 64 - n));
                            self.emit_vex(V::alu(VPOR, rd, rd, VexRm::Reg(SCRATCH_YMM)));
                        }
                    }
                    VPOPCNT if self.simd_avx512_native(op) && self.simd.avx512 => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_evex(E::op(
                            1,
                            2,
                            n & 1 != 0,
                            1,
                            if n < 2 { 0x54 } else { 0x55 },
                            rd,
                            0,
                            VexRm::Reg(ra),
                            None,
                        ));
                    }
                    VPOPCNT => self.emit_simd_popcnt_avx2(op.vd, ra, n),
                    VLZCNT => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_evex(E::op(1, 2, n == 3, 1, 0x44, rd, 0, VexRm::Reg(ra), None));
                    }
                    _ => {
                        // VZEXT / VSEXT from the low part of the source
                        let rd = self.simd_destination(op.vd);
                        self.emit_vex(V::ymm(
                            1,
                            2,
                            Self::widen_opcode(op),
                            rd,
                            0,
                            VexRm::Reg(ra),
                            None,
                        ));
                    }
                }
            }
            VPERMB | VMULTISHIFT => {
                // vpermb / vpmultishiftqb dst, indices / controls (b), table / data (a): both in registers
                let ra = self.simd_pinned(op.va);
                let rb = self.simd_pinned(op.vb);
                let rd = self.simd_destination(op.vd);
                let (w, opcode) = if op.subop == VPERMB {
                    (false, 0x8d)
                } else {
                    (true, 0x83)
                };
                self.emit_evex(E::op(1, 2, w, 1, opcode, rd, rb, VexRm::Reg(ra), None));
            }
            VSRAV32 | VUNPCKLO | VUNPCKHI | VPACKSS | VPACKUS | VSLLV16 | VSRLV16 | VSRAV16
            | VGF2P8AFFINE | VGF2P8MUL | VCLMUL => {
                let ra = self.simd_pinned(op.va);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                let imm = op.off as u8;
                match op.subop {
                    VSRAV32 => self.emit_vex(V::ymm(1, 2, 0x46, rd, ra, b, None)),
                    VUNPCKLO | VUNPCKHI | VPACKSS | VPACKUS => {
                        let (map, opcode) = Self::shuffle_opcode(op);
                        self.emit_vex(V::ymm(1, map, opcode, rd, ra, b, None));
                    }
                    VSLLV16 | VSRLV16 | VSRAV16 => {
                        let opcode = match op.subop {
                            VSLLV16 => 0x12,
                            VSRLV16 => 0x10,
                            _ => 0x11,
                        };
                        self.emit_evex(E::op(1, 2, true, 1, opcode, rd, ra, b, None));
                    }
                    VGF2P8AFFINE => {
                        self.emit_evex(E::op(1, 3, true, 1, 0xce, rd, ra, b, Some(imm)))
                    }
                    VGF2P8MUL => self.emit_evex(E::op(1, 2, false, 1, 0xcf, rd, ra, b, None)),
                    _ => self.emit_vex(V::ymm(1, 3, 0x44, rd, ra, b, Some(imm))),
                }
            }
            VSRAV64 if self.simd.avx512 => {
                let ra = self.simd_pinned(op.va);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu_q(1, 2, 0x46, rd, ra, b));
            }
            VSRAV64 => {
                // ((x ^ s) >> c) ^ s with s = sign mask (counts above 63 give s)
                let ra = self.simd_pinned(op.va);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let t = self.simd_temp();
                self.emit_vex(V::alu(VPXOR, t, t, VexRm::Reg(t)));
                self.emit_vex(V::alu38(0x37, SCRATCH_YMM, t, VexRm::Reg(ra)));
                self.emit_vex(V::alu(VPXOR, t, ra, VexRm::Reg(SCRATCH_YMM)));
                self.emit_simd_shiftv64(false, t, t, b);
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::alu(VPXOR, rd, t, VexRm::Reg(SCRATCH_YMM)));
            }
            VPERMI2B => {
                let ra = self.simd_pinned(op.va);
                self.simd_pinned(op.vd);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::op(1, 2, false, 1, 0x75, rd, ra, b, None));
            }
            VMIN | VMAX | VADDS | VSUBS | VAVG | VADD8 | VSUB8 | VMULHU16 | VMULHRS16 | VMULS32
            | VMADDUBS16 | VSADBW | VALIGNR
                if !(matches!(op.subop, VMIN | VMAX) && op.off & 3 == 3) =>
            {
                let ra = self.simd_pinned(op.va);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                let (map, opcode) = Self::core_opcode(op);
                self.emit_vex(V::ymm(
                    1,
                    map,
                    opcode,
                    rd,
                    ra,
                    b,
                    (op.subop == VALIGNR).then_some(op.off as u8),
                ));
            }
            VMIN | VMAX if self.simd.avx512 => {
                let ra = self.simd_pinned(op.va);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                let (_, opcode) = Self::core_opcode(op);
                self.emit_evex(E::op(1, 2, true, 1, opcode, rd, ra, b, None));
            }
            VMIN | VMAX => {
                // u64 lanes without AVX-512: compare (signed, or with flipped sign bits) and blend
                let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
                let (x, y) = if op.off & 4 != 0 {
                    let (tx, sign) = (self.simd_temp(), self.simd_table(TAB_SIGN + 96));
                    self.emit_vex(V::alu(VPXOR, tx, ra, sign));
                    self.emit_vex(V::alu(VPXOR, SCRATCH_YMM, rb, sign));
                    (tx, SCRATCH_YMM)
                } else {
                    (ra, rb)
                };
                // s = a > b
                let m = self.simd_temp();
                self.emit_vex(V::alu38(0x37, m, x, VexRm::Reg(y)));
                let rd = self.simd_destination(op.vd);
                if op.subop == VMIN {
                    self.emit_vex(V::pblendvb(rd, ra, rb, m));
                } else {
                    self.emit_vex(V::pblendvb(rd, rb, ra, m));
                }
            }
            VABS | VPERMQ | VSHUFLW | VSHUFHW | VTRUNC | VCONFLICT => {
                let ra = self.simd_pinned(op.va);
                let imm = op.off as u8;
                match op.subop {
                    VABS if imm < 3 => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_vex(V::ymm(1, 2, 0x1c + imm, rd, 0, VexRm::Reg(ra), None));
                    }
                    VABS if self.simd.avx512 => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_evex(E::op(1, 2, true, 1, 0x1f, rd, 0, VexRm::Reg(ra), None));
                    }
                    VABS => {
                        // (x ^ s) - s with s the sign mask
                        let t = self.simd_temp();
                        self.emit_vex(V::alu(VPXOR, t, t, VexRm::Reg(t)));
                        self.emit_vex(V::alu38(0x37, SCRATCH_YMM, t, VexRm::Reg(ra)));
                        self.emit_vex(V::alu(VPXOR, t, ra, VexRm::Reg(SCRATCH_YMM)));
                        let rd = self.simd_destination(op.vd);
                        self.emit_vex(V::alu(0xfb, rd, t, VexRm::Reg(SCRATCH_YMM)));
                    }
                    VPERMQ => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_vex(V::permq(rd, ra, imm));
                    }
                    VSHUFLW | VSHUFHW => {
                        let rd = self.simd_destination(op.vd);
                        let pp = if op.subop == VSHUFLW { 3 } else { 2 };
                        self.emit_vex(V::ymm(pp, 1, 0x70, rd, 0, VexRm::Reg(ra), Some(imm)));
                    }
                    VTRUNC if self.simd.avx512 => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_evex(E::op(
                            2,
                            2,
                            false,
                            1,
                            Self::trunc_opcode(op),
                            ra,
                            0,
                            VexRm::Reg(rd),
                            None,
                        ));
                    }
                    VTRUNC => {
                        let t = self.emit_simd_trunc_half(ra, imm);
                        self.simd_rebind(op.vd, t);
                    }
                    _ => {
                        let rd = self.simd_destination(op.vd);
                        self.emit_evex(E::op(1, 2, imm == 3, 1, 0xc4, rd, 0, VexRm::Reg(ra), None));
                    }
                }
            }
            VTESTM if self.simd.avx512 => {
                let ra = self.simd_pinned(op.va);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let size = (op.off & 3) as u8;
                let pp = if op.off & 4 != 0 { 2 } else { 1 };
                self.emit_evex(E::op(
                    pp,
                    2,
                    size & 1 != 0,
                    1,
                    if size < 2 { 0x26 } else { 0x27 },
                    KMASK,
                    ra,
                    b,
                    None,
                ));
                self.emit_vex(V::kmovq_to_gpr(REGISTER_MAP[op.dst as usize], KMASK));
            }
            VTESTM => {
                // (a & b) == 0 per lane, as a mask (inverted for testm)
                let ra = self.simd_pinned(op.va);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let (t, z) = (self.simd_temp(), self.simd_temp());
                let size = (op.off & 3) as usize;
                self.emit_vex(V::alu(VPAND, t, ra, b));
                self.emit_vex(V::alu(VPXOR, z, z, VexRm::Reg(z)));
                self.emit_vex(match size {
                    0 => V::alu(0x74, t, t, VexRm::Reg(z)),
                    1 => V::alu(0x75, t, t, VexRm::Reg(z)),
                    2 => V::alu(0x76, t, t, VexRm::Reg(z)),
                    _ => V::alu38(0x29, t, t, VexRm::Reg(z)),
                });
                let dst = REGISTER_MAP[op.dst as usize];
                self.emit_simd_movemask(t, size, dst);
                if op.off & 4 == 0 {
                    let lanes = 32u32 >> size;
                    let lane_mask = if lanes == 32 {
                        u32::MAX as i64
                    } else {
                        (1i64 << lanes) - 1
                    };
                    self.emit_ins(X86Instruction::alu_immediate(
                        OperandSize::S32,
                        0x81,
                        6,
                        dst,
                        lane_mask,
                        None,
                    ));
                }
            }
            VDPWSSD | VDPBUSD if native => {
                let ra = self.simd_pinned(op.va);
                self.simd_pinned(op.vd);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::op(
                    1,
                    2,
                    false,
                    1,
                    if op.subop == VDPBUSD { 0x50 } else { 0x52 },
                    rd,
                    ra,
                    b,
                    None,
                ));
            }
            VDPWSSD => {
                let ra = self.simd_pinned(op.va);
                self.simd_pinned(op.vd);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                self.emit_vex(V::alu(0xf5, SCRATCH_YMM, ra, b));
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::alu(VPADDD, rd, rd, VexRm::Reg(SCRATCH_YMM)));
            }
            VSHLD | VSHRD | VSHLDV | VSHRDV if native => {
                let ra = self.simd_pinned(op.va);
                if matches!(op.subop, VSHLDV | VSHRDV) {
                    self.simd_pinned(op.vd);
                }
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                let (map, opcode, w, imm) = Self::funnel_opcode(op);
                self.emit_evex(E::op(1, map, w, 1, opcode, rd, ra, b, imm));
            }
            VPERMW => {
                // AVX-512 only (without it, the reference implementation)
                let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::op(1, 2, true, 1, 0x8d, rd, rb, VexRm::Reg(ra), None));
            }
            VSHLD | VSHRD => self.emit_simd_funnel_imm_avx2(op),
            VSHLDV | VSHRDV => self.emit_simd_funnel_var_avx2(op),
            VSRA64 if !self.simd.avx512 => self.emit_simd_sra64_avx2(op),
            VMULLO64 if !self.simd.dq => self.emit_simd_mullo64_avx2(op),
            VMADD52LO | VMADD52HI if !self.simd.ifma => self.emit_simd_madd52_avx2(op),
            VCMP if !self.simd.avx512 => {
                let dst = REGISTER_MAP[op.dst as usize];
                self.emit_simd_cmp_avx2(op.va, op.vb, op.off, dst);
            }
            VBLENDM if !self.simd.avx512 => {
                let m = REGISTER_MAP[op.src as usize];
                self.emit_simd_blendm_avx2(op.vd, op.va, op.vb, op.off as u8, m);
            }
            VCOMPRESS | VEXPAND if !self.simd.avx512 || (op.off < 2 && !self.simd.vbmi2) => {
                self.emit_simd_compress_avx2(op)
            }
            VSLL64 | VSRL64 | VSRA64 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                let imm = op.off as u8;
                match op.subop {
                    VSLL64 => self.emit_vex(V::psllq(rd, ra, imm)),
                    VSRL64 => self.emit_vex(V::psrlq(rd, ra, imm)),
                    _ => self.emit_evex(E::shift_imm_q(1, 0x72, 4, rd, ra, imm)),
                }
            }
            VMULLO64 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu_q(1, 2, 0x40, rd, ra, b));
            }
            VMADD52LO | VMADD52HI => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu_q(
                    1,
                    2,
                    if op.subop == VMADD52LO { 0xb4 } else { 0xb5 },
                    rd,
                    ra,
                    b,
                ));
            }
            VCMP => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                self.emit_evex(E::pcmp(
                    1,
                    (op.off >> 4) as u8,
                    op.off & 8 != 0,
                    KMASK,
                    ra,
                    b,
                    (op.off & 7) as u8,
                ));
                self.emit_vex(V::kmovq_to_gpr(REGISTER_MAP[op.dst as usize], KMASK));
            }
            VBLENDM => {
                self.emit_vex(V::kmov_from_gpr(KMASK, REGISTER_MAP[op.src as usize], true));
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let a = if op.va == op.vb {
                    VexRm::Reg(rb)
                } else {
                    self.simd_source(op.va)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::pblendm(1, op.off as u8, rd, rb, a, KMASK));
            }
            VCOMPRESS | VEXPAND => {
                self.emit_vex(V::kmov_from_gpr(KMASK, REGISTER_MAP[op.src as usize], true));
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                if op.subop == VCOMPRESS {
                    self.emit_evex(E::pcompress(1, op.off as u8, rd, ra, KMASK));
                    self.emit_simd_lane_count(op, 32 >> op.off);
                } else {
                    self.emit_evex(E::pexpand(1, op.off as u8, rd, ra, KMASK));
                }
            }
            VSHUFB => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pshufb(rd, ra, b));
            }
            VPERMT2D => {
                // AVX-512 only (without it, emit_simd_cross)
                let ri = self.simd_ensure_reg(op.va);
                self.simd_pin(ri);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let b = if op.vb == op.va {
                    VexRm::Reg(ri)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::permt2d(1, rd, ri, b));
            }
            VROLV64 if self.simd.avx512 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let b = if op.vb == op.va {
                    VexRm::Reg(ra)
                } else {
                    self.simd_source(op.vb)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::alu_q(1, 2, 0x15, rd, ra, b));
            }
            VROLV64 => {
                // rd = ra << c | (ra >> 1) >> (c ^ 63) for c = rb & 63
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let t = self.simd_allocate();
                self.simd_pin(t);
                let rd = self.simd_destination(op.vd);
                let c63 = VexRm::Rip(self.anchors[ANCHOR_SIMD_C63Q]);
                self.emit_vex(V::load_rip(SCRATCH_YMM, self.anchors[ANCHOR_SIMD_C63Q]));
                self.emit_vex(V::alu(VPAND, SCRATCH_YMM, SCRATCH_YMM, VexRm::Reg(rb)));
                self.emit_vex(V::alu38w(VPSLLVD, t, ra, VexRm::Reg(SCRATCH_YMM)));
                self.emit_vex(V::alu(VPXOR, SCRATCH_YMM, SCRATCH_YMM, c63));
                self.emit_vex(V::psrlq(rd, ra, 1));
                self.emit_vex(V::alu38w(VPSRLVD, rd, rd, VexRm::Reg(SCRATCH_YMM)));
                self.emit_vex(V::alu(VPOR, rd, rd, VexRm::Reg(t)));
            }
            VROR32 if self.simd.avx512 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                self.emit_evex(E::shift_imm(1, 0, rd, ra, op.off as u8));
            }
            VROLV32 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                // A temporary register, not bound to a slot
                let t = self.simd_allocate();
                self.simd_pin(t);
                let rd = self.simd_destination(op.vd);
                self.simd_emit_rolv(rd, ra, rb, t);
            }
            VTERNLOG => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let rd = self.simd_ensure_reg(op.vd);
                self.simd_pin(rd);
                let rd = self.simd_destination(op.vd);
                self.simd_emit_ternlog(rd, ra, rb, op.off as u8);
            }
            VPERMD => {
                let rb = self.simd_ensure_reg(op.vb);
                self.simd_pin(rb);
                let a = if op.va == op.vb {
                    VexRm::Reg(rb)
                } else {
                    self.simd_source(op.va)
                };
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::permd(rd, rb, a));
            }
            VSHUFD => {
                let a = self.simd_source(op.va);
                let rd = self.simd_destination(op.vd);
                self.emit_vex(V::pshufd(rd, a, op.off as u8));
            }
            VROR32 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                self.simd_emit_rotate(rd, ra, op.off as u8);
            }
            VSRL32 | VSLL32 => {
                let ra = self.simd_ensure_reg(op.va);
                self.simd_pin(ra);
                let rd = self.simd_destination(op.vd);
                self.emit_vex(if op.subop == VSRL32 {
                    V::psrld(rd, ra, op.off as u8)
                } else {
                    V::pslld(rd, ra, op.off as u8)
                });
            }
            VMOV => {
                if op.vd != op.va {
                    let a = self.simd_source(op.va);
                    let rd = self.simd_destination(op.vd);
                    match a {
                        VexRm::Reg(ra) => self.emit_vex(V::mov(rd, ra)),
                        VexRm::Mem(base, disp) => self.emit_vex(V::load(rd, base, disp)),
                        VexRm::Rip(_) => unreachable!(),
                    }
                }
            }
            _ => unreachable!(),
        }
        self.simd.pinned = [false; CACHE_REGS];
    }

    /// Slot operands of a vector operation (the first slot of each pair for 512 bit operations),
    /// for everything but vload / vstore / vtranspose8
    fn simd_operand_slots(op: &VecInsn) -> ([u8; 5], usize) {
        use crate::simd::*;
        match op.subop {
            VBCAST32 | VBCAST64 | VINS64 => ([op.vd, 0, 0, 0, 0], 1),
            VEXT64 | VMOVMSK8 => ([op.va, 0, 0, 0, 0], 1),
            VCMP => ([op.va, op.vb, 0, 0, 0], 2),
            VCOMPRESS | VEXPAND | VSLL64 | VSRL64 | VSRA64 => ([op.vd, op.va, 0, 0, 0], 2),
            VMOV | VROR32 | VSHUFD | VSRL32 | VSLL32 => ([op.vd, op.va, 0, 0, 0], 2),
            VMONT16 => ([op.vd, op.va, op.vb, op.vb + 2, op.off as u8], 5),
            VSLL16 | VSRL16 | VSRA16 | VSRA32 | VROR64 | VPOPCNT | VLZCNT | VZEXT | VSEXT
            | VABS | VTRUNC | VPERMQ | VSHUFLW | VSHUFHW | VCONFLICT => {
                ([op.vd, op.va, 0, 0, 0], 2)
            }
            VTESTM => ([op.va, op.vb, 0, 0, 0], 2),
            _ => ([op.vd, op.va, op.vb, 0, 0], 3),
        }
    }

    /// Writes `slot` back from the YMM cache (if dirty) and drops it
    fn simd_drop_slot(&mut self, slot: u8) {
        let state = self.simd.slots[slot as usize];
        if let Some(reg) = state.reg {
            if state.dirty {
                self.simd_ensure_base();
                self.emit_vex(V::store(reg, REGISTER_SCRATCH, Self::slot_disp(slot)));
                self.simd.stats.spills += 1;
            }
            self.simd.slots[slot as usize] = SlotState::default();
            self.simd.regs[reg as usize] = None;
        }
    }

    /// Writes `pair` back from the ZMM cache (if dirty) and drops it
    fn simd_drop_pair(&mut self, pair: u8) {
        let state = self.simd.pairs[pair as usize];
        if let Some(reg) = state.reg {
            if state.dirty {
                self.simd_ensure_base();
                self.emit_evex(E::store(reg, REGISTER_SCRATCH, Self::slot_disp(2 * pair)));
                self.simd.stats.spills += 1;
            }
            self.simd.pairs[pair as usize] = PairState::default();
            self.simd.zregs[(reg - ZBASE) as usize] = None;
        }
    }

    fn simd_ztouch(&mut self, reg: u8) {
        self.simd.clock += 1;
        self.simd.zlast_use[(reg - ZBASE) as usize] = self.simd.clock;
    }

    fn simd_zpin(&mut self, reg: u8) {
        self.simd.zpinned[(reg - ZBASE) as usize] = true;
    }

    /// Picks a free or the least recently used unpinned ZMM register, writing back its pair if dirty
    fn simd_zallocate(&mut self) -> u8 {
        let mut best = None;
        for i in 0..ZREGS {
            if self.simd.zpinned[i] {
                continue;
            }
            if self.simd.zregs[i].is_none() {
                best = Some(i);
                break;
            }
            if best.is_none_or(|b: usize| self.simd.zlast_use[i] < self.simd.zlast_use[b]) {
                best = Some(i);
            }
        }
        let i = best.expect("all zmm registers pinned");
        let reg = ZBASE + i as u8;
        if let Some(victim) = self.simd.zregs[i] {
            if self.simd.pairs[victim as usize].dirty {
                self.simd_ensure_base();
                self.emit_evex(E::store(reg, REGISTER_SCRATCH, Self::slot_disp(2 * victim)));
                self.simd.stats.spills += 1;
            }
            self.simd.pairs[victim as usize] = PairState::default();
            self.simd.stats.evictions += 1;
            self.simd.zregs[i] = None;
        }
        self.simd_ztouch(reg);
        reg
    }

    /// ZMM register holding `pair`, loading it if necessary
    fn simd_zensure(&mut self, pair: u8) -> u8 {
        if let Some(reg) = self.simd.pairs[pair as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_ztouch(reg);
            return reg;
        }
        let reg = self.simd_zallocate();
        self.simd_ensure_base();
        self.emit_evex(E::load(reg, REGISTER_SCRATCH, Self::slot_disp(2 * pair)));
        self.simd.pairs[pair as usize] = PairState {
            reg: Some(reg),
            dirty: false,
        };
        self.simd.zregs[(reg - ZBASE) as usize] = Some(pair);
        reg
    }

    /// Register or memory operand for a read-only source pair
    fn simd_zsource(&mut self, pair: u8) -> VexRm {
        if let Some(reg) = self.simd.pairs[pair as usize].reg {
            self.simd.stats.cache_hits += 1;
            self.simd_ztouch(reg);
            self.simd_zpin(reg);
            VexRm::Reg(reg)
        } else {
            self.simd_ensure_base();
            VexRm::Mem(REGISTER_SCRATCH, Self::slot_disp(2 * pair))
        }
    }

    /// ZMM register which will receive the new value of `pair`
    fn simd_zdestination(&mut self, pair: u8) -> u8 {
        let reg = match self.simd.pairs[pair as usize].reg {
            Some(reg) => reg,
            None => {
                let reg = self.simd_zallocate();
                self.simd.zregs[(reg - ZBASE) as usize] = Some(pair);
                reg
            }
        };
        self.simd.pairs[pair as usize] = PairState {
            reg: Some(reg),
            dirty: true,
        };
        self.simd_ztouch(reg);
        reg
    }

    /// 512 bit operations in ZMM registers (AVX-512F / BW)
    fn emit_simd_zmm(&mut self, op: &VecInsn) {
        use crate::simd::*;
        self.simd_ensure_base();
        self.simd.zpinned = [false; ZREGS];
        let (pd, pa, pb) = (op.vd / 2, op.va / 2, op.vb / 2);
        match op.subop {
            VBCAST32 | VBCAST64 => {
                let zd = self.simd_zdestination(pd);
                let src = REGISTER_MAP[op.src as usize];
                self.emit_evex(if op.subop == VBCAST32 {
                    E::pbroadcastd_gpr(zd, src)
                } else {
                    E::pbroadcastq_gpr(zd, src)
                });
            }
            VMOVMSK8 => {
                let za = match self.simd.pairs[pa as usize].reg {
                    Some(reg) => {
                        self.simd.stats.cache_hits += 1;
                        reg
                    }
                    None => {
                        self.emit_evex(E::load(
                            SCRATCH_ZMM,
                            REGISTER_SCRATCH,
                            Self::slot_disp(op.va),
                        ));
                        SCRATCH_ZMM
                    }
                };
                self.emit_evex(E::pmovb2m(KMASK, za));
                self.emit_vex(V::kmovq_to_gpr(REGISTER_MAP[op.dst as usize], KMASK));
            }
            VADD32 | VXOR | VAND | VOR | VSUB32 | VMULLO32 | VMINU32 | VSLLV32 | VSRLV32
            | VANDN | VROLV32 | VROLV64 | VADD16 | VSUB16 | VMULLO16 | VMULHI16 | VMINU16
            | VMADD16 | VBLENDD | VBLENDW | VSHUFB => {
                if matches!(op.subop, VBLENDD | VBLENDW) {
                    let (anchor, size) = if op.subop == VBLENDD {
                        (ANCHOR_SIMD_BLENDD_MASKS, 2)
                    } else {
                        (ANCHOR_SIMD_BLENDW_MASKS, 4)
                    };
                    let entry = unsafe { self.anchors[anchor].add(size * op.off as usize) };
                    self.emit_vex(V::kmov_load(KMASK, op.subop == VBLENDW, VexRm::Rip(entry)));
                }
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let b = if pb == pa {
                    VexRm::Reg(za)
                } else {
                    self.simd_zsource(pb)
                };
                let zd = self.simd_zdestination(pd);
                let insn = match op.subop {
                    VADD32 => E::alu(2, 0xfe, zd, za, b),
                    VSUB32 => E::alu(2, 0xfa, zd, za, b),
                    VXOR => E::alu(2, 0xef, zd, za, b),
                    VAND => E::alu(2, 0xdb, zd, za, b),
                    VOR => E::alu(2, 0xeb, zd, za, b),
                    VANDN => E::alu(2, 0xdf, zd, za, b),
                    VADD16 => E::alu(2, 0xfd, zd, za, b),
                    VSUB16 => E::alu(2, 0xf9, zd, za, b),
                    VMULLO16 => E::alu(2, 0xd5, zd, za, b),
                    VMULHI16 => E::alu(2, 0xe5, zd, za, b),
                    VMADD16 => E::alu(2, 0xf5, zd, za, b),
                    VMULLO32 => E::alu38(2, 0x40, zd, za, b),
                    VMINU32 => E::alu38(2, 0x3b, zd, za, b),
                    VSLLV32 => E::alu38(2, 0x47, zd, za, b),
                    VSRLV32 => E::alu38(2, 0x45, zd, za, b),
                    VROLV32 => E::alu38(2, 0x15, zd, za, b),
                    VROLV64 => E::alu_q(2, 2, 0x15, zd, za, b),
                    VMINU16 => E::alu38(2, 0x3a, zd, za, b),
                    VSHUFB => E::pshufb(2, zd, za, b),
                    VBLENDD => E::pblendmd(zd, za, b, KMASK),
                    _ => E::pblendmw(zd, za, b, KMASK),
                };
                self.emit_evex(insn);
            }
            VTERNLOG | VPERMI2D => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zensure(pd);
                self.simd_zpin(zd);
                let b = if pb == pa {
                    VexRm::Reg(za)
                } else {
                    self.simd_zsource(pb)
                };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(if op.subop == VTERNLOG {
                    E::pternlogd(2, zd, za, b, op.off as u8)
                } else {
                    E::permi2d(2, zd, za, b)
                });
            }
            VSLL16 | VSRL16 | VSRA16 | VSRA32 | VROR64 | VPOPCNT | VLZCNT | VZEXT | VSEXT => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zdestination(pd);
                let n = op.off as u8;
                let r = VexRm::Reg(za);
                self.emit_evex(match op.subop {
                    VSLL16 => E::op(1, 1, false, 2, 0x71, 6, zd, r, Some(n)),
                    VSRL16 => E::op(1, 1, false, 2, 0x71, 2, zd, r, Some(n)),
                    VSRA16 => E::op(1, 1, false, 2, 0x71, 4, zd, r, Some(n)),
                    VSRA32 => E::op(1, 1, false, 2, 0x72, 4, zd, r, Some(n)),
                    VROR64 => E::op(1, 1, true, 2, 0x72, 0, zd, r, Some(n)),
                    VPOPCNT => E::op(
                        1,
                        2,
                        n & 1 != 0,
                        2,
                        if n < 2 { 0x54 } else { 0x55 },
                        zd,
                        0,
                        r,
                        None,
                    ),
                    VLZCNT => E::op(1, 2, n == 3, 2, 0x44, zd, 0, r, None),
                    _ => E::op(1, 2, false, 2, Self::widen_opcode(op), zd, 0, r, None),
                });
            }
            VSRAV32 | VSRAV64 | VUNPCKLO | VUNPCKHI | VPACKSS | VPACKUS | VSLLV16 | VSRLV16
            | VSRAV16 | VPERMB | VMULTISHIFT | VGF2P8AFFINE | VGF2P8MUL | VCLMUL => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zb = self.simd_zensure(pb);
                self.simd_zpin(zb);
                let zd = self.simd_zdestination(pd);
                let (b, imm) = (VexRm::Reg(zb), op.off as u8);
                self.emit_evex(match op.subop {
                    VSRAV32 => E::op(1, 2, false, 2, 0x46, zd, za, b, None),
                    VSRAV64 => E::op(1, 2, true, 2, 0x46, zd, za, b, None),
                    VUNPCKLO | VUNPCKHI | VPACKSS | VPACKUS => {
                        let (map, opcode) = Self::shuffle_opcode(op);
                        let w = matches!(op.subop, VUNPCKLO | VUNPCKHI) && op.off == 3;
                        E::op(1, map, w, 2, opcode, zd, za, b, None)
                    }
                    VSLLV16 => E::op(1, 2, true, 2, 0x12, zd, za, b, None),
                    VSRLV16 => E::op(1, 2, true, 2, 0x10, zd, za, b, None),
                    VSRAV16 => E::op(1, 2, true, 2, 0x11, zd, za, b, None),
                    VPERMB => E::op(1, 2, false, 2, 0x8d, zd, zb, VexRm::Reg(za), None),
                    VMULTISHIFT => E::op(1, 2, true, 2, 0x83, zd, zb, VexRm::Reg(za), None),
                    VGF2P8AFFINE => E::op(1, 3, true, 2, 0xce, zd, za, b, Some(imm)),
                    VGF2P8MUL => E::op(1, 2, false, 2, 0xcf, zd, za, b, None),
                    _ => E::op(1, 3, false, 2, 0x44, zd, za, b, Some(imm)),
                });
            }
            VPERMI2B => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zensure(pd);
                self.simd_zpin(zd);
                let b = if pb == pa {
                    VexRm::Reg(za)
                } else {
                    self.simd_zsource(pb)
                };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::op(1, 2, false, 2, 0x75, zd, za, b, None));
            }
            VMIN | VMAX | VADDS | VSUBS | VAVG | VADD8 | VSUB8 | VMULHU16 | VMULHRS16 | VMULS32
            | VMADDUBS16 | VSADBW | VALIGNR | VPERMW | VDPBUSD | VDPWSSD | VSHLD | VSHRD
            | VSHLDV | VSHRDV => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                if matches!(op.subop, VDPBUSD | VDPWSSD | VSHLDV | VSHRDV) {
                    let zd = self.simd_zensure(pd);
                    self.simd_zpin(zd);
                }
                let zb = self.simd_zensure(pb);
                self.simd_zpin(zb);
                let zd = self.simd_zdestination(pd);
                let b = VexRm::Reg(zb);
                let insn = match op.subop {
                    VPERMW => E::op(1, 2, true, 2, 0x8d, zd, zb, VexRm::Reg(za), None),
                    VDPBUSD => E::op(1, 2, false, 2, 0x50, zd, za, b, None),
                    VDPWSSD => E::op(1, 2, false, 2, 0x52, zd, za, b, None),
                    VSHLD | VSHRD | VSHLDV | VSHRDV => {
                        let (map, opcode, w, imm) = Self::funnel_opcode(op);
                        E::op(1, map, w, 2, opcode, zd, za, b, imm)
                    }
                    _ => {
                        let (map, opcode) = Self::core_opcode(op);
                        let w = matches!(op.subop, VMIN | VMAX) && op.off & 3 == 3
                            || op.subop == VMULS32;
                        E::op(
                            1,
                            map,
                            w,
                            2,
                            opcode,
                            zd,
                            za,
                            b,
                            (op.subop == VALIGNR).then_some(op.off as u8),
                        )
                    }
                };
                self.emit_evex(insn);
            }
            VABS | VPERMQ | VSHUFLW | VSHUFHW | VTRUNC | VCONFLICT => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zdestination(pd);
                let (r, imm) = (VexRm::Reg(za), op.off as u8);
                self.emit_evex(match op.subop {
                    VABS => E::op(1, 2, imm == 3, 2, 0x1c + imm, zd, 0, r, None),
                    VPERMQ => E::op(1, 3, true, 2, 0x00, zd, 0, r, Some(imm)),
                    VSHUFLW => E::op(3, 1, false, 2, 0x70, zd, 0, r, Some(imm)),
                    VSHUFHW => E::op(2, 1, false, 2, 0x70, zd, 0, r, Some(imm)),
                    VTRUNC => E::op(
                        2,
                        2,
                        false,
                        2,
                        Self::trunc_opcode(op),
                        za,
                        0,
                        VexRm::Reg(zd),
                        None,
                    ),
                    _ => E::op(1, 2, imm == 3, 2, 0xc4, zd, 0, r, None),
                });
            }
            VTESTM => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let b = if pb == pa {
                    VexRm::Reg(za)
                } else {
                    self.simd_zsource(pb)
                };
                let size = (op.off & 3) as u8;
                let pp = if op.off & 4 != 0 { 2 } else { 1 };
                self.emit_evex(E::op(
                    pp,
                    2,
                    size & 1 != 0,
                    2,
                    if size < 2 { 0x26 } else { 0x27 },
                    KMASK,
                    za,
                    b,
                    None,
                ));
                self.emit_vex(V::kmovq_to_gpr(REGISTER_MAP[op.dst as usize], KMASK));
            }
            VMONT16 => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let w = self.simd_zsource(pb);
                let wp = self.simd_zsource(pb + 1);
                let q = self.simd_zsource(op.off as u8 / 2);
                let t = self.simd_zallocate();
                self.simd_zpin(t);
                self.emit_evex(E::alu(2, VPMULLW, SCRATCH_ZMM, za, wp));
                self.emit_evex(E::alu(2, VPMULHW, SCRATCH_ZMM, SCRATCH_ZMM, q));
                self.emit_evex(E::alu(2, VPMULHW, t, za, w));
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::alu(2, VPSUBW, zd, t, VexRm::Reg(SCRATCH_ZMM)));
            }
            VADD64 | VSUB64 | VMULU32 | VSLLV64 | VSRLV64 | VMULLO64 => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let b = if pb == pa {
                    VexRm::Reg(za)
                } else {
                    self.simd_zsource(pb)
                };
                let zd = self.simd_zdestination(pd);
                let (map, opcode) = match op.subop {
                    VADD64 => (1, 0xd4),
                    VSUB64 => (1, 0xfb),
                    VMULU32 => (1, 0xf4),
                    VSLLV64 => (2, 0x47),
                    VSRLV64 => (2, 0x45),
                    _ => (2, 0x40),
                };
                self.emit_evex(E::alu_q(2, map, opcode, zd, za, b));
            }
            VSLL64 | VSRL64 | VSRA64 => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zdestination(pd);
                let (opcode, ext) = match op.subop {
                    VSLL64 => (0x73, 6),
                    VSRL64 => (0x73, 2),
                    _ => (0x72, 4),
                };
                self.emit_evex(E::shift_imm_q(2, opcode, ext, zd, za, op.off as u8));
            }
            VMADD52LO | VMADD52HI => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zensure(pd);
                self.simd_zpin(zd);
                let b = if pb == pa {
                    VexRm::Reg(za)
                } else {
                    self.simd_zsource(pb)
                };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::alu_q(
                    2,
                    2,
                    if op.subop == VMADD52LO { 0xb4 } else { 0xb5 },
                    zd,
                    za,
                    b,
                ));
            }
            VCMP => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let b = if pb == pa {
                    VexRm::Reg(za)
                } else {
                    self.simd_zsource(pb)
                };
                self.emit_evex(E::pcmp(
                    2,
                    (op.off >> 4) as u8,
                    op.off & 8 != 0,
                    KMASK,
                    za,
                    b,
                    (op.off & 7) as u8,
                ));
                self.emit_vex(V::kmovq_to_gpr(REGISTER_MAP[op.dst as usize], KMASK));
            }
            VBLENDM => {
                self.emit_vex(V::kmov_from_gpr(KMASK, REGISTER_MAP[op.src as usize], true));
                let zb = self.simd_zensure(pb);
                self.simd_zpin(zb);
                let a = if pa == pb {
                    VexRm::Reg(zb)
                } else {
                    self.simd_zsource(pa)
                };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::pblendm(2, op.off as u8, zd, zb, a, KMASK));
            }
            VCOMPRESS | VEXPAND => {
                self.emit_vex(V::kmov_from_gpr(KMASK, REGISTER_MAP[op.src as usize], true));
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zdestination(pd);
                if op.subop == VCOMPRESS {
                    self.emit_evex(E::pcompress(2, op.off as u8, zd, za, KMASK));
                    self.emit_simd_lane_count(op, 64 >> op.off);
                } else {
                    self.emit_evex(E::pexpand(2, op.off as u8, zd, za, KMASK));
                }
            }
            VPERMT2D => {
                let zidx = self.simd_zensure(pa);
                self.simd_zpin(zidx);
                let zd = self.simd_zensure(pd);
                self.simd_zpin(zd);
                let b = if pb == pa {
                    VexRm::Reg(zidx)
                } else {
                    self.simd_zsource(pb)
                };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::permt2d(2, zd, zidx, b));
            }
            VPERMD => {
                let zidx = self.simd_zensure(pb);
                self.simd_zpin(zidx);
                let data = if pa == pb {
                    VexRm::Reg(zidx)
                } else {
                    self.simd_zsource(pa)
                };
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::permd(zd, zidx, data));
            }
            VSHUFD => {
                let a = self.simd_zsource(pa);
                let zd = self.simd_zdestination(pd);
                self.emit_evex(E::pshufd(zd, a, op.off as u8));
            }
            VROR32 | VSRL32 | VSLL32 => {
                let za = self.simd_zensure(pa);
                self.simd_zpin(za);
                let zd = self.simd_zdestination(pd);
                let ext = match op.subop {
                    VROR32 => 0,
                    VSRL32 => 2,
                    _ => 6,
                };
                self.emit_evex(E::shift_imm(2, ext, zd, za, op.off as u8));
            }
            VMOV => {
                if pd != pa {
                    let a = self.simd_zsource(pa);
                    let zd = self.simd_zdestination(pd);
                    match a {
                        VexRm::Reg(za) => self.emit_evex(E::mov(zd, za)),
                        VexRm::Mem(base, disp) => self.emit_evex(E::load(zd, base, disp)),
                        VexRm::Rip(_) => unreachable!(),
                    }
                }
            }
            _ => unreachable!(),
        }
        self.simd.zpinned = [false; ZREGS];
    }

    /// compress without a native or template lowering (8 / 16 bit lanes without VBMI2, 512 bit forms
    /// without AVX-512): a branchless scalar loop over the lanes in the vector file (lane i read, stored
    /// at the count, `count += mask bit i`), then the remaining lanes are zeroed. The store goes to
    /// index count <= i after lane i is read, so it never clobbers a lane not yet read and the operands
    /// may alias.
    fn emit_simd_compress_scalar(&mut self, op: &VecInsn) {
        let shift = op.off as u8;
        let size = match shift {
            0 => OperandSize::S8,
            1 => OperandSize::S16,
            2 => OperandSize::S32,
            _ => OperandSize::S64,
        };
        let n = (ebpf::VECTOR_WIDTH * op.halves()) as i32 >> shift;
        // The operands must be in memory
        for slot in [op.vd, op.va] {
            // (in usize: slot 255 + 1 must not wrap to an empty range)
            for s in (slot as usize..slot as usize + op.halves()).map(|s| s as u8) {
                self.simd_drop_slot(s);
                self.simd_drop_pair(s / 2);
            }
        }
        self.simd_ensure_base();
        let (count, mask) = (REGISTER_MAP[op.dst as usize], REGISTER_MAP[op.src as usize]);
        let mut temps = [
            REGISTER_MAP[1],
            REGISTER_MAP[2],
            REGISTER_MAP[3],
            REGISTER_MAP[4],
        ]
        .iter()
        .copied()
        .filter(|r| *r != count && *r != mask);
        let (t, z) = (temps.next().unwrap(), temps.next().unwrap());
        self.emit_ins(X86Instruction::push(t, None));
        self.emit_ins(X86Instruction::push(z, None));
        // The mask is copied first: count may be the same register
        self.emit_ins(X86Instruction::mov(OperandSize::S64, mask, z));
        self.emit_ins(X86Instruction::load_immediate(count, 0));
        let (da, dd) = (Self::slot_disp(op.va), Self::slot_disp(op.vd));
        for i in 0..n {
            self.emit_ins(X86Instruction::load(
                size,
                REGISTER_SCRATCH,
                t,
                X86IndirectAccess::Offset(da + (i << shift)),
            ));
            self.emit_ins(X86Instruction::store(
                size,
                t,
                REGISTER_SCRATCH,
                X86IndirectAccess::OffsetIndexShift(dd, count, shift),
            ));
            // bt z, i (CF = mask bit i), then adc count, 0
            let zr = z as u8;
            self.emit::<u8>(0x48 | if zr >= 8 { 1 } else { 0 });
            self.emit::<u8>(0x0f);
            self.emit::<u8>(0xba);
            self.emit::<u8>(0xe0 | (zr & 7));
            self.emit::<u8>(i as u8);
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0x81,
                2,
                count,
                0,
                None,
            ));
        }
        // for (t = count; t < n; t++) d[t] = 0
        self.emit_ins(X86Instruction::load_immediate(z, 0));
        self.emit_ins(X86Instruction::mov(OperandSize::S64, count, t));
        let top = self.offset_in_text_section;
        self.emit_ins(X86Instruction::cmp_immediate(
            OperandSize::S64,
            t,
            n as i64,
            None,
        ));
        // jae rel32 (patched), emitted raw so that no diversification noop separates it from the patch
        self.emit::<u8>(0x0f);
        self.emit::<u8>(0x83);
        self.emit::<i32>(0);
        let patch = self.offset_in_text_section;
        self.emit_ins(X86Instruction::store(
            size,
            z,
            REGISTER_SCRATCH,
            X86IndirectAccess::OffsetIndexShift(dd, t, shift),
        ));
        self.emit_ins(X86Instruction::alu_immediate(
            OperandSize::S64,
            0x81,
            0,
            t,
            1,
            None,
        ));
        self.emit::<u8>(0xe9);
        let back = top as i64 - (self.offset_in_text_section as i64 + 4);
        self.emit::<i32>(back as i32);
        let distance = (self.offset_in_text_section - patch) as i32;
        unsafe {
            let ptr = self.result.text_section_mut().as_mut_ptr().add(patch - 4);
            std::ptr::write_unaligned(ptr.cast::<i32>(), distance);
        }
        self.emit_ins(X86Instruction::pop(z));
        self.emit_ins(X86Instruction::pop(t));
    }

    /// Whether the AVX-512 lowering has an instruction for the operation on this host
    fn simd_avx512_native(&self, op: &VecInsn) -> bool {
        use crate::simd::*;
        match op.subop {
            VCOMPRESS | VEXPAND => op.off >= 2 || self.simd.vbmi2,
            VMULLO64 => self.simd.dq,
            VMADD52LO | VMADD52HI => self.simd.ifma,
            VPOPCNT => {
                if op.off < 2 {
                    self.simd.bitalg
                } else {
                    self.simd.vpopcntdq
                }
            }
            VLZCNT => self.simd.cd,
            VPERMB | VPERMI2B | VMULTISHIFT => self.simd.vbmi,
            VGF2P8AFFINE | VGF2P8MUL => self.simd.gfni,
            VCLMUL => self.simd.vpclmul,
            VDPBUSD | VDPWSSD => self.simd.vnni,
            VSHLD | VSHRD | VSHLDV | VSHRDV => self.simd.vbmi2,
            VCONFLICT => self.simd.cd,
            _ => true,
        }
    }

    /// `dst = popcnt(src & low lanes)` for a compress over `lanes` lanes
    fn emit_simd_lane_count(&mut self, op: &VecInsn, lanes: u32) {
        let (dst, src) = (REGISTER_MAP[op.dst as usize], REGISTER_MAP[op.src as usize]);
        if lanes == 64 {
            self.emit_ins(X86Instruction::mov(OperandSize::S64, src, dst));
        } else {
            // A 32 bit mov zero extends
            self.emit_ins(X86Instruction::mov(OperandSize::S32, src, dst));
            if lanes < 32 {
                self.emit_ins(X86Instruction::alu_immediate(
                    OperandSize::S32,
                    0x81,
                    4,
                    dst,
                    (1i64 << lanes) - 1,
                    None,
                ));
            }
        }
        // popcnt r64, r64 (no helper for it in the emitter)
        let d = dst as u8;
        self.emit::<u8>(0xf3);
        self.emit::<u8>(0x48 | if d >= 8 { 0x05 } else { 0 });
        self.emit::<u8>(0x0f);
        self.emit::<u8>(0xb8);
        self.emit::<u8>(0xc0 | (d & 7) << 3 | (d & 7));
        self.simd.stats.native_instructions += 1;
    }

    /// Operations without an instruction on this host: one call to the reference implementation
    /// (the scalar result, if any, comes back in the destination register)
    fn emit_simd_fallback(&mut self, op: &VecInsn) {
        // Host calls clobber all vector registers and read and write the vector file in memory
        self.simd_flush(true);
        self.simd.stats.intrinsic_slow_paths += 1;
        let encoded = simd::encode_for_host(op);
        let scalar = if simd::uses_src(op.subop) {
            Value::Register(REGISTER_MAP[op.src as usize])
        } else {
            Value::Constant64(0, false)
        };
        let dst = simd::writes_dst(op.subop).then(|| REGISTER_MAP[op.dst as usize]);
        self.emit_rust_call(
            Value::Constant64(simd::jit_interpret as *const u8 as i64, false),
            &[
                // First, the other arguments may overwrite it
                Argument {
                    index: 2,
                    value: scalar,
                },
                Argument {
                    index: 1,
                    value: Value::Constant64(encoded as i64, false),
                },
                Argument {
                    index: 0,
                    value: Value::RegisterIndirect(
                        REGISTER_PTR_TO_VM,
                        self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase),
                        false,
                    ),
                },
            ],
            dst,
        );
        self.simd.r11 = R11::Unknown;
    }

    /// RIP relative operand into the constant tables
    fn simd_table(&self, offset: usize) -> VexRm {
        VexRm::Rip(unsafe { self.anchors[ANCHOR_SIMD_TABLES].add(offset) })
    }

    /// A pinned temporary YMM register (not bound to a slot)
    fn simd_temp(&mut self) -> u8 {
        let reg = self.simd_allocate();
        self.simd_pin(reg);
        reg
    }

    /// A pinned register holding `slot`
    fn simd_pinned(&mut self, slot: u8) -> u8 {
        let reg = self.simd_ensure_reg(slot);
        self.simd_pin(reg);
        reg
    }

    /// sra64 without AVX-512: the high u32 halves shift arithmetically, the low ones come from a
    /// logical shift (n < 32) or from the high halves (n >= 32)
    fn emit_simd_sra64_avx2(&mut self, op: &VecInsn) {
        let n = op.off as u8;
        let ra = self.simd_pinned(op.va);
        if n == 0 {
            let rd = self.simd_destination(op.vd);
            if rd != ra {
                self.emit_vex(V::mov(rd, ra));
            }
        } else if n < 32 {
            self.emit_vex(V::psrad(SCRATCH_YMM, ra, n));
            let rd = self.simd_destination(op.vd);
            self.emit_vex(V::psrlq(rd, ra, n));
            self.emit_vex(V::pblendd(rd, rd, VexRm::Reg(SCRATCH_YMM), 0xaa));
        } else {
            self.emit_vex(V::psrad(SCRATCH_YMM, ra, n - 32));
            self.emit_vex(V::pshufd(SCRATCH_YMM, VexRm::Reg(SCRATCH_YMM), 0xf5));
            let t = self.simd_temp();
            self.emit_vex(V::psrad(t, ra, 31));
            let rd = self.simd_destination(op.vd);
            self.emit_vex(V::pblendd(rd, SCRATCH_YMM, VexRm::Reg(t), 0xaa));
        }
    }

    /// mullo64 without AVX512DQ: lo(a) lo(b) + (hi(a) lo(b) + lo(a) hi(b)) << 32 (LLVM's lowering)
    fn emit_simd_mullo64_avx2(&mut self, op: &VecInsn) {
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        let (s, t) = (SCRATCH_YMM, self.simd_temp());
        self.emit_vex(V::psrlq(s, rb, 32));
        self.emit_vex(V::alu(0xf4, s, s, VexRm::Reg(ra)));
        self.emit_vex(V::psrlq(t, ra, 32));
        self.emit_vex(V::alu(0xf4, t, t, VexRm::Reg(rb)));
        self.emit_vex(V::alu(0xd4, s, s, VexRm::Reg(t)));
        self.emit_vex(V::psllq(s, s, 32));
        let rd = self.simd_destination(op.vd);
        self.emit_vex(V::alu(0xf4, rd, ra, VexRm::Reg(rb)));
        self.emit_vex(V::alu(0xd4, rd, rd, VexRm::Reg(s)));
    }

    /// madd52 without AVX512_IFMA, from 26 bit limbs: p = a1 b1 2^52 + (a1 b0 + a0 b1) 2^26 + a0 b0,
    /// lo = (a0 b0 + mid << 26) mod 2^52, hi = a1 b1 + ((mid + (a0 b0 >> 26)) >> 26)
    fn emit_simd_madd52_avx2(&mut self, op: &VecInsn) {
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        let rd = self.simd_pinned(op.vd);
        let (a0, a1, b0, b1, p, m) = (
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
        );
        let (s, m26, m52) = (
            SCRATCH_YMM,
            self.simd_table(TAB_M26),
            self.simd_table(TAB_M52),
        );
        let r = VexRm::Reg;
        self.emit_vex(V::alu(VPAND, a0, ra, m26));
        self.emit_vex(V::psrlq(a1, ra, 26));
        self.emit_vex(V::alu(VPAND, a1, a1, m26));
        self.emit_vex(V::alu(VPAND, b0, rb, m26));
        self.emit_vex(V::psrlq(b1, rb, 26));
        self.emit_vex(V::alu(VPAND, b1, b1, m26));
        self.emit_vex(V::alu(0xf4, p, a0, r(b0)));
        self.emit_vex(V::alu(0xf4, m, a1, r(b0)));
        self.emit_vex(V::alu(0xf4, s, a0, r(b1)));
        self.emit_vex(V::alu(0xd4, m, m, r(s)));
        if op.subop == simd::VMADD52LO {
            self.emit_vex(V::psllq(m, m, 26));
            self.emit_vex(V::alu(0xd4, p, p, r(m)));
            self.emit_vex(V::alu(VPAND, p, p, m52));
        } else {
            self.emit_vex(V::psrlq(s, p, 26));
            self.emit_vex(V::alu(0xd4, m, m, r(s)));
            self.emit_vex(V::psrlq(m, m, 26));
            self.emit_vex(V::alu(0xf4, p, a1, r(b1)));
            self.emit_vex(V::alu(0xd4, p, p, r(m)));
        }
        let rd2 = self.simd_destination(op.vd);
        debug_assert_eq!(rd, rd2);
        self.emit_vex(V::alu(0xd4, rd2, rd2, r(p)));
    }

    /// cmp without AVX-512 into the scalar register `target`: vpcmpgt / vpcmpeq (unsigned through
    /// flipped sign bits), a byte / u32 / u64 sign mask, inverted for le / ne / ge
    fn emit_simd_cmp_avx2(&mut self, va: u8, vb: u8, off: i16, target: X86Register) {
        let (size, pred, unsigned) = ((off >> 4) as usize, off & 7, off & 8 != 0);
        let lanes = 32u32 >> size;
        let lane_mask = if lanes == 32 {
            u32::MAX as i64
        } else {
            (1i64 << lanes) - 1
        };
        if pred == 3 || pred == 7 {
            self.emit_ins(X86Instruction::load_immediate(
                target,
                if pred == 7 { lane_mask } else { 0 },
            ));
            return;
        }
        let (mut x, mut y) = (self.simd_pinned(va), self.simd_pinned(vb));
        if unsigned {
            let (tx, sign) = (self.simd_temp(), self.simd_table(TAB_SIGN + 32 * size));
            self.emit_vex(V::alu(VPXOR, tx, x, sign));
            self.emit_vex(V::alu(VPXOR, SCRATCH_YMM, y, sign));
            (x, y) = (tx, SCRATCH_YMM);
        }
        let r = self.simd_temp();
        let (eq, gt) = match size {
            0 => (
                V::alu(0x74, r, x, VexRm::Reg(y)),
                V::alu(0x64, r, x, VexRm::Reg(y)),
            ),
            1 => (
                V::alu(0x75, r, x, VexRm::Reg(y)),
                V::alu(0x65, r, x, VexRm::Reg(y)),
            ),
            2 => (
                V::alu(0x76, r, x, VexRm::Reg(y)),
                V::alu(0x66, r, x, VexRm::Reg(y)),
            ),
            _ => (
                V::alu38(0x29, r, x, VexRm::Reg(y)),
                V::alu38(0x37, r, x, VexRm::Reg(y)),
            ),
        };
        // eq / ne: x == y, lt / ge: y > x, le / gt: x > y
        let swapped = |mut i: V| {
            if let VexRm::Reg(yy) = i.rm {
                i.rm = VexRm::Reg(i.vvvv);
                i.vvvv = yy;
            }
            i
        };
        self.emit_vex(match pred {
            0 | 4 => eq,
            1 | 5 => swapped(gt),
            _ => gt,
        });
        match size {
            0 => self.emit_vex(V::pmovmskb(target, r)),
            1 => {
                self.emit_vex(V::alu(0x63, r, r, VexRm::Reg(r)));
                self.emit_vex(V::permq(r, r, 0xd8));
                self.emit_vex(V::pmovmskb(target, r));
                self.emit_ins(X86Instruction::alu_immediate(
                    OperandSize::S32,
                    0x81,
                    4,
                    target,
                    0xffff,
                    None,
                ));
            }
            _ => self.emit_vex(V::movmskps(target, r, size == 3)),
        }
        if matches!(pred, 2 | 4 | 5) {
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S32,
                0x81,
                6,
                target,
                lane_mask,
                None,
            ));
        }
    }

    /// ymm15 = all ones in the lanes whose bit is set in the scalar register `m`
    fn emit_simd_mask_vector(&mut self, size: u8, m: X86Register) {
        let (s, bits) = (SCRATCH_YMM, self.simd_table(TAB_BITS + 32 * size as usize));
        match size {
            0 => {
                self.emit_vex(V::movd_from_gpr(s, m));
                self.emit_vex(V::pbroadcastd(s, s));
                self.emit_vex(V::pshufb(s, s, self.simd_table(TAB_BSHUF)));
            }
            1 => {
                self.emit_vex(V::movd_from_gpr(s, m));
                self.emit_vex(V::pbroadcastw(s, s));
            }
            2 => {
                self.emit_vex(V::movd_from_gpr(s, m));
                self.emit_vex(V::pbroadcastd(s, s));
            }
            _ => {
                self.emit_vex(V::movq_from_gpr(s, m));
                self.emit_vex(V::pbroadcastq(s, s));
            }
        }
        self.emit_vex(V::alu(VPAND, s, s, bits));
        self.emit_vex(match size {
            0 => V::alu(0x74, s, s, bits),
            1 => V::alu(0x75, s, s, bits),
            2 => V::alu(0x76, s, s, bits),
            _ => V::alu38(0x29, s, s, bits),
        });
    }

    /// blendm without AVX-512: the mask expanded to a vector, then vpblendvb
    fn emit_simd_blendm_avx2(&mut self, vd: u8, va: u8, vb: u8, size: u8, m: X86Register) {
        self.emit_simd_mask_vector(size, m);
        if m == REGISTER_SCRATCH {
            self.simd.r11 = R11::Unknown;
        }
        let (ra, rb) = (self.simd_pinned(va), self.simd_pinned(vb));
        let rd = self.simd_destination(vd);
        self.emit_vex(V::pblendvb(rd, rb, ra, SCRATCH_YMM));
    }

    /// 512 bit cmp / blendm without AVX-512: two halves, the high half's mask in r11
    fn emit_simd_mask_split(&mut self, op: &VecInsn) {
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        if op.subop == simd::VCMP {
            let dst = REGISTER_MAP[op.dst as usize];
            let lanes = (32u32 >> (op.off >> 4)) as i64;
            self.emit_simd_cmp_avx2(op.va, op.vb, op.off, dst);
            self.simd.pinned = [false; CACHE_REGS];
            self.emit_simd_cmp_avx2(op.va + 1, op.vb + 1, op.off, REGISTER_SCRATCH);
            self.simd.r11 = R11::Unknown;
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0xc1,
                4,
                REGISTER_SCRATCH,
                lanes,
                None,
            ));
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x09,
                REGISTER_SCRATCH,
                dst,
                None,
            ));
        } else {
            let (m, size) = (REGISTER_MAP[op.src as usize], op.off as u8);
            self.emit_simd_blendm_avx2(op.vd, op.va, op.vb, size, m);
            self.simd.pinned = [false; CACHE_REGS];
            self.emit_ins(X86Instruction::mov(OperandSize::S64, m, REGISTER_SCRATCH));
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0xc1,
                5,
                REGISTER_SCRATCH,
                (32 >> size) as i64,
                None,
            ));
            self.emit_simd_blendm_avx2(op.vd + 1, op.va + 1, op.vb + 1, size, REGISTER_SCRATCH);
        }
        self.simd.pinned = [false; CACHE_REGS];
    }

    /// compress / expand of u32 / u64 lanes without AVX-512: vpermd indices from a table indexed by
    /// the mask, then the unused lanes are zeroed
    fn emit_simd_compress_avx2(&mut self, op: &VecInsn) {
        let (size, src) = (op.off as u8, REGISTER_MAP[op.src as usize]);
        let (lanes, table) = match (size, op.subop == simd::VEXPAND) {
            (2, false) => (8, TAB_C32),
            (2, true) => (8, TAB_E32),
            (_, false) => (4, TAB_C64),
            _ => (4, TAB_E64),
        };
        // ymm15 = the 8 u32 indices at table + 8 * (mask & lanes)
        let t = [REGISTER_MAP[1], REGISTER_MAP[2], REGISTER_MAP[3]]
            .iter()
            .copied()
            .find(|r| *r != src && *r != REGISTER_MAP[op.dst as usize])
            .unwrap();
        self.emit_ins(X86Instruction::push(t, None));
        self.emit_ins(X86Instruction::mov(OperandSize::S32, src, t));
        self.emit_ins(X86Instruction::alu_immediate(
            OperandSize::S32,
            0x81,
            4,
            t,
            (1i64 << lanes) - 1,
            None,
        ));
        self.emit_ins(X86Instruction::alu_immediate(
            OperandSize::S64,
            0xc1,
            4,
            t,
            3,
            None,
        ));
        let base = unsafe { self.anchors[ANCHOR_SIMD_TABLES].add(table) } as i64;
        self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, base));
        self.emit_ins(X86Instruction::alu(
            OperandSize::S64,
            0x01,
            t,
            REGISTER_SCRATCH,
            None,
        ));
        self.emit_ins(X86Instruction::pop(t));
        self.emit_vex(V::pmovzxbd(SCRATCH_YMM, REGISTER_SCRATCH, 0));
        self.simd.r11 = R11::Unknown;
        let ra = self.simd_pinned(op.va);
        let rd = self.simd_destination(op.vd);
        self.emit_vex(V::permd(rd, SCRATCH_YMM, VexRm::Reg(ra)));
        if op.subop == simd::VCOMPRESS {
            // Lanes from the count on are zero
            self.emit_simd_lane_count(op, lanes);
            self.emit_vex(V::movd_from_gpr(SCRATCH_YMM, REGISTER_MAP[op.dst as usize]));
            self.emit_vex(V::pbroadcastd(SCRATCH_YMM, SCRATCH_YMM));
            let iota = self.simd_table(TAB_IOTA + if size == 2 { 0 } else { 32 });
            self.emit_vex(V::alu(0x66, SCRATCH_YMM, SCRATCH_YMM, iota));
        } else {
            // Lanes whose bit is clear are zero
            self.emit_simd_mask_vector(size, src);
        }
        self.emit_vex(V::alu(VPAND, rd, rd, VexRm::Reg(SCRATCH_YMM)));
    }

    /// vpmovzx* / vpmovsx* opcode (map 0F38) for VZEXT / VSEXT
    fn widen_opcode(op: &VecInsn) -> u8 {
        let (from, to) = (op.off & 3, op.off >> 2);
        let index = match (from, to) {
            (0, 1) => 0,
            (0, 2) => 1,
            (0, 3) => 2,
            (1, 2) => 3,
            (1, 3) => 4,
            _ => 5,
        };
        (if op.subop == simd::VSEXT { 0x20 } else { 0x30 }) + index
    }

    /// (map, opcode) of the unpacks and packs
    fn shuffle_opcode(op: &VecInsn) -> (u8, u8) {
        use crate::simd::*;
        match (op.subop, op.off) {
            (VUNPCKLO, k) => (1, [0x60, 0x61, 0x62, 0x6c][k as usize]),
            (VUNPCKHI, k) => (1, [0x68, 0x69, 0x6a, 0x6d][k as usize]),
            (VPACKSS, 0) => (1, 0x63),
            (VPACKSS, _) => (1, 0x6b),
            (VPACKUS, 0) => (1, 0x67),
            _ => (2, 0x2b),
        }
    }

    /// popcnt without AVX-512: nibble counts with vpshufb, then summed into wider lanes (vpmaddubsw,
    /// vpmaddwd, vpsadbw)
    fn emit_simd_popcnt_avx2(&mut self, vd: u8, ra: u8, size: u8) {
        let (t, u) = (self.simd_temp(), self.simd_temp());
        let s = SCRATCH_YMM;
        self.emit_vex(V::alu(VPAND, t, ra, self.simd_table(TAB_0F)));
        self.emit_vex(V::ymm(1, 1, 0x71, 2, s, VexRm::Reg(ra), Some(4)));
        self.emit_vex(V::alu(VPAND, s, s, self.simd_table(TAB_0F)));
        self.emit_vex(V::load_rip(u, unsafe {
            self.anchors[ANCHOR_SIMD_TABLES].add(TAB_NIB)
        }));
        self.emit_vex(V::pshufb(t, u, VexRm::Reg(t)));
        self.emit_vex(V::pshufb(s, u, VexRm::Reg(s)));
        let rd = self.simd_destination(vd);
        self.emit_vex(V::alu(0xfc, rd, t, VexRm::Reg(s)));
        match size {
            0 => {}
            1 => self.emit_vex(V::ymm(1, 2, 0x04, rd, rd, self.simd_table(TAB_ONE8), None)),
            2 => {
                self.emit_vex(V::ymm(1, 2, 0x04, rd, rd, self.simd_table(TAB_ONE8), None));
                self.emit_vex(V::alu(0xf5, rd, rd, self.simd_table(TAB_ONE16)));
            }
            _ => {
                self.emit_vex(V::alu(VPXOR, t, t, VexRm::Reg(t)));
                self.emit_vex(V::ymm(1, 1, 0xf6, rd, rd, VexRm::Reg(t), None));
            }
        }
    }

    /// 512 bit zext / sext without AVX-512: the high half widens the next part of the source
    fn emit_simd_widen_split(&mut self, op: &VecInsn) {
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        let ratio = 1u8 << ((op.off >> 2) - (op.off & 3));
        let ra = self.simd_pinned(op.va);
        // The high half's source first: the destination may alias the source
        match ratio {
            2 => self.emit_vex(V::extracti128(SCRATCH_YMM, ra, 1)),
            _ => self.emit_vex(V::ymm(
                1,
                1,
                0x73,
                3,
                SCRATCH_YMM,
                VexRm::Reg(ra),
                Some(32 / ratio),
            )),
        }
        let opcode = Self::widen_opcode(op);
        let hi = self.simd_destination(op.vd + 1);
        self.emit_vex(V::ymm(1, 2, opcode, hi, 0, VexRm::Reg(SCRATCH_YMM), None));
        self.simd_pin(hi);
        let lo = self.simd_destination(op.vd);
        self.emit_vex(V::ymm(1, 2, opcode, lo, 0, VexRm::Reg(ra), None));
        self.simd.pinned = [false; CACHE_REGS];
    }

    /// (map, opcode) of the lane-wise core operations (VEX and EVEX share them)
    fn core_opcode(op: &VecInsn) -> (u8, u8) {
        use crate::simd::*;
        let (size, unsigned) = ((op.off & 3) as usize, op.off & 4 != 0);
        match op.subop {
            VMIN if unsigned => [(1, 0xda), (2, 0x3a), (2, 0x3b), (2, 0x3b)][size],
            VMIN => [(2, 0x38), (1, 0xea), (2, 0x39), (2, 0x39)][size],
            VMAX if unsigned => [(1, 0xde), (2, 0x3e), (2, 0x3f), (2, 0x3f)][size],
            VMAX => [(2, 0x3c), (1, 0xee), (2, 0x3d), (2, 0x3d)][size],
            VADDS => [[(1, 0xec), (1, 0xed)], [(1, 0xdc), (1, 0xdd)]][unsigned as usize][size],
            VSUBS => [[(1, 0xe8), (1, 0xe9)], [(1, 0xd8), (1, 0xd9)]][unsigned as usize][size],
            VAVG => [(1, 0xe0), (1, 0xe3)][op.off as usize],
            VADD8 => (1, 0xfc),
            VSUB8 => (1, 0xf8),
            VMULHU16 => (1, 0xe4),
            VMULHRS16 => (2, 0x0b),
            VMULS32 => (2, 0x28),
            VMADDUBS16 => (2, 0x04),
            VSADBW => (1, 0xf6),
            _ => (3, 0x0f), // VALIGNR
        }
    }

    /// vpmov{wb,db,qb,dw,qw,qd} opcode (F3 0F38)
    fn trunc_opcode(op: &VecInsn) -> u8 {
        match (op.off & 3, op.off >> 2) {
            (1, 0) => 0x30,
            (2, 0) => 0x31,
            (3, 0) => 0x32,
            (2, 1) => 0x33,
            (3, 1) => 0x34,
            _ => 0x35,
        }
    }

    /// (map, opcode, W, immediate) of the VBMI2 funnel shifts
    fn funnel_opcode(op: &VecInsn) -> (u8, u8, bool, Option<u8>) {
        use crate::simd::*;
        let size = if matches!(op.subop, VSHLD | VSHRD) {
            op.off >> 6
        } else {
            op.off
        } as u8;
        let base = match op.subop {
            VSHLD | VSHLDV => 0x70,
            _ => 0x72,
        } + (size > 1) as u8;
        let map = if matches!(op.subop, VSHLD | VSHRD) {
            3
        } else {
            2
        };
        let imm = matches!(op.subop, VSHLD | VSHRD).then_some((op.off & 63) as u8);
        (map, base, size != 2, imm)
    }

    /// A lane mask (all ones lanes of `size`) to the scalar register `target`
    fn emit_simd_movemask(&mut self, r: u8, size: usize, target: X86Register) {
        match size {
            0 => self.emit_vex(V::pmovmskb(target, r)),
            1 => {
                self.emit_vex(V::alu(0x63, r, r, VexRm::Reg(r)));
                self.emit_vex(V::permq(r, r, 0xd8));
                self.emit_vex(V::pmovmskb(target, r));
                self.emit_ins(X86Instruction::alu_immediate(
                    OperandSize::S32,
                    0x81,
                    4,
                    target,
                    0xffff,
                    None,
                ));
            }
            _ => self.emit_vex(V::movmskps(target, r, size == 3)),
        }
    }

    /// Immediate shift of lanes of `1 << size` bytes (0x71 / 0x72 / 0x73, /6 left, /2 right)
    fn shift_imm(size: u8, left: bool, dst: u8, src: u8, n: u8) -> V {
        V::ymm(
            1,
            1,
            0x70 + size,
            if left { 6 } else { 2 },
            dst,
            VexRm::Reg(src),
            Some(n),
        )
    }

    /// shld / shrd by an immediate without VBMI2: two immediate shifts and an or
    fn emit_simd_funnel_imm_avx2(&mut self, op: &VecInsn) {
        let (size, c) = ((op.off >> 6) as u8, (op.off & 63) as u8);
        let bits = 8u8 << size;
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        let left = op.subop == simd::VSHLD;
        if c == 0 {
            let rd = self.simd_destination(op.vd);
            if rd != ra {
                self.emit_vex(V::mov(rd, ra));
            }
            return;
        }
        self.emit_vex(Self::shift_imm(size, !left, SCRATCH_YMM, rb, bits - c));
        let rd = self.simd_destination(op.vd);
        self.emit_vex(Self::shift_imm(size, left, rd, ra, c));
        self.emit_vex(V::alu(VPOR, rd, rd, VexRm::Reg(SCRATCH_YMM)));
    }

    /// shldv / shrdv of u32 / u64 lanes without VBMI2: c = b mod w, (d << c) | (a >> (w - c))
    fn emit_simd_funnel_var_avx2(&mut self, op: &VecInsn) {
        let q = op.off == 3;
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        let rd0 = self.simd_pinned(op.vd);
        let (c, t) = (self.simd_temp(), self.simd_temp());
        let (mask, width) = if q {
            (
                VexRm::Rip(self.anchors[ANCHOR_SIMD_C63Q]),
                self.simd_table(TAB_C64Q),
            )
        } else {
            (
                VexRm::Rip(self.anchors[ANCHOR_SIMD_C31]),
                VexRm::Rip(self.anchors[ANCHOR_SIMD_C32]),
            )
        };
        let shift = |left: bool, dst: u8, src: u8, count: u8| {
            let opcode = if left { 0x47 } else { 0x45 };
            if q {
                V::alu38w(opcode, dst, src, VexRm::Reg(count))
            } else {
                V::alu38(opcode, dst, src, VexRm::Reg(count))
            }
        };
        let left = op.subop == simd::VSHLDV;
        self.emit_vex(V::alu(VPAND, c, rb, mask));
        // ymm15 = w - c (a shift by w gives 0, so c = 0 keeps d)
        self.emit_vex(V::alu(
            VPXOR,
            SCRATCH_YMM,
            SCRATCH_YMM,
            VexRm::Reg(SCRATCH_YMM),
        ));
        self.emit_vex(V::alu(VPOR, SCRATCH_YMM, SCRATCH_YMM, width));
        self.emit_vex(V::alu(
            if q { 0xfb } else { VPSUBD },
            SCRATCH_YMM,
            SCRATCH_YMM,
            VexRm::Reg(c),
        ));
        let rd = if q {
            self.emit_simd_shiftv64(!left, t, ra, VexRm::Reg(SCRATCH_YMM));
            let rd = self.simd_destination(op.vd);
            self.emit_simd_shiftv64(left, rd, rd, VexRm::Reg(c));
            rd
        } else {
            self.emit_vex(shift(!left, t, ra, SCRATCH_YMM));
            let rd = self.simd_destination(op.vd);
            self.emit_vex(shift(left, rd, rd, c));
            rd
        };
        debug_assert_eq!(rd, rd0);
        self.emit_vex(V::alu(VPOR, rd, rd, VexRm::Reg(t)));
    }

    /// vpsllvq (`left`) / vpsrlvq. Under Rosetta 2 (x86-64 macOS builds only) the lanes with counts
    /// above 63 are zeroed explicitly: the in range mask is computed first (the count may alias
    /// `dst`), then and-ed into the result. Every register operand must be pinned.
    fn emit_simd_shiftv64(&mut self, left: bool, dst: u8, src: u8, count: VexRm) {
        let opcode = if left { 0x47 } else { 0x45 };
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        if self.simd.rosetta {
            self.simd_pin(dst);
            let m = self.simd_temp();
            self.emit_vex(V::load_rip(m, unsafe {
                self.anchors[ANCHOR_SIMD_TABLES].add(TAB_NOT63Q)
            }));
            self.emit_vex(V::alu(VPAND, m, m, count));
            self.emit_vex(V::alu38(0x29, m, m, self.simd_table(TAB_ZERO)));
            self.emit_vex(V::alu38w(opcode, dst, src, count));
            self.emit_vex(V::alu(VPAND, dst, dst, VexRm::Reg(m)));
            self.simd_unpin(m);
            return;
        }
        self.emit_vex(V::alu38w(opcode, dst, src, count));
    }

    /// Masked load / store: one call which translates the span of the set lanes and copies them
    fn emit_simd_transfer_masked(&mut self, op: &VecInsn) {
        self.simd_flush(true);
        let store = op.subop == simd::VSTOREM;
        let slot = if store { op.va } else { op.vd };
        let spec =
            store as u64 | (slot as u64) << 8 | (op.vb as u64) << 16 | (op.halves() as u64) << 24;
        self.emit_ins(X86Instruction::load_immediate(
            REGISTER_SCRATCH,
            op.off as i64,
        ));
        self.emit_ins(X86Instruction::alu(
            OperandSize::S64,
            0x01,
            REGISTER_MAP[op.src as usize],
            REGISTER_SCRATCH,
            None,
        ));
        self.simd.stats.translations += 1;
        self.emit_rust_call(
            Value::Constant64(simd::jit_transfer_masked as *const u8 as i64, false),
            &[
                // The registers first (the mask may live in an argument register)
                Argument {
                    index: 5,
                    value: Value::Register(REGISTER_MAP[op.dst as usize]),
                },
                Argument {
                    index: 3,
                    value: Value::Register(REGISTER_SCRATCH),
                },
                Argument {
                    index: 4,
                    value: Value::Constant64(spec as i64, false),
                },
                Argument {
                    index: 2,
                    value: Value::RegisterIndirect(
                        REGISTER_PTR_TO_VM,
                        self.slot_in_vm(RuntimeEnvironmentSlot::SimdVectorBase),
                        false,
                    ),
                },
                Argument {
                    index: 1,
                    value: Value::RegisterIndirect(
                        REGISTER_PTR_TO_VM,
                        self.slot_in_vm(RuntimeEnvironmentSlot::MemoryMapping),
                        false,
                    ),
                },
                Argument {
                    index: 0,
                    value: Value::RegisterPlusConstant32(
                        REGISTER_PTR_TO_VM,
                        self.slot_in_vm(RuntimeEnvironmentSlot::ProgramResult),
                        false,
                    ),
                },
            ],
            None,
        );
        self.emit_check_result_is_ok();
        self.emit_ins(X86Instruction::load_immediate(
            REGISTER_SCRATCH,
            self.pc as i64,
        ));
        self.emit_ins(X86Instruction::conditional_jump_immediate(
            0x85,
            self.relative_to_anchor(ANCHOR_THROW_EXCEPTION, 6),
        ));
        self.simd.r11 = R11::Unknown;
    }

    /// Truncation of one 256 bit register without AVX-512 into a new temporary (low part, the rest
    /// zero): every 128 bit lane's kept bytes packed with vpshufb, then the two lanes joined by a
    /// VEX.128 unpack (which zeroes the upper half)
    fn emit_simd_trunc_half(&mut self, ra: u8, imm: u8) -> u8 {
        let (from, to) = (imm & 3, imm >> 2);
        let k = TRUNC_PAIRS.iter().position(|p| *p == (from, to)).unwrap();
        let t = self.simd_temp();
        self.emit_vex(V::pshufb(t, ra, self.simd_table(TAB_TRUNC + 32 * k)));
        self.emit_vex(V::extracti128(SCRATCH_YMM, t, 1));
        let opcode = match from - to {
            1 => 0x6c,
            2 => 0x62,
            _ => 0x61,
        };
        self.emit_vex(V {
            l: false,
            ..V::alu(opcode, t, t, VexRm::Reg(SCRATCH_YMM))
        });
        t
    }

    /// 512 bit trunc without AVX-512: both halves truncated, joined into the low slot, the high slot
    /// zero
    fn emit_simd_trunc_split(&mut self, op: &VecInsn) {
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        let imm = op.off as u8;
        let (lo_src, hi_src) = (self.simd_pinned(op.va), self.simd_pinned(op.va + 1));
        let lo = self.emit_simd_trunc_half(lo_src, imm);
        let hi = self.emit_simd_trunc_half(hi_src, imm);
        match (imm & 3) - (imm >> 2) {
            1 => self.emit_vex(V::inserti128(lo, lo, hi, 1)),
            2 => self.emit_vex(V {
                l: false,
                ..V::alu(0x6c, lo, lo, VexRm::Reg(hi))
            }),
            _ => self.emit_vex(V {
                l: false,
                ..V::alu(0x62, lo, lo, VexRm::Reg(hi))
            }),
        }
        self.emit_vex(V::alu(VPXOR, hi, hi, VexRm::Reg(hi)));
        self.simd_rebind(op.vd, lo);
        self.simd_rebind(op.vd + 1, hi);
        self.simd.pinned = [false; CACHE_REGS];
    }

    /// testm without AVX-512 into the scalar register `target`
    fn emit_simd_testm_avx2(&mut self, va: u8, vb: u8, off: i16, target: X86Register) {
        let ra = self.simd_pinned(va);
        let b = if vb == va {
            VexRm::Reg(ra)
        } else {
            self.simd_source(vb)
        };
        let (t, z) = (self.simd_temp(), self.simd_temp());
        let size = (off & 3) as usize;
        self.emit_vex(V::alu(VPAND, t, ra, b));
        self.emit_vex(V::alu(VPXOR, z, z, VexRm::Reg(z)));
        self.emit_vex(match size {
            0 => V::alu(0x74, t, t, VexRm::Reg(z)),
            1 => V::alu(0x75, t, t, VexRm::Reg(z)),
            2 => V::alu(0x76, t, t, VexRm::Reg(z)),
            _ => V::alu38(0x29, t, t, VexRm::Reg(z)),
        });
        self.emit_simd_movemask(t, size, target);
        if off & 4 == 0 {
            let lanes = 32u32 >> size;
            let lane_mask = if lanes == 32 {
                u32::MAX as i64
            } else {
                (1i64 << lanes) - 1
            };
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S32,
                0x81,
                6,
                target,
                lane_mask,
                None,
            ));
        }
    }

    /// 512 bit testm without AVX-512: two halves, the high half's mask in r11
    fn emit_simd_testm_split(&mut self, op: &VecInsn) {
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        let dst = REGISTER_MAP[op.dst as usize];
        let lanes = (32u32 >> (op.off & 3)) as i64;
        self.emit_simd_testm_avx2(op.va, op.vb, op.off, dst);
        self.simd.pinned = [false; CACHE_REGS];
        self.emit_simd_testm_avx2(op.va + 1, op.vb + 1, op.off, REGISTER_SCRATCH);
        self.simd.r11 = R11::Unknown;
        self.emit_ins(X86Instruction::alu_immediate(
            OperandSize::S64,
            0xc1,
            4,
            REGISTER_SCRATCH,
            lanes,
            None,
        ));
        self.emit_ins(X86Instruction::alu(
            OperandSize::S64,
            0x09,
            REGISTER_SCRATCH,
            dst,
            None,
        ));
        self.simd.pinned = [false; CACHE_REGS];
    }

    fn simd_unpin(&mut self, reg: u8) {
        self.simd.pinned[reg as usize] = false;
    }

    /// Byte table lookup without VBMI: every 128 bit lane of the table broadcast and indexed with
    /// vpshufb (index bits 0..4), then selected by the higher index bits with a vpblendvb tree.
    /// `idx` holds byte indices below 16 * lanes.len(); returns a temporary
    fn emit_simd_lookup(&mut self, lanes: &[(u8, u8)], idx: u8) -> u8 {
        if lanes.len() == 1 {
            let (reg, lane) = lanes[0];
            let t = self.simd_temp();
            self.emit_vex(V::permq(t, reg, if lane == 0 { 0x44 } else { 0xee }));
            self.emit_vex(V::pshufb(t, t, VexRm::Reg(idx)));
            return t;
        }
        let half = lanes.len() / 2;
        let l = self.emit_simd_lookup(&lanes[..half], idx);
        let r = self.emit_simd_lookup(&lanes[half..], idx);
        // Index bit 4 + log2(half) to the sign bit of every byte (a u16 shift by at most 3 keeps bit 7
        // of both bytes inside their byte)
        let bit = 4 + half.trailing_zeros() as u8;
        self.emit_vex(V::ymm(
            1,
            1,
            0x71,
            6,
            SCRATCH_YMM,
            VexRm::Reg(idx),
            Some(7 - bit),
        ));
        self.emit_vex(V::pblendvb(l, l, r, SCRATCH_YMM));
        self.simd_unpin(r);
        l
    }

    /// permb / permi2b / permw without VBMI / AVX-512 (both widths) as byte table lookups
    fn emit_simd_byte_table(&mut self, op: &VecInsn) {
        use crate::simd::*;
        self.simd_ensure_base();
        self.simd.pinned = [false; CACHE_REGS];
        let h = op.halves() as u8;
        let mut table_slots: Vec<u8> = (0..h).map(|k| op.va + k).collect();
        if op.subop == VPERMI2B {
            table_slots.extend((0..h).map(|k| op.vb + k));
        }
        let mut lanes = Vec::new();
        for slot in table_slots {
            let reg = self.simd_pinned(slot);
            lanes.push((reg, 0));
            lanes.push((reg, 1));
        }
        let index_slot = if op.subop == VPERMI2B { op.vd } else { op.vb };
        let mut out = Vec::new();
        for k in 0..h {
            let idx = self.simd_pinned(index_slot + k);
            let bi = self.simd_temp();
            if op.subop == VPERMW {
                // u16 index w -> bytes (2 w, 2 w + 1)
                self.emit_vex(V::alu(
                    VPAND,
                    bi,
                    idx,
                    self.simd_table(if h == 1 { TAB_W15 } else { TAB_W31 }),
                ));
                self.emit_vex(V::ymm(1, 1, 0x71, 6, bi, VexRm::Reg(bi), Some(1)));
                self.emit_vex(V::ymm(1, 1, 0x71, 6, SCRATCH_YMM, VexRm::Reg(bi), Some(8)));
                self.emit_vex(V::alu(VPOR, bi, bi, VexRm::Reg(SCRATCH_YMM)));
                self.emit_vex(V::alu(VPADDW, bi, bi, self.simd_table(TAB_W0100)));
            } else {
                let m = match lanes.len() {
                    2 => 0,
                    4 => 1,
                    _ => 2,
                };
                self.emit_vex(V::alu(VPAND, bi, idx, self.simd_table(TAB_BMASK + 32 * m)));
            }
            let r = self.emit_simd_lookup(&lanes, bi);
            self.simd_unpin(bi);
            out.push(r);
        }
        for (k, r) in out.into_iter().enumerate() {
            self.simd_rebind(op.vd + k as u8, r);
        }
        self.simd.pinned = [false; CACHE_REGS];
    }

    /// u16 variable shifts without AVX-512BW, from u32 variable shifts on the even and odd u16
    /// lanes; `out` must not be an input
    fn emit_simd_var16(&mut self, subop: u8, out: u8, x: u8, c: u8) {
        use crate::simd::*;
        let (t, u, s) = (self.simd_temp(), self.simd_temp(), SCRATCH_YMM);
        let (lo16, hi16) = (self.simd_table(TAB_LO16), self.simd_table(TAB_HI16));
        let r = VexRm::Reg;
        self.emit_vex(V::alu(VPAND, s, c, lo16));
        self.emit_vex(V::psrld(u, c, 16));
        match subop {
            VSRLV16 => {
                self.emit_vex(V::alu(VPAND, t, x, lo16));
                self.emit_vex(V::alu38(0x45, t, t, r(s)));
                self.emit_vex(V::psrld(s, x, 16));
                self.emit_vex(V::alu38(0x45, s, s, r(u)));
                self.emit_vex(V::pslld(s, s, 16));
            }
            VSLLV16 => {
                self.emit_vex(V::alu(VPAND, t, x, lo16));
                self.emit_vex(V::alu38(0x47, t, t, r(s)));
                self.emit_vex(V::alu(VPAND, t, t, lo16));
                self.emit_vex(V::alu(VPAND, s, x, hi16));
                self.emit_vex(V::alu38(0x47, s, s, r(u)));
                self.emit_vex(V::alu(VPAND, s, s, hi16));
            }
            _ => {
                self.emit_vex(V::pslld(t, x, 16));
                self.emit_vex(V::psrad(t, t, 16));
                self.emit_vex(V::alu38(0x46, t, t, r(s)));
                self.emit_vex(V::alu(VPAND, t, t, lo16));
                self.emit_vex(V::psrad(s, x, 16));
                self.emit_vex(V::alu38(0x46, s, s, r(u)));
                self.emit_vex(V::pslld(s, s, 16));
            }
        }
        self.emit_vex(V::alu(VPOR, out, t, r(s)));
        self.simd_unpin(t);
        self.simd_unpin(u);
    }

    /// shldv / shrdv of u16 lanes without VBMI2: c = b & 15, then the u16 variable shifts
    fn emit_simd_funnel16_avx2(&mut self, op: &VecInsn) {
        use crate::simd::*;
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        let rd0 = self.simd_pinned(op.vd);
        let (c, w) = (self.simd_temp(), self.simd_temp());
        self.emit_vex(V::alu(VPAND, c, rb, self.simd_table(TAB_W15)));
        self.emit_vex(V::load_rip(w, unsafe {
            self.anchors[ANCHOR_SIMD_TABLES].add(TAB_C16W)
        }));
        self.emit_vex(V::alu(VPSUBW, w, w, VexRm::Reg(c)));
        let (p, q) = (self.simd_temp(), self.simd_temp());
        let (first, second) = if op.subop == VSHLDV {
            (VSLLV16, VSRLV16)
        } else {
            (VSRLV16, VSLLV16)
        };
        self.emit_simd_var16(first, p, rd0, c);
        self.emit_simd_var16(second, q, ra, w);
        let rd = self.simd_destination(op.vd);
        self.emit_vex(V::alu(VPOR, rd, p, VexRm::Reg(q)));
    }

    /// lzcnt without AVX-512CD: smear the bits right, then width - popcnt
    fn emit_simd_lzcnt_avx2(&mut self, op: &VecInsn) {
        let q = op.off == 3;
        let ra = self.simd_pinned(op.va);
        let t = self.simd_temp();
        self.emit_vex(V::mov(t, ra));
        for sh in [1u8, 2, 4, 8, 16, 32] {
            if sh == 32 && !q {
                break;
            }
            self.emit_vex(if q {
                V::psrlq(SCRATCH_YMM, t, sh)
            } else {
                V::psrld(SCRATCH_YMM, t, sh)
            });
            self.emit_vex(V::alu(VPOR, t, t, VexRm::Reg(SCRATCH_YMM)));
        }
        self.emit_simd_popcnt_avx2(op.vd, t, op.off as u8);
        let rd = self.simd.slots[op.vd as usize].reg.unwrap();
        let width = if q {
            unsafe { self.anchors[ANCHOR_SIMD_TABLES].add(TAB_C64Q) }
        } else {
            self.anchors[ANCHOR_SIMD_C32]
        };
        self.emit_vex(V::load_rip(SCRATCH_YMM, width));
        self.emit_vex(V::alu(
            if q { 0xfb } else { VPSUBD },
            rd,
            SCRATCH_YMM,
            VexRm::Reg(rd),
        ));
    }

    /// multishift without VBMI: per byte position j, the u64 lanes rotated right by the control byte
    /// j (& 63), the low byte placed at j
    fn emit_simd_multishift_avx2(&mut self, op: &VecInsn) {
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        let (acc, t, u, k) = (
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
        );
        let (c63, c64, ffq) = (
            VexRm::Rip(self.anchors[ANCHOR_SIMD_C63Q]),
            self.simd_table(TAB_C64Q),
            self.simd_table(TAB_FFQ),
        );
        let r = VexRm::Reg;
        self.emit_vex(V::alu(VPXOR, acc, acc, r(acc)));
        for j in 0..8u8 {
            let s = SCRATCH_YMM;
            if j == 0 {
                self.emit_vex(V::alu(VPAND, s, rb, c63));
            } else {
                self.emit_vex(V::psrlq(s, rb, 8 * j));
                self.emit_vex(V::alu(VPAND, s, s, c63));
            }
            self.emit_simd_shiftv64(false, t, ra, r(s));
            self.emit_vex(V::alu(VPXOR, k, k, r(k)));
            self.emit_vex(V::alu(VPOR, k, k, c64));
            self.emit_vex(V::alu(0xfb, k, k, r(s)));
            self.emit_simd_shiftv64(true, u, ra, r(k));
            self.emit_vex(V::alu(VPOR, t, t, r(u)));
            self.emit_vex(V::alu(VPAND, t, t, ffq));
            if j > 0 {
                self.emit_vex(V::psllq(t, t, 8 * j));
            }
            self.emit_vex(V::alu(VPOR, acc, acc, r(t)));
        }
        self.simd_rebind(op.vd, acc);
    }

    /// gf2p8mul without GFNI: 8 steps of (b bit k ? a : 0) xor-accumulated, a doubled modulo 0x11b
    fn emit_simd_gf2p8mul_avx2(&mut self, op: &VecInsn) {
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        let (acc, xa, m, z) = (
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
        );
        let r = VexRm::Reg;
        let s = SCRATCH_YMM;
        self.emit_vex(V::alu(VPXOR, acc, acc, r(acc)));
        self.emit_vex(V::alu(VPXOR, z, z, r(z)));
        self.emit_vex(V::mov(xa, ra));
        for k in 0..8usize {
            let bit = self.simd_table(TAB_BYTEBIT + 32 * k);
            self.emit_vex(V::alu(VPAND, m, rb, bit));
            self.emit_vex(V::alu(0x74, m, m, bit));
            self.emit_vex(V::alu(VPAND, m, m, r(xa)));
            self.emit_vex(V::alu(VPXOR, acc, acc, r(m)));
            if k < 7 {
                self.emit_vex(V::alu(0x64, s, z, r(xa)));
                self.emit_vex(V::alu(VPAND, s, s, self.simd_table(TAB_1B)));
                self.emit_vex(V::alu(0xfc, xa, xa, r(xa)));
                self.emit_vex(V::alu(VPXOR, xa, xa, r(s)));
            }
        }
        self.simd_rebind(op.vd, acc);
    }

    /// gf2p8affine without GFNI: per result bit i (from 7 down), the matrix byte 7 - i of every u64
    /// broadcast, and-ed with the data, its parity from a nibble table, shifted into the result
    fn emit_simd_gf2p8affine_avx2(&mut self, op: &VecInsn) {
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        let (acc, t, lo, ptab) = (
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
            self.simd_temp(),
        );
        let r = VexRm::Reg;
        let s = SCRATCH_YMM;
        let nib = self.simd_table(TAB_0F);
        self.emit_vex(V::alu(VPXOR, acc, acc, r(acc)));
        self.emit_vex(V::load_rip(ptab, unsafe {
            self.anchors[ANCHOR_SIMD_TABLES].add(TAB_PARITY)
        }));
        for i in (0..8usize).rev() {
            self.emit_vex(V::pshufb(t, rb, self.simd_table(TAB_AFFIDX + 32 * i)));
            self.emit_vex(V::alu(VPAND, t, t, r(ra)));
            self.emit_vex(V::alu(VPAND, lo, t, nib));
            self.emit_vex(V::ymm(1, 1, 0x71, 2, s, r(t), Some(4)));
            self.emit_vex(V::alu(VPAND, s, s, nib));
            self.emit_vex(V::pshufb(lo, ptab, r(lo)));
            self.emit_vex(V::pshufb(s, ptab, r(s)));
            self.emit_vex(V::alu(VPXOR, lo, lo, r(s)));
            self.emit_vex(V::alu(0xfc, acc, acc, r(acc)));
            self.emit_vex(V::alu(VPOR, acc, acc, r(lo)));
        }
        let imm = op.off as u8;
        if imm != 0 {
            self.emit_ins(X86Instruction::load_immediate(
                REGISTER_SCRATCH,
                imm as i64 * 0x0101_0101,
            ));
            self.simd.r11 = R11::Unknown;
            self.emit_vex(V::movd_from_gpr(s, REGISTER_SCRATCH));
            self.emit_vex(V::pbroadcastd(s, s));
            self.emit_vex(V::alu(VPXOR, acc, acc, r(s)));
        }
        self.simd_rebind(op.vd, acc);
    }

    /// conflict of a 256 bit register without AVX-512CD: compared with its lane rotations by k, the
    /// bit 1 << (i - k) set where equal (i >= k)
    fn emit_simd_conflict_avx2(&mut self, op: &VecInsn) {
        let ra = self.simd_pinned(op.va);
        let (acc, t) = (self.simd_temp(), self.simd_temp());
        let r = VexRm::Reg;
        self.emit_vex(V::alu(VPXOR, acc, acc, r(acc)));
        if op.off == 2 {
            for k in 1..8usize {
                self.emit_vex(V::load_rip(SCRATCH_YMM, unsafe {
                    self.anchors[ANCHOR_SIMD_TABLES].add(TAB_CONF_IDX32 + 32 * (k - 1))
                }));
                self.emit_vex(V::permd(t, SCRATCH_YMM, r(ra)));
                self.emit_vex(V::alu(0x76, t, t, r(ra)));
                self.emit_vex(V::alu(
                    VPAND,
                    t,
                    t,
                    self.simd_table(TAB_CONF_BIT32 + 32 * (k - 1)),
                ));
                self.emit_vex(V::alu(VPOR, acc, acc, r(t)));
            }
        } else {
            for k in 1..4u8 {
                let imm = (0..4u8).fold(0, |m, i| m | ((i + 4 - k) % 4) << (2 * i));
                self.emit_vex(V::permq(t, ra, imm));
                self.emit_vex(V::alu38(0x29, t, t, r(ra)));
                self.emit_vex(V::alu(
                    VPAND,
                    t,
                    t,
                    self.simd_table(TAB_CONF_BIT64 + 32 * (k as usize - 1)),
                ));
                self.emit_vex(V::alu(VPOR, acc, acc, r(t)));
            }
        }
        self.simd_rebind(op.vd, acc);
    }

    /// dpbusd without VNNI, exactly: the bytes widened to u16 (a zero extended, b sign extended) in
    /// even / odd positions, vpmaddwd each, accumulated
    fn emit_simd_dpbusd_avx2(&mut self, op: &VecInsn) {
        let (ra, rb) = (self.simd_pinned(op.va), self.simd_pinned(op.vb));
        self.simd_pinned(op.vd);
        let (ae, be, ao) = (self.simd_temp(), self.simd_temp(), self.simd_temp());
        let r = VexRm::Reg;
        let s = SCRATCH_YMM;
        self.emit_vex(V::alu(VPAND, ae, ra, self.simd_table(TAB_W00FF)));
        self.emit_vex(V::ymm(1, 1, 0x71, 6, be, r(rb), Some(8)));
        self.emit_vex(V::ymm(1, 1, 0x71, 4, be, r(be), Some(8)));
        self.emit_vex(V::alu(0xf5, ae, ae, r(be)));
        self.emit_vex(V::ymm(1, 1, 0x71, 2, ao, r(ra), Some(8)));
        self.emit_vex(V::ymm(1, 1, 0x71, 4, s, r(rb), Some(8)));
        self.emit_vex(V::alu(0xf5, ao, ao, r(s)));
        let rd = self.simd_destination(op.vd);
        self.emit_vex(V::alu(VPADDD, rd, rd, r(ae)));
        self.emit_vex(V::alu(VPADDD, rd, rd, r(ao)));
    }

    /// expand without a native or table lowering (8 / 16 bit lanes, 512 bit forms): a branchless
    /// scalar loop from the top lane down, k = rank of the set bit: d[i] = bit i ? a[k--] : 0. The
    /// reads are at k <= i and the writes go to indices above every later read, so the operands may
    /// alias.
    fn emit_simd_expand_scalar(&mut self, op: &VecInsn) {
        let shift = op.off as u8;
        let size = match shift {
            0 => OperandSize::S8,
            1 => OperandSize::S16,
            2 => OperandSize::S32,
            _ => OperandSize::S64,
        };
        let n = (ebpf::VECTOR_WIDTH * op.halves()) as i32 >> shift;
        for slot in [op.vd, op.va] {
            // (in usize: slot 255 + 1 must not wrap to an empty range)
            for s in (slot as usize..slot as usize + op.halves()).map(|s| s as u8) {
                self.simd_drop_slot(s);
                self.simd_drop_pair(s / 2);
            }
        }
        self.simd_ensure_base();
        let mask = REGISTER_MAP[op.src as usize];
        let regs: Vec<X86Register> = REGISTER_MAP[1..7]
            .iter()
            .copied()
            .filter(|r| *r != mask)
            .take(5)
            .collect();
        let (k, idx, t, z, m) = (regs[0], regs[1], regs[2], regs[3], regs[4]);
        for reg in &regs {
            self.emit_ins(X86Instruction::push(*reg, None));
        }
        // m = mask & lanes, k = popcnt(m) - 1
        if n == 64 {
            self.emit_ins(X86Instruction::mov(OperandSize::S64, mask, m));
        } else {
            self.emit_ins(X86Instruction::mov(OperandSize::S32, mask, m));
            if n < 32 {
                self.emit_ins(X86Instruction::alu_immediate(
                    OperandSize::S32,
                    0x81,
                    4,
                    m,
                    (1i64 << n) - 1,
                    None,
                ));
            }
        }
        let (kr, mr) = (k as u8, m as u8);
        self.emit::<u8>(0xf3);
        self.emit::<u8>(0x48 | if kr >= 8 { 4 } else { 0 } | if mr >= 8 { 1 } else { 0 });
        self.emit::<u8>(0x0f);
        self.emit::<u8>(0xb8);
        self.emit::<u8>(0xc0 | (kr & 7) << 3 | (mr & 7));
        self.emit_ins(X86Instruction::alu_immediate(
            OperandSize::S64,
            0x81,
            5,
            k,
            1,
            None,
        ));
        self.emit_ins(X86Instruction::load_immediate(z, 0));
        let (da, dd) = (Self::slot_disp(op.va), Self::slot_disp(op.vd));
        for i in (0..n).rev() {
            self.emit_ins(X86Instruction::mov(OperandSize::S64, k, idx));
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0x81,
                4,
                idx,
                (n - 1) as i64,
                None,
            ));
            self.emit_ins(X86Instruction::load(
                size,
                REGISTER_SCRATCH,
                t,
                X86IndirectAccess::OffsetIndexShift(da, idx, shift),
            ));
            // bt m, i (CF = mask bit i), t = CF ? t : 0, store, k -= CF
            self.emit::<u8>(0x48 | if mr >= 8 { 1 } else { 0 });
            self.emit::<u8>(0x0f);
            self.emit::<u8>(0xba);
            self.emit::<u8>(0xe0 | (mr & 7));
            self.emit::<u8>(i as u8);
            self.emit_ins(X86Instruction::cmov(OperandSize::S64, 0x43, z, t));
            self.emit_ins(X86Instruction::store(
                size,
                t,
                REGISTER_SCRATCH,
                X86IndirectAccess::Offset(dd + (i << shift)),
            ));
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0x81,
                3,
                k,
                0,
                None,
            ));
        }
        for reg in regs.iter().rev() {
            self.emit_ins(X86Instruction::pop(*reg));
        }
    }
}
