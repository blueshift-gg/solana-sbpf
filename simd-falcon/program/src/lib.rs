//! Falcon-512 verification (compressed signatures, compile time prepared pubkey) using vector
//! intrinsics for SHAKE-256, the NTT, the pointwise product and the norm. Same instruction format and
//! entrypoint as blueshift-gg/solana-falcon512's example program.
//!
//! The vector file is private to the VM: the scalar code exchanges data with it through buffers in
//! guest memory and bulk `load` / `store` (up to 32 slots per call).

#![no_std]

mod codec;
mod iv;
mod shake;
mod generated {
    include!(concat!(env!("OUT_DIR"), "/vector_core.rs"));
}

const N: usize = 512;
const SIG_LEN: usize = 666;
const NONCE_LEN: usize = 40;
const L2_BOUND: u64 = 34_034_726;
const SIG_HEADER: u8 = 0x39;
const ERR_INVALID_DATA: u64 = 0x0000_0003_0000_0000;
const ERR_VERIFY_FAILED: u64 = 3;

#[cfg(target_os = "solana")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Rows of 8 u32 lanes, the memory image of consecutive slots
type Rows<const R: usize> = [[u32; 8]; R];

/// The memory image of four slot pairs as u64 lanes
#[repr(align(32))]
struct Lanes([u64; 32]);

/// The Keccak state lives in the vector file (see build.rs), the absorb input goes through a buffer
/// (rate rows 0..4 as slot pairs, lane x of row y at u64 8 y + x), the samples are compressed out
/// of the state into the candidates
struct VectorSponge<'a> {
    /// Absorb buffer, slots K_IN..K_IN + 8
    input: Lanes,
    /// hash_to_point candidates (u16 < 5q)
    candidates: &'a mut Candidates,
}

/// 512 candidates and the up to 32 values a block may write past them (rounded up)
#[repr(align(32))]
struct Candidates([u16; N + 68]);

impl shake::Sponge for VectorSponge<'_> {
    #[inline(always)]
    fn set_input(&mut self, lane: usize, value: u64) {
        self.input.0[8 * (lane / 5) + lane % 5] = value;
    }
    #[inline(always)]
    fn absorb_permute(&mut self) {
        iv::load(generated::K_IN, 8, self.input.0.as_ptr().cast());
        generated::keccak_absorb();
        generated::keccak_f();
    }
    #[inline(always)]
    fn permute(&mut self) {
        generated::keccak_f();
    }
    #[inline(always)]
    fn squeeze(&mut self) {}
    #[inline(always)]
    fn take(&mut self, n: usize) -> usize {
        unsafe {
            let start = self.candidates.0.as_mut_ptr();
            generated::take_candidates(start.add(n)).offset_from(start) as usize
        }
    }
}

#[inline(never)]
fn verify(sig: &[u8; SIG_LEN], message: &[u8]) -> bool {
    if sig[0] != SIG_HEADER {
        return false;
    }
    let nonce = &sig[1..1 + NONCE_LEN];
    let comp: &[u8; codec::COMP_LEN] = sig[1 + NONCE_LEN..].try_into().unwrap();
    // SHAKE-256 first: its state lives in the slots the polynomial uses afterwards. The candidates
    // and then s2 share one buffer.
    let mut buffer = Candidates([0; N + 68]);
    generated::keccak_init();
    let mut sponge = VectorSponge { input: Lanes([0; 32]), candidates: &mut buffer };
    shake::hash_to_point(&mut sponge, nonce, message);
    iv::load(generated::C_SLOT, 32, buffer.0.as_ptr().cast());
    if !codec::decompress_signature(comp, buffer.0.as_mut_ptr()) {
        return false;
    }
    iv::load(generated::P_SLOT, 32, buffer.0.as_ptr().cast());
    generated::vector_core();
    // ACC1 and ACC2 are slot pairs
    let mut acc: Rows<4> = [[0; 8]; 4];
    iv::store(acc.as_mut_ptr().cast(), generated::ACC1, 4);
    acc.iter().flatten().map(|x| *x as u64).sum::<u64>() <= L2_BOUND
}

/// Instruction data: `[signature (666 bytes)][message]`, no accounts
///
/// # Safety
/// `input` points to the serialized input region provided by the runtime.
#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    let ix_data_len = unsafe { core::ptr::read(input.add(8) as *const u64) } as usize;
    if ix_data_len < SIG_LEN {
        return ERR_INVALID_DATA;
    }
    let data = unsafe { core::slice::from_raw_parts(input.add(16), ix_data_len) };
    let Some((sig, message)) = data.split_first_chunk::<SIG_LEN>() else {
        return ERR_INVALID_DATA;
    };
    if verify(sig, message) {
        0
    } else {
        ERR_VERIFY_FAILED
    }
}
