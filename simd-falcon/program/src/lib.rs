//! Falcon-512 verification (compressed signatures, compile time prepared pubkey) using vector
//! intrinsics for SHAKE-256, the NTT, the pointwise product and the norm. Same instruction format and
//! entrypoint as blueshift-gg/solana-falcon512's example program.

#![no_std]

mod codec;
mod iv;
mod shake;
mod generated {
    include!(concat!(env!("OUT_DIR"), "/vector_core.rs"));
}

const SIG_LEN: usize = 666;
const NONCE_LEN: usize = 40;
const L2_BOUND: u64 = 34_034_726;
const SIG_HEADER: u8 = 0x39;
/// Host view of the vector region (slot n at REGION + 32 n)
const REGION: u64 = 0x6_0000_0000;
const ERR_INVALID_DATA: u64 = 0x0000_0003_0000_0000;
const ERR_VERIFY_FAILED: u64 = 3;

#[cfg(target_os = "solana")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

fn slot_ptr(slot: usize) -> *mut u32 {
    (REGION + 32 * slot as u64) as *mut u32
}

fn lane_sum(slot: usize) -> u64 {
    (0..8).map(|i| unsafe { slot_ptr(slot).add(i).read_volatile() } as u64).sum()
}

/// The Keccak state in the vector region, see build.rs
struct VectorSponge;

impl shake::Sponge for VectorSponge {
    #[inline(always)]
    fn set_input(&mut self, lane: usize, even: u32, odd: u32) {
        unsafe {
            slot_ptr(generated::K_IN_E + lane / 5).add(lane % 5).write_volatile(even);
            slot_ptr(generated::K_IN_O + lane / 5).add(lane % 5).write_volatile(odd);
        }
    }
    #[inline(always)]
    fn absorb_permute(&mut self) {
        generated::keccak_absorb();
        generated::keccak_f();
    }
    #[inline(always)]
    fn permute(&mut self) {
        generated::keccak_f();
    }
    #[inline(always)]
    fn squeeze(&mut self) {
        generated::keccak_squeeze();
    }
    #[inline(always)]
    fn take(&mut self, n: usize) -> usize {
        // The candidates may run up to 68 values past the 512 of C, into slots which the vector code
        // initializes before reading (see build.rs)
        unsafe {
            let start = slot_ptr(generated::C_SLOT) as *mut u16;
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
    // SHAKE-256 first: its state lives in the slots the polynomial uses afterwards
    generated::keccak_init();
    shake::hash_to_point(&mut VectorSponge, nonce, message);
    if !codec::decompress_signature(comp, slot_ptr(generated::P_SLOT) as *mut u16) {
        return false;
    }
    generated::vector_core();
    lane_sum(generated::ACC1) + lane_sum(generated::ACC2) <= L2_BOUND
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
