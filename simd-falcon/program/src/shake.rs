// SHAKE-256 hash-to-point driver over a vectorized Keccak-f[1600] (see build.rs for the layout).
//
// Shared by the program and by build.rs, which runs it against a simulation of the generated
// vector code and checks the result against a reference SHAKE-256. No dependencies.

/// Keccak state access, implemented with vector intrinsics in the program
pub trait Sponge {
    /// Writes rate lane `lane` into the absorb buffer
    fn set_input(&mut self, lane: usize, value: u64);
    /// state ^= absorb buffer, then Keccak-f
    fn absorb_permute(&mut self);
    /// Keccak-f
    fn permute(&mut self);
    /// Makes the rate lanes available to `take`
    fn squeeze(&mut self);
    /// Appends the accepted samples of the squeezed block (the 68 big-endian 16 bit words of the
    /// rate, in order, kept if below 5Q) after the first `n` candidates, returns the new count.
    /// May write up to 68 values past the last accepted one.
    fn take(&mut self, n: usize) -> usize;
}

pub const RATE_LANES: usize = 17;
const RATE: usize = 8 * RATE_LANES;
const FALCON_N: usize = 512;

/// Bytes `[offset, offset + 8)` of `a || b` (zero beyond the end) as a little-endian lane
#[inline(always)]
fn read_lane(a: &[u8], b: &[u8], offset: usize) -> u64 {
    if offset + 8 <= a.len() {
        return unsafe { (a.as_ptr().add(offset) as *const u64).read_unaligned() };
    }
    if offset >= a.len() && offset - a.len() + 8 <= b.len() {
        return unsafe { (b.as_ptr().add(offset - a.len()) as *const u64).read_unaligned() };
    }
    let mut lane = 0u64;
    for i in 0..8 {
        let p = offset + i;
        let byte = if p < a.len() {
            a[p]
        } else if p - a.len() < b.len() {
            b[p - a.len()]
        } else {
            0
        };
        lane |= (byte as u64) << (8 * i);
    }
    lane
}

/// Falcon hash_to_point: SHAKE-256(nonce || message), big-endian 16 bit candidates below 5Q are
/// kept (unreduced) until N are found
pub fn hash_to_point<S: Sponge>(s: &mut S, nonce: &[u8], message: &[u8]) {
    let total = nonce.len() + message.len();
    let mut pos = 0;
    while pos + RATE <= total {
        for lane in 0..RATE_LANES {
            s.set_input(lane, read_lane(nonce, message, pos + 8 * lane));
        }
        s.absorb_permute();
        pos += RATE;
    }
    // Last (possibly empty) block with the SHAKE padding: 0x1f after the data, 0x80 in the last byte
    let rest = total - pos;
    for lane in 0..RATE_LANES {
        let start = pos + 8 * lane;
        let mut value = if start < total { read_lane(nonce, message, start) } else { 0 };
        if start < total && total - start < 8 {
            value &= (1u64 << (8 * (total - start))) - 1;
        }
        if rest / 8 == lane {
            value ^= 0x1f << (8 * (rest % 8));
        }
        if lane == RATE_LANES - 1 {
            value ^= 0x80 << 56;
        }
        s.set_input(lane, value);
    }
    s.absorb_permute();
    let mut n = 0;
    loop {
        s.squeeze();
        n = s.take(n);
        if n >= FALCON_N {
            return;
        }
        s.permute();
    }
}
