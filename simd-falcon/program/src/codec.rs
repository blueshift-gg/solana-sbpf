//! Signature decompression into signed 16 bit words (Blueshift's
//! blueshift-gg/solana-falcon512 format checks, Firedancer's windowed decoding).

const N: usize = 512;

/// Length of the compressed s2 in a 666 byte signature
pub const COMP_LEN: usize = 625;

/// Leading zeros of a byte (8 for 0)
static CLZ8: [u8; 256] = {
    let mut t = [8u8; 256];
    let mut i = 1;
    while i < 256 {
        t[i] = (i as u8).leading_zeros() as u8;
        i += 1;
    }
    t
};

/// Coefficients whose 12 leading bits hold the whole code (high bits k < 4): `value | length << 26`
/// (value as i16; the length in the top 6 bits needs no masking as a shift amount) indexed by
/// those 12 bits, 0 if the code is longer (or -0, which is invalid)
static SHORT: [u32; 4096] = {
    let mut t = [0u32; 4096];
    let mut i = 0;
    while i < 4096 {
        let unary = i & 0xf;
        if unary != 0 {
            let k = (unary as u8).leading_zeros() - 4;
            let m = (k << 7) | ((i >> 4) & 0x7f) as u32;
            let negative = i & 0x800 != 0;
            if !(negative && m == 0) {
                let v = if negative { (m as i16).wrapping_neg() } else { m as i16 };
                t[i] = v as u16 as u32 | (9 + k) << 26;
            }
        }
        i += 1;
    }
    t
};

/// One coefficient from the top of `w` (at least 24 valid bits): sign, 7 low bits, then the high
/// bits in unary (k zeros and a one, k < 16). Returns (signed value, bits used).
#[inline(always)]
fn decode_one(w: u64) -> Option<(u16, u64)> {
    let header = (w >> 56) as u32;
    let b1 = ((w >> 48) & 0xff) as usize;
    let k = if b1 != 0 {
        CLZ8[b1] as u32
    } else {
        let b2 = ((w >> 40) & 0xff) as usize;
        if b2 == 0 {
            return None;
        }
        8 + CLZ8[b2] as u32
    };
    let m = (k << 7) | (header & 0x7f);
    let v = if header & 0x80 != 0 {
        if m == 0 {
            return None; // -0
        }
        (m as u16).wrapping_neg()
    } else {
        m as u16
    };
    Some((v, 9 + k as u64))
}

/// The rare long code (or an invalid one) at bit `pos`, inline: a call would clobber the registers
/// the fast loop keeps its state in
#[inline(always)]
fn decode_long(padded: &[u8; COMP_LEN + 15], pos: u64) -> Option<(u16, u64)> {
    decode_one(window(padded, pos))
}

/// 57 valid bits from bit `pos` (big-endian), at the top of a u64
#[inline(always)]
fn window(padded: &[u8; COMP_LEN + 15], pos: u64) -> u64 {
    u64::from_be(unsafe { (padded.as_ptr().add((pos >> 3) as usize) as *const u64).read_unaligned() }) << (pos & 7)
}

/// Decodes the compressed s2 into `out[0..N]` as signed values (|s2| < 2048).
///
/// Firedancer's approach (fd_falcon_signature_parse): a 64 bit big-endian window and a table
/// lookup per coefficient instead of one bit per step. Short codes (12 bits at most, nearly all of
/// them) come from one table load; four fit a window. The input is copied into a zero padded
/// buffer, so a window never reads past it and a run past the end fails on the all zero unary part.
#[inline(never)]
pub fn decompress_signature(buf: &[u8; COMP_LEN], out: *mut u16) -> bool {
    let mut padded = [0u8; COMP_LEN + 15];
    padded[..COMP_LEN].copy_from_slice(buf);
    let mut pos = 0u64;
    // An opaque pointer into the padded stack buffer and the table address: kept in registers
    // instead of a frame address or a 64 bit immediate per access. `out` is opaque already (the
    // function isn't inlined)
    let mut out = out;
    let end = unsafe { out.add(N) };
    let bits = core::hint::black_box(padded.as_ptr());
    let short = core::hint::black_box(SHORT.as_ptr());
    // Fast windows while four coefficients remain, then single codes
    let last_window = unsafe { end.sub(4) };
    while out <= last_window {
        // Up to four short codes per window, stored at constant offsets from `out`; a code which
        // isn't short (rare) ends the window and is decoded inline (a call would clobber the
        // registers holding the loop state)
        let mut w = u64::from_be(unsafe { (bits.add((pos >> 3) as usize) as *const u64).read_unaligned() }) << (pos & 7);
        let mut k = 0;
        while k < 4 {
            let e = unsafe { *short.add((w >> 52) as usize) };
            if e == 0 {
                break;
            }
            unsafe { out.add(k).write_volatile(e as u16) };
            let n = (e >> 26) as u64;
            pos += n;
            w <<= n;
            k += 1;
        }
        out = unsafe { out.add(k) };
        if k < 4 {
            let Some((v, n)) = decode_long(&padded, pos) else { return false };
            unsafe { out.write_volatile(v) };
            out = unsafe { out.add(1) };
            pos += n;
        }
    }
    while out != end {
        let Some((v, n)) = decode_long(&padded, pos) else { return false };
        unsafe { out.write_volatile(v) };
        out = unsafe { out.add(1) };
        pos += n;
    }
    let len_bits = 8 * COMP_LEN as u64;
    if pos > len_bits {
        return false;
    }
    // Trailing bits must be zero
    let mut byte = (pos >> 3) as usize;
    if pos & 7 != 0 {
        if padded[byte] & (0xff >> (pos & 7)) != 0 {
            return false;
        }
        byte += 1;
    }
    while byte + 8 <= COMP_LEN {
        if unsafe { (padded.as_ptr().add(byte) as *const u64).read_unaligned() } != 0 {
            return false;
        }
        byte += 8;
    }
    while byte < COMP_LEN {
        if padded[byte] != 0 {
            return false;
        }
        byte += 1;
    }
    true
}
