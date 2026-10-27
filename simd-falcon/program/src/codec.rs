//! Signature decompression and hash-to-point, adapted from blueshift-gg/solana-falcon512
//! (src/codec.rs, MIT) to write their output as u32 lanes straight into the vector region.

const N: usize = 512;
const Q: u32 = 12289;

/// Decodes the compressed s2 into `out[0..N]` as values mod Q in [0, Q)
#[inline(always)]
pub fn decompress_signature(buf: &[u8], out: *mut u32) -> bool {
    let mut acc: u64 = 0;
    let mut acc_len: u64 = 0;
    let mut idx_in = 0usize;
    for i in 0..N {
        if idx_in >= buf.len() {
            return false;
        }
        acc = (acc << 8) | buf[idx_in] as u64;
        idx_in += 1;
        let b = acc >> acc_len;
        let s = b & 128;
        let mut m: u64 = b & 127;
        loop {
            if acc_len == 0 {
                if idx_in >= buf.len() {
                    return false;
                }
                acc = (acc << 8) | buf[idx_in] as u64;
                idx_in += 1;
                acc_len = 8;
            }
            acc_len -= 1;
            if ((acc >> acc_len) & 1) != 0 {
                break;
            }
            m += 128;
            if m >= 2048 {
                return false;
            }
        }
        if s != 0 && m == 0 {
            return false;
        }
        let v = if s != 0 { Q - m as u32 } else { m as u32 };
        unsafe { out.add(i).write_volatile(v) };
    }
    if (acc & ((1u64 << acc_len) - 1)) != 0 {
        return false;
    }
    let mut i = idx_in;
    while i + 8 <= buf.len() {
        let chunk = unsafe { (buf.as_ptr().add(i) as *const u64).read_unaligned() };
        if chunk != 0 {
            return false;
        }
        i += 8;
    }
    while i < buf.len() {
        if buf[i] != 0 {
            return false;
        }
        i += 1;
    }
    true
}

