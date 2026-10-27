//! Writes ../fixtures/sigs.bin: signatures by ../fixtures/falcon.sk, each entry
//! `[signature, zero padded to 666 bytes][u16 LE message length][message]`.
//! Entry 0 is Blueshift's sample_sig.bin with its benchmark message.

use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey, SecretKey};

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
    let sk = falcon512::SecretKey::from_bytes(&std::fs::read(dir.join("falcon.sk")).unwrap()).unwrap();
    let pk = falcon512::PublicKey::from_bytes(&std::fs::read(dir.join("falcon.pk")).unwrap()).unwrap();
    let mut out = Vec::new();
    let mut push = |sig: &[u8], msg: &[u8]| {
        let mut padded = sig.to_vec();
        assert!(padded.len() <= 666 && padded[0] == 0x39);
        padded.resize(666, 0);
        out.extend_from_slice(&padded);
        out.extend_from_slice(&(msg.len() as u16).to_le_bytes());
        out.extend_from_slice(msg);
    };
    push(&std::fs::read(dir.join("sample_sig.bin")).unwrap(), b"deterministic falcon-512 verify benchmark");
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for i in 0..63usize {
        let len = [0, 1, 32, 64, 135, 136, 200, 256, 1000][i % 9];
        let msg: Vec<u8> = (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (seed >> 56) as u8
            })
            .collect();
        let sig = falcon512::detached_sign(&msg, &sk);
        falcon512::verify_detached_signature(&sig, &msg, &pk).unwrap();
        push(sig.as_bytes(), &msg);
    }
    std::fs::write(dir.join("sigs.bin"), &out).unwrap();
    println!("wrote 64 signatures, {} bytes", out.len());
}
