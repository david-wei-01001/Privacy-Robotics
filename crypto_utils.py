"""Symmetric building blocks from Section 3 of the paper.

* (Enc, Dec)  AES-128-GCM with a random 96-bit nonce.  Being an AEAD it is IND$-CPA
              *and* "random-key robust": decrypting under an independent random key
              fails except with probability 2^-128.  The server relies on this to tell
              matches from non-matches.
* H0          key-derivation function E(F_p) -> K0, instantiated with HKDF-SHA256.
* F           PRF  K'' x ID -> F_Sh^2 x X x R, instantiated with HMAC-SHA256 in counter mode.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .ec import Point
from .params import ELL, KEY_LEN, NONCE_LEN, P_SH, TAG_LEN

# ----------------------------------------------------------------------------
# (Enc, Dec)
# ----------------------------------------------------------------------------


def random_key() -> bytes:
    return os.urandom(KEY_LEN)


def enc(key: bytes, plaintext: bytes) -> bytes:
    """nonce || ciphertext || tag   (length = len(plaintext) + 28)."""
    nonce = os.urandom(NONCE_LEN)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, None)


def dec(key: bytes, blob: bytes) -> Optional[bytes]:
    """Return the plaintext, or None if authentication fails (wrong key / corrupted)."""
    if len(blob) < NONCE_LEN + TAG_LEN:
        return None
    try:
        return AESGCM(key).decrypt(blob[:NONCE_LEN], blob[NONCE_LEN:], None)
    except InvalidTag:
        return None


# ----------------------------------------------------------------------------
# H0 : E(F_p) -> K0   (HKDF-SHA256, RFC 5869, empty salt)
# ----------------------------------------------------------------------------
_H0_INFO = b"APPLE-PSI-PROTOTYPE-V1-H0"


def H0(point: Point) -> bytes:
    ikm = point.to_bytes()
    prk = hmac.new(b"\x00" * 32, ikm, hashlib.sha256).digest()          # HKDF-Extract
    okm = hmac.new(prk, _H0_INFO + b"\x01", hashlib.sha256).digest()     # HKDF-Expand (1 block)
    return okm[:KEY_LEN]


# ----------------------------------------------------------------------------
# PRF  F(fkey, id) -> (x, z, x0, r)
#   x, z in F_Sh   : x-coordinate and (for synthetics) y-coordinate of the Shamir share
#   x0   in F_ell  : input to the detectable hash function
#   r    in R      : (s_max+1) elements of F_ell, used as the DHF value of a synthetic
# ----------------------------------------------------------------------------


def _prf_stream(key: bytes, id_: bytes, nbytes: int) -> bytes:
    out, counter = b"", 0
    while len(out) < nbytes:
        out += hmac.new(key, b"APPLE-PSI-F" + counter.to_bytes(4, "big") + id_, hashlib.sha256).digest()
        counter += 1
    return out[:nbytes]


def prf_F(fkey: bytes, id_: bytes, s_max: int) -> tuple[int, int, int, tuple[int, ...]]:
    """Deterministic in (fkey, id): duplicate triples map to the *same* Shamir x-coordinate."""
    stream = _prf_stream(fkey, id_, 32 + 32 + 16 + (s_max + 1) * 16)
    x = 1 + int.from_bytes(stream[0:32], "big") % (P_SH - 1)      # never 0 (x=0 would be the secret)
    z = int.from_bytes(stream[32:64], "big") % P_SH
    x0 = int.from_bytes(stream[64:80], "big") % ELL
    r = tuple(int.from_bytes(stream[80 + 16 * i:96 + 16 * i], "big") % ELL for i in range(s_max + 1))
    return x, z, x0, r
