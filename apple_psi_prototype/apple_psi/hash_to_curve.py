"""Hash-to-curve for P-256: the random oracle  H : U -> E(F_p) \\ {O}.

Implements the suite  P256_XMD:SHA-256_SSWU_RO_  of RFC 9380 (the final form of the
IETF CFRG draft that the paper cites as [FHSS+20]).  It is unit-tested against the
RFC's published test vectors.
"""
from __future__ import annotations

import hashlib

from .ec import A, B, P, Point

_Z = (-10) % P          # SSWU non-square constant for P-256
_L = 48                 # ceil((ceil(log2 p) + k) / 8) with k = 128
_DST_PREFIX = b"APPLE-PSI-PROTOTYPE-V1-H2C-"


def expand_message_xmd(msg: bytes, dst: bytes, len_in_bytes: int) -> bytes:
    """RFC 9380, section 5.3.1 (hash = SHA-256)."""
    b_in_bytes, r_in_bytes = 32, 64
    ell = -(-len_in_bytes // b_in_bytes)
    if ell > 255 or len_in_bytes > 65535 or len(dst) > 255:
        raise ValueError("expand_message_xmd: invalid lengths")
    dst_prime = dst + bytes([len(dst)])
    msg_prime = bytes(r_in_bytes) + msg + len_in_bytes.to_bytes(2, "big") + b"\x00" + dst_prime
    b0 = hashlib.sha256(msg_prime).digest()
    out = [hashlib.sha256(b0 + b"\x01" + dst_prime).digest()]
    for i in range(2, ell + 1):
        mixed = bytes(x ^ y for x, y in zip(b0, out[-1]))
        out.append(hashlib.sha256(mixed + bytes([i]) + dst_prime).digest())
    return b"".join(out)[:len_in_bytes]


def hash_to_field(msg: bytes, dst: bytes, count: int = 2) -> list[int]:
    uniform = expand_message_xmd(msg, dst, count * _L)
    return [int.from_bytes(uniform[i * _L:(i + 1) * _L], "big") % P for i in range(count)]


def _is_square(x: int) -> bool:
    return pow(x, (P - 1) // 2, P) in (0, 1)


def _map_to_curve_sswu(u: int) -> Point:
    """Simplified SWU map (RFC 9380, section 6.6.2) for A != 0, B != 0."""
    u2 = u * u % P
    tv1 = (_Z * _Z % P * u2 % P * u2 + _Z * u2) % P
    tv1 = pow(tv1, P - 2, P) if tv1 else 0               # inv0
    if tv1 == 0:
        x1 = B * pow(_Z * A % P, P - 2, P) % P
    else:
        x1 = (-B) * pow(A, P - 2, P) % P * (1 + tv1) % P
    gx1 = (x1 * x1 % P * x1 + A * x1 + B) % P
    x2 = _Z * u2 % P * x1 % P
    gx2 = (x2 * x2 % P * x2 + A * x2 + B) % P
    if _is_square(gx1):
        x, y = x1, pow(gx1, (P + 1) // 4, P)
    else:
        x, y = x2, pow(gx2, (P + 1) // 4, P)
    if (u & 1) != (y & 1):                               # sgn0(u) != sgn0(y)
        y = (-y) % P
    return Point.from_xy(x, y)


def hash_to_curve(msg: bytes, dst: bytes) -> Point:
    """Full random-oracle encoding (two field elements, two map calls, add)."""
    u0, u1 = hash_to_field(msg, dst, 2)
    pt = _map_to_curve_sswu(u0) + _map_to_curve_sswu(u1)  # cofactor of P-256 is 1
    if pt.is_infinity():                                  # probability ~2^-256
        raise ValueError("hash_to_curve produced the identity")
    return pt


def hash_item(item: bytes, hash_nonce: bytes) -> Point:
    """H(y) used by the protocol; `hash_nonce` is the 128-bit domain-separation nonce in pdata."""
    return hash_to_curve(item, _DST_PREFIX + hash_nonce)
