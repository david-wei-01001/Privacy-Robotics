"""The only two messages of the protocol (paper Section 2.3).

    PData    server -> every client (shipped with an OS update); size ~ n, independent of #clients
    Voucher  client -> server, one per triple; ALL vouchers have exactly Params.voucher_len bytes
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .ec import Point
from .params import CT_RKEY_LEN, ID_LEN, NONCE_H_LEN, POINT_LEN, Params

_MAGIC = b"APSI"
_VERSION = 1


@dataclass
class PData:
    """pdata = (L, P_1, ..., P_n') plus the descriptions of the hash functions H, h_1..h_d."""

    L: Point
    points: List[Point]
    hash_nonce: bytes
    cuckoo_nonces: List[bytes]

    @property
    def table_size(self) -> int:
        return len(self.points)

    # --- client-side validation (client setup, "Obtain pdata") -----------------
    def validate(self) -> None:
        """L and all P_i must be non-zero points and pairwise distinct; otherwise the client aborts."""
        if len(self.hash_nonce) != NONCE_H_LEN or any(len(n) != NONCE_H_LEN for n in self.cuckoo_nonces):
            raise ValueError("bad nonce length")
        if len(self.cuckoo_nonces) < 2:
            raise ValueError("need at least two cuckoo hash functions")
        if self.table_size < len(self.cuckoo_nonces):
            raise ValueError("table too small")
        seen = set()
        for pt in [self.L, *self.points]:
            if pt.is_infinity():
                raise ValueError("pdata contains the point at infinity")
            key = (pt.x, pt.y)
            if key in seen:
                raise ValueError("pdata contains duplicate points")
            seen.add(key)

    # --- serialisation -----------------------------------------------------------
    def to_bytes(self) -> bytes:
        d = len(self.cuckoo_nonces)
        header = _MAGIC + bytes([_VERSION, d]) + self.table_size.to_bytes(4, "big")
        return (header + self.hash_nonce + b"".join(self.cuckoo_nonces)
                + self.L.to_bytes() + b"".join(p.to_bytes() for p in self.points))

    @classmethod
    def from_bytes(cls, data: bytes) -> "PData":
        if data[:4] != _MAGIC or data[4] != _VERSION:
            raise ValueError("not an Apple-PSI-prototype pdata blob")
        d = data[5]
        n = int.from_bytes(data[6:10], "big")
        off = 10
        need = off + NONCE_H_LEN * (1 + d) + POINT_LEN * (1 + n)
        if len(data) != need:
            raise ValueError("pdata has the wrong length")
        hash_nonce = data[off:off + NONCE_H_LEN]
        off += NONCE_H_LEN
        nonces = [data[off + i * NONCE_H_LEN: off + (i + 1) * NONCE_H_LEN] for i in range(d)]
        off += d * NONCE_H_LEN
        L = Point.from_bytes(data[off:off + POINT_LEN])
        off += POINT_LEN
        points = [Point.from_bytes(data[off + i * POINT_LEN: off + (i + 1) * POINT_LEN]) for i in range(n)]
        pdata = cls(L, points, hash_nonce, nonces)
        pdata.validate()
        return pdata


@dataclass
class Voucher:
    """voucher = (id, Q_b, ct_b, Q_{3-b}, ct_{3-b}, rct)   (generalised to d (Q, ct) pairs)."""

    id: bytes
    pairs: List[Tuple[Point, bytes]]
    rct: bytes

    def to_bytes(self) -> bytes:
        out = self.id
        for q, ct in self.pairs:
            out += q.to_bytes() + ct
        return out + self.rct

    @classmethod
    def from_bytes(cls, data: bytes, params: Params) -> "Voucher":
        if len(data) != params.voucher_len:
            raise ValueError(f"voucher must be exactly {params.voucher_len} bytes, got {len(data)}")
        off = ID_LEN
        id_ = data[:off]
        pairs = []
        for _ in range(params.num_hashes):
            q = Point.from_bytes(data[off:off + POINT_LEN])   # rejects points not on the curve
            off += POINT_LEN
            pairs.append((q, data[off:off + CT_RKEY_LEN]))
            off += CT_RKEY_LEN
        return cls(id_, pairs, data[off:])
