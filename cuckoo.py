"""Cuckoo hashing for the server's public data (paper Section 4.1, server setup step 1).

* The hash functions h_1..h_d are domain-separated by random 128-bit nonces that are
  published inside pdata.
* The paper's "Caution": h_1(y) and h_2(y) must never collide (the client would
  otherwise read the same cell twice and the server could decrypt two ciphertexts).  We
  therefore bump a colliding position by +1 (mod table size) until all d are distinct.
* Items that cannot be placed are *dropped* (Remark 3: the protocol tolerates a small
  fraction of false negatives).  `insert` reports which item was lost.
"""
from __future__ import annotations

import hashlib
import random
from typing import Optional, Sequence


def cuckoo_positions(item: bytes, nonces: Sequence[bytes], size: int) -> list[int]:
    """The d *distinct* table positions h_1(item) .. h_d(item)."""
    out: list[int] = []
    for nonce in nonces:
        pos = int.from_bytes(hashlib.sha256(b"APPLE-PSI-CUCKOO" + nonce + item).digest(), "big") % size
        while pos in out:
            pos = (pos + 1) % size
        out.append(pos)
    return out


class CuckooTable:
    def __init__(self, size: int, nonces: Sequence[bytes], max_kicks: int = 500,
                 rng: random.Random | None = None):
        if size < len(nonces):
            raise ValueError("table smaller than the number of hash functions")
        self.size = size
        self.nonces = list(nonces)
        self.max_kicks = max_kicks
        self.slots: list[Optional[bytes]] = [None] * size
        self._rng = rng or random.SystemRandom()

    def positions(self, item: bytes) -> list[int]:
        return cuckoo_positions(item, self.nonces, self.size)

    def insert(self, item: bytes) -> Optional[bytes]:
        """Insert `item`; return None on success or the item that had to be dropped."""
        cur, prev = item, None
        for _ in range(self.max_kicks):
            pos = self.positions(cur)
            for p in pos:
                if self.slots[p] is None:
                    self.slots[p] = cur
                    return None
            choices = [p for p in pos if p != prev] or pos
            p = self._rng.choice(choices)
            cur, self.slots[p] = self.slots[p], cur       # evict a resident item
            prev = p
        return cur                                        # table stays consistent; `cur` is lost

    def lookup(self, item: bytes) -> Optional[int]:
        for p in self.positions(item):
            if self.slots[p] == item:
                return p
        return None
