"""The server side of the ftPSI-AD protocol (paper Sections 4.1 / 4.2).

Lifecycle
---------
    server = Server(params)
    pdata  = server.setup(X)                 # once per database version; publish pdata
    server.ingest(voucher)                   # for every voucher, as soon as it arrives
    out    = server.reveal()                 # periodically ("once a week")

The server stores only its secret alpha plus, for vouchers that decrypted (real or
synthetic matches), a small record.  Non-matching vouchers are classified and dropped
immediately -- exactly the "marks it as non-matching" step of Section 2.3.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .crypto_utils import H0, dec
from .cuckoo import CuckooTable
from .dhf import detect
from .ec import G, Point, random_point, random_scalar
from .hash_to_curve import hash_item
from .messages import PData, Voucher
from .params import ELL, ELL_LEN, KEY_LEN, NONCE_H_LEN, P_SH, SH_LEN, Params
from .shamir import reconstruct


@dataclass
class _MatchRecord:
    id: bytes
    adct: bytes
    x: int
    y: int
    r: Tuple[int, ...]


@dataclass
class ServerOutput:
    """What the server learns (the ftPSI-AD output of Section 2.2)."""

    idlist: List[bytes]                         # every id the client sent (not secret)
    outset_ids: Set[bytes]                      # ids of (real U synthetic) matches
    above_threshold: bool                       # True iff associated data could be decrypted
    revealed: Dict[bytes, bytes] = field(default_factory=dict)   # id -> associated data (real matches)
    synthetic_ids: Set[bytes] = field(default_factory=set)       # ids identified as synthetic
    notes: List[str] = field(default_factory=list)


class Server:
    def __init__(self, params: Params):
        self.params = params
        self._alpha: Optional[int] = None       # the server's short secret key
        self.pdata: Optional[PData] = None
        self.dropped: List[bytes] = []          # items that did not fit in the cuckoo table (false negatives)
        self.idlist: List[bytes] = []
        self._matches: Dict[bytes, _MatchRecord] = {}
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ setup
    def setup(self, X: Iterable[bytes], keep_key: bool = False) -> PData:
        """Server setup, steps 0-4 of Section 4.1.

        keep_key=True re-uses the secret alpha of an earlier setup (an *update*, Section 2.3): vouchers
        computed by clients under the old pdata stay processable and, because the client's shares do not
        depend on pdata, count towards the threshold together with vouchers made under the new pdata.
        """
        p = self.params
        items = sorted(set(X))                                   # step 0: remove duplicates
        if not items:
            raise ValueError("server set X is empty")
        n = len(items)
        size = max(math.ceil(p.table_expansion * n), p.num_hashes + 1)
        hash_nonce = os.urandom(NONCE_H_LEN)
        nonces = [os.urandom(NONCE_H_LEN) for _ in range(p.num_hashes)]

        table = CuckooTable(size, nonces, p.max_kicks)           # step 1: cuckoo table
        self.dropped = [d for d in (table.insert(it) for it in items) if d is not None]

        if self._alpha is None or not keep_key:                  # step 2: secret alpha, L = alpha*G
            self._alpha = random_scalar()
        L = self._alpha * G
        points: List[Point] = []                                 # step 3: P_i = alpha*H(T[i]) or random
        for slot in table.slots:
            if slot is None:
                points.append(random_point())
            else:
                points.append(self._alpha * hash_item(slot, hash_nonce))
        self.pdata = PData(L, points, hash_nonce, nonces)        # step 4
        self.pdata.validate()
        return self.pdata

    # ------------------------------------------------------- voucher processing
    def ingest_bytes(self, data: bytes) -> str:
        return self.ingest(Voucher.from_bytes(data, self.params))

    def ingest(self, v: Voucher) -> str:
        """Process one voucher.  Returns 'non-match', 'match' or 'invalid'."""
        if self._alpha is None:
            raise RuntimeError("call setup() first")
        p = self.params
        self.idlist.append(v.id)
        plaintexts = []
        for q, ct in v.pairs:
            s_hat = self._alpha * q                              # S^_j = alpha * Q_j
            rkey = dec(H0(s_hat), ct)                            # fails unless (L, H(y), P_w) was a DH tuple
            if rkey is None:
                continue
            opened = dec(rkey, v.rct)
            if opened is not None:
                plaintexts.append(opened)
        if not plaintexts:
            self.stats["non-match"] += 1
            return "non-match"
        if len(plaintexts) != 1 or len(plaintexts[0]) != p.rct_plain_len:
            self.stats["invalid"] += 1                           # several ciphertexts opened: invalid voucher
            return "invalid"
        pt = plaintexts[0]
        r = tuple(int.from_bytes(pt[i * ELL_LEN:(i + 1) * ELL_LEN], "big") for i in range(p.s_max + 1))
        off = p.r_len
        adct = pt[off:off + p.adct_len]
        off += p.adct_len
        x = int.from_bytes(pt[off:off + SH_LEN], "big")
        y = int.from_bytes(pt[off + SH_LEN:off + 2 * SH_LEN], "big")
        if not (0 < x < P_SH and 0 <= y < P_SH) or any(c >= ELL for c in r):
            self.stats["invalid"] += 1
            return "invalid"
        self._matches[v.id] = _MatchRecord(v.id, adct, x, y, r)  # same id => same triple => one record
        self.stats["match"] += 1
        return "match"

    # ------------------------------------------------------------------ reveal
    def reveal(self) -> ServerOutput:
        """Steps 1-2 of the ftPSI-AD "server processes the vouchers" procedure (Section 4.2)."""
        p = self.params
        by_x: Dict[int, _MatchRecord] = {}
        for rec in self._matches.values():                       # t' = number of distinct Shamir shares
            by_x.setdefault(rec.x, rec)
        recs = list(by_x.values())
        ids = {r.id for r in recs}
        below = ServerOutput(list(self.idlist), ids, above_threshold=False)

        if len(recs) <= p.t:
            return below
        detected = detect([r.r for r in recs], p.s_max, p.t)
        if detected is None or len(detected) < p.t + 1:
            below.notes.append("detection failed: with high probability <= t real matches")
            return below

        real = [recs[i] for i in detected]
        secret = reconstruct([(r.x, r.y) for r in real[:p.t + 1]])
        if secret >> (8 * KEY_LEN):
            below.notes.append("reconstructed key out of range: vouchers inconsistent")
            return below
        adkey = secret.to_bytes(KEY_LEN, "big")
        revealed = {}
        for r in real:
            ad = dec(adkey, r.adct)
            if ad is not None:
                revealed[r.id] = ad
        if len(revealed) <= p.t:
            below.notes.append("too few decryptable vouchers: detection error or invalid vouchers")
            return below
        out = ServerOutput(list(self.idlist), ids, True, revealed, ids - {r.id for r in real})
        if len(revealed) != len(real):
            out.notes.append(f"{len(real) - len(revealed)} detected voucher(s) failed to decrypt (invalid)")
        return out
