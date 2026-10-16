"""The client side of the ftPSI-AD protocol (paper Sections 4.1 / 4.2).

    client  = Client(params, pdata)             # validates pdata, picks (hkey, adkey, fkey, polynomial)
    voucher = client.create_voucher(y, ad)      # called for each triple, as soon as it arrives
    ...send voucher.to_bytes() to the server...

The client is *streaming*: each triple is fully processed on arrival, using only pdata and
its own small secret state.  The work and size per voucher are independent of |X| and of
how many triples the client already has.

Multi-device accounts (Section 5): all devices of one user share `ClientState`
(hkey, adkey, fkey, Shamir polynomial, designated synthetic ids).  Because the Shamir
x-coordinate is F(fkey, id), a triple seen on two devices yields the *same* share.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Set

from .crypto_utils import H0, enc, prf_F, random_key
from .cuckoo import cuckoo_positions
from .dhf import DHF
from .ec import G, multi_mul, random_point, random_scalar
from .hash_to_curve import hash_item
from .messages import PData, Voucher
from .params import ELL_LEN, ID_LEN, SH_LEN, Params
from .shamir import SharingPolynomial


@dataclass
class ClientState:
    """The small secret state shared by all of one user's devices."""

    hkey: List[List[int]]            # DHF key: s polynomials of degree <= t-1 over F_ell
    adkey: bytes                     # key that encrypts every associated-data blob
    fkey: bytes                      # PRF key
    poly_coeffs: List[int]           # Shamir polynomial (secret = adkey)
    synthetic_ids: Set[bytes] = field(default_factory=set)   # the set S, designated incrementally

    @classmethod
    def generate(cls, params: Params) -> "ClientState":
        adkey = random_key()
        poly = SharingPolynomial(int.from_bytes(adkey, "big"), params.t)
        return cls(DHF(params.s_max, params.t).key, adkey, random_key(), poly.coeffs)

    def to_json(self) -> str:
        return json.dumps({
            "hkey": self.hkey, "adkey": self.adkey.hex(), "fkey": self.fkey.hex(),
            "poly": [str(c) for c in self.poly_coeffs],
            "synthetic_ids": sorted(i.hex() for i in self.synthetic_ids)})

    @classmethod
    def from_json(cls, blob: str) -> "ClientState":
        d = json.loads(blob)
        return cls(d["hkey"], bytes.fromhex(d["adkey"]), bytes.fromhex(d["fkey"]),
                   [int(c) for c in d["poly"]], {bytes.fromhex(i) for i in d["synthetic_ids"]})


class Client:
    def __init__(self, params: Params, pdata: PData | bytes, state: ClientState | None = None,
                 synthetic_prob: float = 0.0, rng: random.Random | None = None):
        """
        synthetic_prob : probability that a *new* triple is designated synthetic (the client's policy
                         for choosing S; the paper leaves the distribution to the system design).
                         The total number of synthetics never exceeds params.s_max.
        """
        self.params = params
        self.pdata = PData.from_bytes(pdata) if isinstance(pdata, (bytes, bytearray)) else pdata
        self.pdata.validate()                                    # abort on malformed pdata
        if len(self.pdata.cuckoo_nonces) != params.num_hashes:
            raise ValueError("pdata uses a different number of cuckoo hash functions than Params")
        self.state = state or ClientState.generate(params)
        self._dhf = DHF(params.s_max, params.t, self.state.hkey)
        self._poly = SharingPolynomial(int.from_bytes(self.state.adkey, "big"), params.t, self.state.poly_coeffs)
        self.synthetic_prob = synthetic_prob
        self._rng = rng or random.SystemRandom()

    # ------------------------------------------------------------ public API
    def create_voucher(self, y: bytes, ad: bytes, id_: bytes | None = None) -> Voucher:
        """Voucher for the triple (y, id, ad); applies the synthetic-designation policy."""
        id_ = id_ or os.urandom(ID_LEN)
        if self._designate_synthetic(id_):
            return self.create_synthetic_voucher(id_)
        return self.create_real_voucher(y, ad, id_)

    def _designate_synthetic(self, id_: bytes) -> bool:
        if id_ in self.state.synthetic_ids:                      # keep the decision stable per id
            return True
        if (len(self.state.synthetic_ids) < self.params.s_max
                and self._rng.random() < self.synthetic_prob):
            self.state.synthetic_ids.add(id_)
            return True
        return False

    # ------------------------------------------------------- Case 1: real voucher
    def create_real_voucher(self, y: bytes, ad: bytes, id_: bytes) -> Voucher:
        p = self.params
        if len(ad) != p.ad_len:
            raise ValueError(f"associated data must be exactly {p.ad_len} bytes")
        if len(id_) != ID_LEN:
            raise ValueError("bad id length")
        adct = enc(self.state.adkey, ad)                                      # step 1
        x, _z, x0, _r = prf_F(self.state.fkey, id_, p.s_max)                  # step 2
        r = self._dhf(x0)                                                     # r = DHF(hkey, x0)
        _, share_y = self._poly.share(x)                                      # step 3
        rkey = random_key()                                                   # step 4
        rct = enc(rkey, self._pack(r, adct, x, share_y))

        h_y = hash_item(y, self.pdata.hash_nonce)                             # step 5
        L = self.pdata.L
        pairs = []
        for w in cuckoo_positions(y, self.pdata.cuckoo_nonces, self.pdata.table_size):
            P_w = self.pdata.points[w]
            while True:                                                       # Naor-Reingold self-reduction
                beta, gamma = random_scalar(), random_scalar()
                Q = multi_mul(beta, h_y, gamma, G)                            # Q = beta*H(y) + gamma*G
                S = multi_mul(beta, P_w, gamma, L)                            # S = beta*P_w + gamma*L
                if not Q.is_infinity() and not S.is_infinity():
                    break
            pairs.append((Q, enc(H0(S), rkey)))                               # ct_j = Enc(H0(S_j), rkey)
        self._rng.shuffle(pairs)                                              # step 6: random order
        return Voucher(id_, pairs, rct)

    # --------------------------------------------------- Case 2: synthetic voucher
    def create_synthetic_voucher(self, id_: bytes | None = None) -> Voucher:
        p = self.params
        id_ = id_ or os.urandom(ID_LEN)
        ckey = random_key()
        adct = enc(ckey, bytes(p.ad_len))                                     # step 1: Enc(ckey, 000..0)
        x, z, _x0, r = prf_F(self.state.fkey, id_, p.s_max)                   # step 2: r is pseudorandom in R
        rkey = random_key()                                                   # steps 3-4: dummy share (x, z)
        rct = enc(rkey, self._pack(r, adct, x, z))

        beta = random_scalar()                                                # step 5
        pairs = [(beta * G, enc(H0(beta * self.pdata.L), rkey))]              # (Q1, S1 = beta*L = alpha*Q1)
        for _ in range(p.num_hashes - 1):                                     # remaining pairs: random (Q, S)
            pairs.append((random_point(), enc(H0(random_point()), rkey)))
        self._rng.shuffle(pairs)                                              # step 6
        return Voucher(id_, pairs, rct)

    # ------------------------------------------------------------------ helpers
    def _pack(self, r, adct: bytes, x: int, y: int) -> bytes:
        out = b"".join(c.to_bytes(ELL_LEN, "big") for c in r) + adct + x.to_bytes(SH_LEN, "big") + y.to_bytes(SH_LEN, "big")
        assert len(out) == self.params.rct_plain_len
        return out
