"""Protocol parameters and fixed encodings.

All sizes that determine the *wire format* live here so that every voucher has
exactly the same length (a hard requirement of the paper, Section 2.3).

Paper notation  ->  name in this code
-------------------------------------
t  (threshold)           Params.t          server learns ad iff #real matches > t
s  (max synthetics)      Params.s_max      upper bound on |S|
eps' (cuckoo slack)      Params.table_expansion  (table size n' = expansion * n)
ell (64-bit prime)       ELL               field of the detectable hash function
F_Sh (Shamir field)      P_SH              field of the Shamir secret sharing
"""
from __future__ import annotations

from dataclasses import dataclass

# --- fields -----------------------------------------------------------------
ELL = 2**64 - 59      # largest prime below 2**64 (Section 4.3: "ell is a 64-bit prime")
P_SH = 2**130 - 5     # prime (the Poly1305 prime); > 2**128 so a 128-bit key is a field element

# --- symmetric sizes (AES-128-GCM with random 96-bit nonce, Section 3) -------
KEY_LEN = 16
NONCE_LEN = 12
TAG_LEN = 16

# --- encodings ---------------------------------------------------------------
POINT_LEN = 33        # SEC1 compressed P-256 point
SH_LEN = 17           # bytes for one element of F_Sh (130 bits)
ELL_LEN = 8           # bytes for one element of F_ell
ID_LEN = 16           # random 128-bit identifiers
NONCE_H_LEN = 16      # 128-bit domain-separation nonces for H, h_1..h_d

CT_RKEY_LEN = NONCE_LEN + KEY_LEN + TAG_LEN  # length of ct_j = Enc(H0(S_j), rkey)


@dataclass(frozen=True)
class Params:
    """Public protocol parameters, known to both client and server."""

    t: int = 5                      # threshold (Apple's published deployment value was 30)
    s_max: int = 10                 # upper bound on the number of synthetic vouchers
    ad_len: int = 64                # all associated data has the same public length
    num_hashes: int = 2             # cuckoo hash functions; the paper uses 2 (3 mentioned as option)
    table_expansion: float = 2.5    # n' = table_expansion * n  (2 hashes need > 2n to place ~all items)
    max_kicks: int = 500            # cuckoo insertion budget before an item is dropped

    def __post_init__(self) -> None:
        if self.t < 1:
            raise ValueError("t must be >= 1")
        if self.s_max < 0:
            raise ValueError("s_max must be >= 0")
        if self.num_hashes < 2:
            raise ValueError("need at least 2 cuckoo hash functions")
        if self.ad_len < 1:
            raise ValueError("ad_len must be >= 1")

    # --- derived wire-format lengths -----------------------------------------
    @property
    def r_len(self) -> int:
        """|R| in bytes: an element of R = F_ell^(s+1)."""
        return (self.s_max + 1) * ELL_LEN

    @property
    def adct_len(self) -> int:
        return NONCE_LEN + self.ad_len + TAG_LEN

    @property
    def rct_plain_len(self) -> int:
        return self.r_len + self.adct_len + 2 * SH_LEN

    @property
    def rct_len(self) -> int:
        return NONCE_LEN + self.rct_plain_len + TAG_LEN

    @property
    def voucher_len(self) -> int:
        return ID_LEN + self.num_hashes * (POINT_LEN + CT_RKEY_LEN) + self.rct_len
