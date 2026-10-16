"""(s, t)-detectable hash function (DHF) -- paper Section 4.3.

Key      : s random polynomials p_1..p_s over F_ell, each of degree <= t-1.
Eval     : DHF(k, x0) = (x0, p_1(x0), ..., p_s(x0))  in F_ell^(s+1).
Property : any t outputs on random x0 look uniform (weak t-wise independence), yet when
           >= t+1 of the columns handed to the server are DHF outputs and <= s are
           uniformly random, `detect` finds exactly the DHF columns.

Why detection works
-------------------
Expand every column (x0, r_1..r_s) to  (1, x0, ..., x0^(t-1), r_1..r_s)  in F_ell^(s+t).
All genuine DHF columns then lie in one t-dimensional subspace V (the last s coordinates
are fixed linear functions of the first t).  Any t of them are independent (Vandermonde),
so t+1 genuine columns are linearly dependent, whereas <= s random columns are -- with
overwhelming probability -- independent of V and of each other.  Hence every linear
dependency of the matrix is supported on genuine columns only, and the genuine columns
are exactly those inside the span of one dependency's support.
"""
from __future__ import annotations

import secrets
from typing import Optional, Sequence

from .params import ELL


class DHF:
    def __init__(self, s: int, t: int, key: Sequence[Sequence[int]] | None = None):
        self.s, self.t = s, t
        if key is None:
            key = [[secrets.randbelow(ELL) for _ in range(t)] for _ in range(s)]
        if len(key) != s or any(len(row) != t for row in key):
            raise ValueError("DHF key must be an s x t matrix")
        self.key = [list(row) for row in key]

    def __call__(self, x0: int) -> tuple[int, ...]:
        out = [x0 % ELL]
        for coeffs in self.key:               # coeffs[j] multiplies x0^j
            acc = 0
            for c in reversed(coeffs):        # Horner
                acc = (acc * x0 + c) % ELL
            out.append(acc)
        return tuple(out)


# ----------------------------------------------------------------------------
# linear algebra over F_ell
# ----------------------------------------------------------------------------
class _Echelon:
    """Incremental echelon basis of a set of column vectors (for span-membership tests)."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, list[int]]] = []     # (pivot index, vector with vec[pivot] == 1)

    def reduce(self, v: Sequence[int]) -> list[int]:
        v = list(v)
        for piv, b in self.rows:
            c = v[piv]
            if c:
                for i in range(len(v)):
                    v[i] = (v[i] - c * b[i]) % ELL
        return v

    def add(self, v: Sequence[int]) -> bool:
        v = self.reduce(v)
        piv = next((i for i, a in enumerate(v) if a), None)
        if piv is None:
            return False
        inv = pow(v[piv], -1, ELL)
        self.rows.append((piv, [a * inv % ELL for a in v]))
        return True


def _first_dependency(cols: Sequence[Sequence[int]]) -> Optional[dict[int, int]]:
    """Scan columns left to right; return a kernel vector (sparse dict) at the first column that
    is a linear combination of earlier ones, or None if all columns are independent."""
    basis: list[tuple[int, list[int], dict[int, int]]] = []     # (pivot, vector, combo)
    for idx, col in enumerate(cols):
        v = list(col)
        combo = {idx: 1}
        for piv, b, bc in basis:
            c = v[piv]
            if c:
                for i in range(len(v)):
                    v[i] = (v[i] - c * b[i]) % ELL
                for k, ck in bc.items():
                    combo[k] = (combo.get(k, 0) - c * ck) % ELL
        piv = next((i for i, a in enumerate(v) if a), None)
        if piv is None:                                   # sum_k combo[k] * col_k == 0
            return {k: c for k, c in combo.items() if c}
        inv = pow(v[piv], -1, ELL)
        basis.append((piv, [a * inv % ELL for a in v], {k: c * inv % ELL for k, c in combo.items()}))
    return None


def _expand(col: Sequence[int], t: int) -> list[int]:
    x0 = col[0] % ELL
    powers, p = [], 1
    for _ in range(t):
        powers.append(p)
        p = p * x0 % ELL
    return powers + [c % ELL for c in col[1:]]


def detect(columns: Sequence[Sequence[int]], s: int, t: int) -> Optional[list[int]]:
    """Return the sorted indices of the columns that are DHF outputs, or None ("fail").

    Guarantee (paper): if >= max(t+1, m-s) of the m columns are genuine DHF outputs, the exact set
    is returned except with probability ~ m/ell.  If only <= t are genuine (and <= s random), the
    matrix has full column rank, no dependency exists and the result is None -- which is what
    hides the real matches below the threshold.
    """
    m = len(columns)
    if m < t + 1:
        return None
    if any(len(c) != s + 1 for c in columns):
        raise ValueError("each column must have s+1 entries")
    cols = [_expand(c, t) for c in columns]
    w = _first_dependency(cols)
    if w is None:
        return None
    ech = _Echelon()
    for k in w:
        ech.add(cols[k])
    return [i for i, c in enumerate(cols) if not any(ech.reduce(c))]
