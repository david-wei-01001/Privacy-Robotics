"""Shamir secret sharing over F_Sh = GF(2^130 - 5).

The client shares its 128-bit `adkey` with a degree-t polynomial, so t+1 shares are
needed to reconstruct it.  Every voucher carries one share (x, f(x)) whose
x-coordinate is derived pseudorandomly from the triple's id (see crypto_utils.prf_F),
so a *duplicate* triple always yields the *same* share and counts once.
"""
from __future__ import annotations

import secrets
from typing import Sequence

from .params import P_SH


class SharingPolynomial:
    """f(X) = secret + c_1 X + ... + c_t X^t  over F_Sh."""

    def __init__(self, secret: int, t: int, coeffs: Sequence[int] | None = None):
        if not 0 <= secret < P_SH:
            raise ValueError("secret must be a field element")
        if coeffs is None:
            coeffs = [secret] + [secrets.randbelow(P_SH) for _ in range(t)]
        if len(coeffs) != t + 1 or coeffs[0] != secret:
            raise ValueError("inconsistent polynomial")
        self.coeffs = list(coeffs)

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def share(self, x: int) -> tuple[int, int]:
        if x % P_SH == 0:
            raise ValueError("x = 0 would reveal the secret")
        acc = 0
        for c in reversed(self.coeffs):          # Horner
            acc = (acc * x + c) % P_SH
        return x, acc


def reconstruct(shares: Sequence[tuple[int, int]]) -> int:
    """Lagrange interpolation at 0.  Needs t+1 shares with distinct x for a degree-t polynomial."""
    xs = [x for x, _ in shares]
    if len(set(xs)) != len(xs):
        raise ValueError("shares must have distinct x-coordinates")
    secret = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i != j:
                num = num * xj % P_SH
                den = den * (xj - xi) % P_SH
        secret = (secret + yi * num * pow(den, -1, P_SH)) % P_SH
    return secret
