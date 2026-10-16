"""Minimal pure-Python NIST P-256 (secp256r1) group arithmetic.

The Apple protocol needs *point* arithmetic (beta*H(y) + gamma*G, alpha*Q, ...),
which the high-level ECDH/ECDSA APIs of common libraries do not expose, so the
group law is implemented here directly.

!! Educational prototype: NOT constant time, NOT side-channel hardened. !!
"""
from __future__ import annotations

import secrets
from typing import Optional

# --- curve parameters (FIPS 186-4, P-256) -------------------------------------
P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
A = P - 3
B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551  # group order (prime)
GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5

# --- Jacobian coordinates: (X, Y, Z) <-> affine (X/Z^2, Y/Z^3); Z == 0 is infinity
_INF = (1, 1, 0)


def _jdbl(pt):
    X, Y, Z = pt
    if Z == 0 or Y == 0:
        return _INF
    # dbl-2001-b (a = -3)
    delta = Z * Z % P
    gamma = Y * Y % P
    beta = X * gamma % P
    alpha = 3 * (X - delta) * (X + delta) % P
    X3 = (alpha * alpha - 8 * beta) % P
    Z3 = ((Y + Z) * (Y + Z) - gamma - delta) % P
    Y3 = (alpha * (4 * beta - X3) - 8 * gamma * gamma) % P
    return (X3, Y3, Z3)


def _jadd(p1, p2):
    X1, Y1, Z1 = p1
    X2, Y2, Z2 = p2
    if Z1 == 0:
        return p2
    if Z2 == 0:
        return p1
    # add-2007-bl
    Z1Z1 = Z1 * Z1 % P
    Z2Z2 = Z2 * Z2 % P
    U1 = X1 * Z2Z2 % P
    U2 = X2 * Z1Z1 % P
    S1 = Y1 * Z2 % P * Z2Z2 % P
    S2 = Y2 * Z1 % P * Z1Z1 % P
    if U1 == U2:
        if S1 != S2:
            return _INF
        return _jdbl(p1)
    H = (U2 - U1) % P
    I = (2 * H) * (2 * H) % P
    J = H * I % P
    r = 2 * (S2 - S1) % P
    V = U1 * I % P
    X3 = (r * r - J - 2 * V) % P
    Y3 = (r * (V - X3) - 2 * S1 * J) % P
    Z3 = (((Z1 + Z2) * (Z1 + Z2) - Z1Z1 - Z2Z2) * H) % P
    return (X3, Y3, Z3)


def _jmul(k: int, pj) -> tuple:
    """k * pj with a fixed 4-bit window."""
    k %= N
    if k == 0 or pj[2] == 0:
        return _INF
    table = [_INF, pj]
    for _ in range(2, 16):
        table.append(_jadd(table[-1], pj))
    acc = _INF
    top = ((k.bit_length() + 3) // 4 - 1) * 4
    for shift in range(top, -1, -4):
        acc = _jdbl(_jdbl(_jdbl(_jdbl(acc))))
        d = (k >> shift) & 0xF
        if d:
            acc = _jadd(acc, table[d])
    return acc


def _jmul2(k1: int, p1, k2: int, p2) -> tuple:
    """k1*p1 + k2*p2 with Straus/Shamir's simultaneous double-and-add."""
    k1 %= N
    k2 %= N
    p12 = _jadd(p1, p2)
    acc = _INF
    for i in range(max(k1.bit_length(), k2.bit_length()) - 1, -1, -1):
        acc = _jdbl(acc)
        b1 = (k1 >> i) & 1
        b2 = (k2 >> i) & 1
        if b1 and b2:
            acc = _jadd(acc, p12)
        elif b1:
            acc = _jadd(acc, p1)
        elif b2:
            acc = _jadd(acc, p2)
    return acc


class Point:
    """An affine point of P-256 (or the point at infinity)."""

    __slots__ = ("x", "y")

    def __init__(self, x: Optional[int], y: Optional[int]):
        self.x = x
        self.y = y

    # --- constructors / predicates ------------------------------------------
    @classmethod
    def infinity(cls) -> "Point":
        return cls(None, None)

    def is_infinity(self) -> bool:
        return self.x is None

    @classmethod
    def from_xy(cls, x: int, y: int) -> "Point":
        """Validate that (x, y) is on the curve and return the point."""
        if not (0 <= x < P and 0 <= y < P):
            raise ValueError("coordinate out of range")
        if (y * y - (x * x * x + A * x + B)) % P != 0:
            raise ValueError("point is not on the curve")
        return cls(x, y)

    # --- Jacobian bridge ------------------------------------------------------
    def _jac(self):
        return _INF if self.x is None else (self.x, self.y, 1)

    @classmethod
    def _from_jac(cls, pt) -> "Point":
        X, Y, Z = pt
        if Z == 0:
            return cls(None, None)
        zi = pow(Z, -1, P)
        zi2 = zi * zi % P
        return cls(X * zi2 % P, Y * zi2 % P * zi % P)

    # --- group law -------------------------------------------------------------
    def __add__(self, other: "Point") -> "Point":
        return Point._from_jac(_jadd(self._jac(), other._jac()))

    def __neg__(self) -> "Point":
        return self if self.x is None else Point(self.x, (-self.y) % P)

    def __sub__(self, other: "Point") -> "Point":
        return self + (-other)

    def __mul__(self, k: int) -> "Point":
        return Point._from_jac(_jmul(k, self._jac()))

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return "Point(inf)" if self.x is None else f"Point({self.x:064x}, {self.y:064x})"

    # --- SEC1 compressed encoding ---------------------------------------------
    def to_bytes(self) -> bytes:
        if self.x is None:
            raise ValueError("cannot encode the point at infinity")
        return bytes([2 + (self.y & 1)]) + self.x.to_bytes(32, "big")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Point":
        if len(data) != 33 or data[0] not in (2, 3):
            raise ValueError("bad compressed point encoding")
        x = int.from_bytes(data[1:], "big")
        if x >= P:
            raise ValueError("x out of range")
        rhs = (x * x * x + A * x + B) % P
        y = pow(rhs, (P + 1) // 4, P)  # P = 3 mod 4
        if y * y % P != rhs:
            raise ValueError("x is not on the curve")
        if (y & 1) != (data[0] & 1):
            y = P - y
        return cls(x, y)


G = Point(GX, GY)
INFINITY = Point.infinity()


def multi_mul(k1: int, p1: Point, k2: int, p2: Point) -> Point:
    """k1*p1 + k2*p2 (about 1.5x cheaper than two separate multiplications)."""
    return Point._from_jac(_jmul2(k1, p1._jac(), k2, p2._jac()))


def random_scalar() -> int:
    """Uniform scalar in [1, N-1]."""
    return 1 + secrets.randbelow(N - 1)


def random_point() -> Point:
    """A uniformly random non-identity point of the (prime-order) group."""
    return random_scalar() * G
