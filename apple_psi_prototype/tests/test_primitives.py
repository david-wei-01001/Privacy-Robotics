"""Tests for the cryptographic building blocks (paper Section 3 and 4.3)."""
import os
import random
import secrets

import pytest
from cryptography.hazmat.primitives.asymmetric import ec as openssl_ec

from apple_psi.crypto_utils import H0, dec, enc, prf_F, random_key
from apple_psi.cuckoo import CuckooTable, cuckoo_positions
from apple_psi.dhf import DHF, detect
from apple_psi.ec import G, INFINITY, N, Point, multi_mul, random_point, random_scalar
from apple_psi.hash_to_curve import hash_item, hash_to_curve
from apple_psi.params import ELL, P_SH
from apple_psi.shamir import SharingPolynomial, reconstruct


# ------------------------------------------------------------------ elliptic curve
def test_group_order_and_generator():
    assert (N * G).is_infinity()
    assert (G + (-G)).is_infinity()
    assert G + G == 2 * G


def test_scalar_mult_matches_openssl():
    for _ in range(25):
        k = random_scalar()
        pub = openssl_ec.derive_private_key(k, openssl_ec.SECP256R1()).public_key().public_numbers()
        q = k * G
        assert (q.x, q.y) == (pub.x, pub.y)


def test_multi_mul_equals_two_muls_and_edge_cases():
    for _ in range(10):
        a, b, p1, p2 = random_scalar(), random_scalar(), random_point(), random_point()
        assert multi_mul(a, p1, b, p2) == a * p1 + b * p2
    assert multi_mul(5, G, N - 5, G).is_infinity()
    assert multi_mul(0, G, 7, G) == 7 * G


def test_point_encoding_roundtrip_and_validation():
    for _ in range(10):
        p = random_point()
        assert Point.from_bytes(p.to_bytes()) == p
    with pytest.raises(ValueError):
        Point.from_bytes(b"\x04" + bytes(32))               # bad prefix / length
    with pytest.raises(ValueError):
        Point.from_xy(1, 1)                                  # not on the curve
    with pytest.raises(ValueError):
        INFINITY.to_bytes()
    # about half of all x values are not the x-coordinate of any curve point; those must be rejected
    rejected = 0
    for x in range(1, 41):
        try:
            Point.from_bytes(b"\x02" + x.to_bytes(32, "big"))
        except ValueError:
            rejected += 1
    assert 5 <= rejected <= 35


# ------------------------------------------------------------------ hash to curve
RFC9380_DST = b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_RO_"


@pytest.mark.parametrize("msg,x,y", [
    (b"", 0x2c15230b26dbc6fc9a37051158c95b79656e17a1a920b11394ca91c44247d3e4,
          0x8a7a74985cc5c776cdfe4b1f19884970453912e9d31528c060be9ab5c43e8415),
    (b"abc", 0x0bb8b87485551aa43ed54f009230450b492fead5f1cc91658775dac4a3388a0f,
             0x5c41b3d0731a27a7b14bc0bf0ccded2d8751f83493404c84a88e71ffd424212e),
])
def test_hash_to_curve_rfc9380_vectors(msg, x, y):
    pt = hash_to_curve(msg, RFC9380_DST)
    assert (pt.x, pt.y) == (x, y)


def test_hash_item_deterministic_and_domain_separated():
    n1, n2 = os.urandom(16), os.urandom(16)
    assert hash_item(b"abc", n1) == hash_item(b"abc", n1)
    assert hash_item(b"abc", n1) != hash_item(b"abc", n2)
    assert hash_item(b"abc", n1) != hash_item(b"abd", n1)


# ------------------------------------------------------------------ the two fields
def _is_probable_prime(n, rounds=30):
    if n < 4:
        return n in (2, 3)
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def test_fields_are_prime_and_big_enough():
    assert _is_probable_prime(ELL) and ELL.bit_length() == 64
    assert _is_probable_prime(P_SH) and P_SH > 2**128


# ------------------------------------------------------------------ symmetric
def test_enc_dec_roundtrip_and_lengths():
    k = random_key()
    for n in (0, 1, 16, 100):
        pt = os.urandom(n)
        ct = enc(k, pt)
        assert len(ct) == n + 28 and dec(k, ct) == pt


def test_random_key_robustness():
    """Decrypting under an independent random key must fail (paper eq. (1))."""
    ct = enc(random_key(), b"message")
    assert all(dec(random_key(), ct) is None for _ in range(500))
    assert dec(random_key(), b"short") is None


def test_H0_and_prf_are_deterministic():
    p = random_point()
    assert H0(p) == H0(p) and len(H0(p)) == 16 and H0(p) != H0(random_point())
    fkey, id_ = random_key(), os.urandom(16)
    a, b = prf_F(fkey, id_, 5), prf_F(fkey, id_, 5)
    assert a == b
    x, z, x0, r = a
    assert 0 < x < P_SH and 0 <= z < P_SH and 0 <= x0 < ELL and len(r) == 6 and all(0 <= c < ELL for c in r)
    assert prf_F(fkey, os.urandom(16), 5) != a


# ------------------------------------------------------------------ Shamir
def test_shamir_threshold_behaviour():
    t = 7
    secret = secrets.randbelow(2**128)
    poly = SharingPolynomial(secret, t)
    shares = [poly.share(1 + secrets.randbelow(P_SH - 1)) for _ in range(t + 3)]
    assert reconstruct(shares[:t + 1]) == secret
    assert reconstruct(shares[2:t + 3]) == secret
    assert reconstruct(shares[:t]) != secret           # t shares reveal nothing
    with pytest.raises(ValueError):
        poly.share(0)


# ------------------------------------------------------------------ DHF + detection
def _mixture(t, s, n_real, n_rand, rng):
    h = DHF(s, t)
    cols = [h(secrets.randbelow(ELL)) for _ in range(n_real)]
    cols += [tuple(secrets.randbelow(ELL) for _ in range(s + 1)) for _ in range(n_rand)]
    order = list(range(len(cols)))
    rng.shuffle(order)
    return [cols[i] for i in order], sorted(j for j, i in enumerate(order) if i < n_real)


def test_dhf_detects_exactly_the_genuine_columns():
    rng = random.Random(7)
    for _ in range(40):
        t, s = rng.randint(2, 12), rng.randint(1, 12)
        cols, truth = _mixture(t, s, rng.randint(t + 1, t + 15), rng.randint(0, s), rng)
        assert detect(cols, s, t) == truth


def test_dhf_hides_when_only_t_genuine():
    """t genuine + up to s random columns: detection must fail (weak t-wise independence)."""
    rng = random.Random(8)
    for _ in range(30):
        t, s = rng.randint(2, 10), rng.randint(1, 10)
        cols, _ = _mixture(t, s, t, s, rng)
        assert detect(cols, s, t) is None


def test_dhf_output_shape():
    h = DHF(4, 3)
    out = h(12345)
    assert len(out) == 5 and out[0] == 12345


# ------------------------------------------------------------------ cuckoo
def test_cuckoo_positions_are_distinct_even_in_tiny_tables():
    nonces = [os.urandom(16) for _ in range(3)]
    for _ in range(200):
        pos = cuckoo_positions(os.urandom(12), nonces, 3)
        assert sorted(pos) == [0, 1, 2]


def test_cuckoo_places_everything_at_low_load():
    nonces = [os.urandom(16) for _ in range(2)]
    items = [os.urandom(12) for _ in range(200)]
    table = CuckooTable(int(2.5 * len(items)), nonces)
    assert [table.insert(i) for i in items] == [None] * len(items)
    for i in items:
        assert table.lookup(i) is not None
    assert sum(s is not None for s in table.slots) == len(items)


def test_cuckoo_drops_items_when_overloaded_but_stays_consistent():
    nonces = [os.urandom(16) for _ in range(2)]
    items = [os.urandom(12) for _ in range(300)]
    table = CuckooTable(int(1.05 * len(items)), nonces)
    dropped = [d for d in (table.insert(i) for i in items) if d is not None]
    assert dropped, "2 hash functions cannot fit n items into 1.05n cells"
    placed = {s for s in table.slots if s is not None}
    assert placed.isdisjoint(dropped) and len(placed) + len(dropped) == len(items)
    assert all(table.lookup(p) is not None for p in placed)


# ------------------------------------------------------------------ DH random self-reduction
def test_naor_reingold_self_reduction():
    """Section 3.1: DH tuple -> (L, Q, S) with S = alpha*Q ; non-DH tuple -> S != alpha*Q."""
    alpha = random_scalar()
    L = alpha * G
    T = random_point()
    for P, is_dh in ((alpha * T, True), (random_point(), False)):
        beta, gamma = random_scalar(), random_scalar()
        Q = multi_mul(beta, T, gamma, G)
        S = multi_mul(beta, P, gamma, L)
        assert (S == alpha * Q) is is_dh
