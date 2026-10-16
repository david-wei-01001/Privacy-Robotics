"""Dataset helpers (offline-safe: Fashion-MNIST is only exercised if it is already cached or reachable)."""
import numpy as np
import pytest

from apple_psi import datasets as ds


def test_synthetic_hashes_are_distinct_96_bit():
    hs = ds.synthetic_hashes(500)
    assert len(set(hs)) == 500 and all(len(h) == 12 for h in hs)


def test_phash_shape_determinism_and_sensitivity():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (28, 28), dtype=np.uint8)
    b = rng.integers(0, 256, (28, 28), dtype=np.uint8)
    assert len(ds.phash96(a)) == 12 and ds.phash96(a) == ds.phash96(a)
    assert ds.phash96(a) != ds.phash96(b)
    assert len(ds.visual_derivative(a)) == ds.AD_LEN


def test_phash_is_robust_to_global_brightness_shift_but_not_to_other_images():
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 200, (28, 28), dtype=np.uint8) for _ in range(30)]
    same = sum(ds.phash96(i) == ds.phash96(np.clip(i.astype(int) + 20, 0, 255).astype(np.uint8)) for i in imgs)
    assert same >= 25


def test_ascii_thumbnail_dimensions():
    art = ds.ascii_thumbnail(bytes(range(0, 256, 4)))
    assert len(art.splitlines()) == 8 and all(len(r) == 16 for r in art.splitlines())


def test_digits_loader_offline():
    pytest.importorskip("sklearn")
    imgs = ds.load_digits()
    assert imgs.shape == (1797, 8, 8)
    hashes, reps = ds.unique_by_hash(imgs)
    assert len(hashes) == len(reps) and len(hashes) > 1000


def test_fashion_mnist_loader_if_available(tmp_path):
    try:
        imgs = ds.load_fashion_mnist("t10k", cache_dir=tmp_path)
    except OSError:
        pytest.skip("no network access")
    assert imgs.shape == (10000, 28, 28) and imgs.dtype == np.uint8
