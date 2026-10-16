"""Stand-in data for the prototype.

The paper ships no data: the protocol only ever sees opaque byte strings (the hashes y)
and fixed-length associated data (ad).  Apple's real inputs -- NeuralHash values of
photos and a proprietary server-side hash list -- are not public (and must not be
reconstructed).  This module provides benign substitutes with the *same shape*:

    synthetic_hashes   uniformly random 96-bit strings (96 bits = NeuralHash output size)
    load_fashion_mnist Fashion-MNIST (Zalando, MIT licence) fetched from GitHub; 28x28 images
    load_digits        scikit-learn's bundled 8x8 digits (works fully offline)
    load_image_folder  any directory of images you point it at
    phash96            96-bit DCT perceptual hash (image -> 12 bytes)
    visual_derivative  8x8 grayscale thumbnail (image -> 64 bytes) used as associated data

phash96 is a classic, non-neural perceptual hash (DCT low-frequency coefficients vs. their
median).  It is *not* NeuralHash, but it exhibits the property the system relies on: a mildly
modified copy of an image often (not always!) maps to the identical hash, so an exact-match
PSI on hashes can detect near-duplicates.  Whatever the hash misses, the PSI misses too.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

HASH_LEN = 12   # bytes (96 bits)
AD_LEN = 64     # bytes (8x8 thumbnail)

_FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{name}"
_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def default_cache_dir() -> Path:
    return Path(os.environ.get("APPLE_PSI_CACHE", Path.home() / ".cache" / "apple_psi"))


# --------------------------------------------------------------------------- hashes
def synthetic_hashes(n: int, nbytes: int = HASH_LEN) -> List[bytes]:
    """n distinct uniformly random hash values."""
    out: set[bytes] = set()
    while len(out) < n:
        out.add(os.urandom(nbytes))
    return list(out)


def _as_pil(img) -> Image.Image:
    return img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img, dtype=np.uint8))


_N = 32
_k = np.arange(_N)[:, None]
_n = np.arange(_N)[None, :]
_DCT = np.cos(np.pi * (2 * _n + 1) * _k / (2 * _N))
_DCT[0] *= 1 / np.sqrt(2)
_DCT *= np.sqrt(2 / _N)           # orthonormal DCT-II matrix


def phash96(img) -> bytes:
    """96-bit perceptual hash: grayscale -> 32x32 -> 2-D DCT -> keep 96 low-frequency
    coefficients of the top-left 10x10 block (DC dropped) -> bit = coefficient > median.
    Robust to global brightness/contrast changes and rescaling; output is 12 bytes."""
    a = np.asarray(_as_pil(img).convert("L").resize((_N, _N), Image.BICUBIC), dtype=np.float64)
    block = (_DCT @ a @ _DCT.T)[:10, :10].flatten()[1:97]
    return np.packbits(block > np.median(block)).tobytes()


def visual_derivative(img) -> bytes:
    """8x8 grayscale thumbnail (64 bytes): the 'low-resolution visual derivative' stand-in."""
    return np.asarray(_as_pil(img).convert("L").resize((8, 8), _RESAMPLE), dtype=np.uint8).tobytes()


def ascii_thumbnail(ad: bytes, width: int = 8) -> str:
    """Render a visual_derivative() blob as ASCII art (two characters per pixel for aspect ratio)."""
    ramp = " .:-=+*#%@"
    rows = []
    for i in range(0, len(ad), width):
        rows.append("".join(ramp[min(b * len(ramp) // 256, len(ramp) - 1)] * 2 for b in ad[i:i + width]))
    return "\n".join(rows)


def perturb(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """A mild 'edit' of an image (brightness/contrast change + faint noise), standing in for
    re-saving / re-compressing a photo.  Perceptual hashes are designed to survive this."""
    a = np.asarray(img, dtype=np.float32)
    a = a * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8) * strength
    a = a + rng.normal(0, 1.0 * strength, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- image sources
def load_fashion_mnist(split: str = "t10k", cache_dir: Path | None = None) -> np.ndarray:
    """Return an (N, 28, 28) uint8 array.  split: 't10k' (10,000 images) or 'train' (60,000)."""
    if split not in ("t10k", "train"):
        raise ValueError("split must be 't10k' or 'train'")
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    name = f"{split}-images-idx3-ubyte.gz"
    path = cache / name
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(_FASHION_URL.format(name=name), tmp)
        tmp.rename(path)
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError("not an IDX image file")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)


def load_digits() -> np.ndarray:
    """scikit-learn's 1,797 8x8 digit images, scaled to 0..255 (no download needed)."""
    from sklearn.datasets import load_digits as _ld
    return (_ld().images * (255.0 / 16.0)).astype(np.uint8)


def load_image_folder(folder: str | Path, limit: int | None = None) -> np.ndarray:
    """Load every readable image under `folder` as a 32x32 grayscale array (for any custom data)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
    imgs = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in exts:
            try:
                imgs.append(np.asarray(Image.open(p).convert("L").resize((32, 32), _RESAMPLE), dtype=np.uint8))
            except OSError:
                continue
            if limit and len(imgs) >= limit:
                break
    if not imgs:
        raise ValueError(f"no images found under {folder}")
    return np.stack(imgs)


def unique_by_hash(images: Sequence[np.ndarray]) -> tuple[list[bytes], list[np.ndarray]]:
    """Hash every image and keep the first image for each distinct hash."""
    seen: dict[bytes, np.ndarray] = {}
    for im in images:
        seen.setdefault(phash96(im), im)
    return list(seen.keys()), list(seen.values())
