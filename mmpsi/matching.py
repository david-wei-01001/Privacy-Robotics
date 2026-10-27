"""Matching module (PLAN §6): post-processing, plaintext matchers, LSH codes, Apple-PSI equivalence check."""
from __future__ import annotations

import time

import numpy as np

from .core import register


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


class Prepared:
    """Centered (on the benign-train mean) and L2-normalized embeddings; PCA fitted on reference + benign-train."""

    def __init__(self, E: np.ndarray, ref: np.ndarray, benign_train: np.ndarray, pca_dim: int = 128):
        self.raw = E.astype(np.float32)
        mu = self.raw[benign_train].mean(0) if benign_train.any() else self.raw.mean(0)
        self.centered = self.raw - mu
        self.unit = l2n(self.centered)
        fit = self.unit[ref | benign_train]
        k = min(pca_dim, fit.shape[1], max(1, fit.shape[0] - 1))
        self.pca_mean = fit.mean(0)
        _, _, vt = np.linalg.svd(fit - self.pca_mean, full_matrices=False)
        self.pca = (self.unit - self.pca_mean) @ vt[:k].T


def _topk_cos(Q: np.ndarray, R: np.ndarray, k: int, chunk: int = 4096) -> np.ndarray:
    out = np.empty((len(Q), k), np.float32)
    for i in range(0, len(Q), chunk):
        s = Q[i:i + chunk] @ R.T
        part = np.partition(s, -k, axis=1)[:, -k:]
        out[i:i + chunk] = np.sort(part, axis=1)[:, ::-1]
    return out


class Matcher:
    discrete = False

    def fit(self, P: Prepared, ref: np.ndarray, benign_train: np.ndarray):
        return self

    def score(self, P: Prepared, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@register("matcher", "cosine_top1")
class CosineTop1(Matcher):
    def fit(self, P, ref, benign_train):
        self.R = P.unit[ref]
        return self

    def score(self, P, idx):
        return _topk_cos(P.unit[idx], self.R, 1)[:, 0]


@register("matcher", "cosine_top5")
class CosineTop5(CosineTop1):
    def score(self, P, idx):
        return _topk_cos(P.unit[idx], self.R, min(5, len(self.R))).mean(1)


@register("matcher", "l2_raw")
class L2Raw(Matcher):
    """Nearest-reference L2 distance on centered but not normalized embeddings (negated: higher = closer)."""

    def fit(self, P, ref, benign_train):
        self.R = P.centered[ref]
        self.R2 = (self.R ** 2).sum(1)
        return self

    def score(self, P, idx):
        Q = P.centered[idx]
        out = np.empty(len(Q), np.float32)
        for i in range(0, len(Q), 4096):
            q = Q[i:i + 4096]
            d2 = (q ** 2).sum(1)[:, None] + self.R2[None, :] - 2 * q @ self.R.T
            out[i:i + 4096] = -np.sqrt(np.maximum(d2.min(1), 0))
        return out


@register("matcher", "hamming_sign")
class HammingSign(Matcher):
    """Sign bits of the PCA projection; score = negated minimum Hamming distance to any reference."""

    def fit(self, P, ref, benign_train):
        self.R = (P.pca[ref] > 0).astype(np.float32)
        return self

    def score(self, P, idx):
        Q = (P.pca[idx] > 0).astype(np.float32)
        nb = Q.shape[1]
        out = np.empty(len(Q), np.float32)
        for i in range(0, len(Q), 4096):
            q = Q[i:i + 4096]
            agree = q @ self.R.T + (1 - q) @ (1 - self.R).T
            out[i:i + 4096] = -(nb - agree.max(1))
        return out


def simhash_codes(X: np.ndarray, k: int, L: int, seed: int = 0) -> np.ndarray:
    """Hyperplane LSH: (n, L) integer codes, k bits per band. The seed is public (PLAN §6.2)."""
    planes = np.random.default_rng(seed).standard_normal((L * k, X.shape[1])).astype(np.float32)
    bits = (X @ planes.T > 0).reshape(len(X), L, k)
    return (bits * (1 << np.arange(k))).sum(2).astype(np.int64)


@register("matcher", "lsh")
class LSHExact(Matcher):
    """Score = number of bands whose code appears in the reference table of that band (decision: >= m)."""
    discrete = True

    def __init__(self, k: int = 12, L: int = 16, seed: int = 0):
        self.k, self.L, self.seed = k, L, seed

    def fit(self, P, ref, benign_train):
        codes = simhash_codes(P.unit[ref], self.k, self.L, self.seed)
        self.tables = [set(codes[:, b].tolist()) for b in range(self.L)]
        self.coverage = float(np.mean([len(t) / (1 << self.k) for t in self.tables]))
        return self

    def score(self, P, idx):
        codes = simhash_codes(P.unit[idx], self.k, self.L, self.seed)
        return np.stack([np.isin(codes[:, b], list(self.tables[b])) for b in range(self.L)], 1).sum(1).astype(np.float32)


@register("matcher", "logreg")
class LogReg(Matcher):
    """Local supervised classifier on the same embeddings: a reference point, not a private matcher (PLAN §7.2)."""

    def fit(self, P, ref, benign_train):
        from sklearn.linear_model import LogisticRegression
        X = np.concatenate([P.unit[ref], P.unit[benign_train]])
        y = np.r_[np.ones(ref.sum()), np.zeros(benign_train.sum())]
        self.clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X, y)
        return self

    def score(self, P, idx):
        return self.clf.decision_function(P.unit[idx]).astype(np.float32)


# ----------------------------------------------------------------------------- Apple PSI equivalence
def band_items(codes: np.ndarray, k: int) -> list[list[bytes]]:
    nbytes = (k + 7) // 8
    return [[bytes([b]) + int(c).to_bytes(nbytes, "big") for b, c in enumerate(row)] for row in codes]


def apple_psi_equivalence(P: Prepared, ref: np.ndarray, q: np.ndarray, k: int, L: int, m: int,
                          seed: int = 0, log=None) -> dict:
    """Run the prototype on band items: does the server's decision (>= m bands open) equal plaintext LSH?

    `q` holds the query windows (the caller mixes violent and benign windows so that both decisions occur). A
    disagreement is expected only where a matching band item was dropped from the server's cuckoo table."""
    from apple_psi import Client, Params, Server
    ref_codes = simhash_codes(P.unit[ref], k, L, seed)
    X = sorted({it for row in band_items(ref_codes, k) for it in row})
    params = Params(t=5, s_max=10, ad_len=64)
    server = Server(params)
    t0 = time.time()
    pdata = server.setup(X)
    setup_s = time.time() - t0
    in_table = set(X) - set(server.dropped)
    client = Client(params, pdata.to_bytes(), synthetic_prob=0.0)
    q_codes = simhash_codes(P.unit[q], k, L, seed)
    plain = LSHExact(k, L, seed).fit(P, ref, ref).score(P, q) >= m
    agree, explained, n_vouchers, c_time, s_time, vbytes = 0, 0, 0, 0.0, 0.0, 0
    for i, row in enumerate(band_items(q_codes, k)):
        opened = 0
        for it in row:
            t = time.time()
            v = client.create_voucher(it, bytes(64)).to_bytes()
            c_time += time.time() - t
            vbytes = len(v)
            t = time.time()
            opened += server.ingest_bytes(v) == "match"
            s_time += time.time() - t
            n_vouchers += 1
        if (opened >= m) == bool(plain[i]):
            agree += 1
        elif (opened >= m) == (sum(it in in_table for it in row) >= m):
            explained += 1
        if log and i % 50 == 49:
            log.info(f"  apple_psi: {i + 1}/{len(q)} windows")
    return dict(k=k, L=L, m=m, n_windows=int(len(q)), n_plain_positive=int(plain.sum()), n_items_X=len(X),
                cuckoo_dropped=len(server.dropped), equivalence_rate=agree / len(q), n_disagree=int(len(q) - agree),
                n_disagree_explained_by_drops=explained, voucher_bytes=vbytes, setup_s=round(setup_s, 2),
                client_ms_per_voucher=round(1000 * c_time / max(1, n_vouchers), 2),
                server_ms_per_voucher=round(1000 * s_time / max(1, n_vouchers), 2),
                pdata_bytes=len(pdata.to_bytes()))
