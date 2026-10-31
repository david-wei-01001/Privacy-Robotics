"""A server-view adversary: what can the PSI server infer about household activity from benign uploads (PLAN §11.2)?

THREAT. Honest-but-curious server. It sees, for every benign window, the server view of mmpsi.replay: which of the L
band-vouchers opened and, for the opened ones, the k-bit band code; and the time order of the stream. It holds the
reference list, so it can interpret the codes. It does not see raw embeddings, audio, or the robot's decisions.

TARGETS. The SINS activity label of the window (9 classes) and occupancy (absence vs anything else). SINS labels are real.
They come per 10 s clip, and the clips of one session belong to one activity.

WHAT THE SERVER SEES (checked against apple_psi/server.py and client.py). For each voucher it learns match or
non-match, nothing else: the band and the k-bit code are inside the hashed item and the blinded point; the share
coordinate x is a PRF of a random voucher id under the client's secret key; the pairs inside a voucher are shuffled.
So, with the vouchers of a window sent together in shuffled order, the server's per-window view is the COUNT of
vouchers that opened. This is the leak of grouping the vouchers by window, which the m-of-L rule needs.

FEATURES
  alarm        one bit per window: >= m bands open (the decision; carries almost nothing for benign windows)
  count_hist   the protocol as implemented: the histogram of per-window counts over the pooled windows
  count        the mean count only (a weaker summary of the same view)
 WHAT-IFS, not what this protocol reveals; they bound weaker designs:
  open         which bands opened (L bits): a stream sent in band order, or a design that labels vouchers by band
  open+codes   which bands opened and their k code bits: a design whose server learns the matched table entry
and, as the NON-PRIVATE upper bound, the raw embedding (PCA 64): what a server that received embeddings would learn.

CONTEXT. The server sees timing, so it can pool features over a window, a 10 s clip, C consecutive clips of one device,
or a whole session (a minute or so). Longer context is a stronger adversary.

PROTOCOL. Stratified group k-fold, groups = sessions, so no session is in both train and test (the four recording nodes
share a session id, so they stay together). The session id is never a feature (SINS numbers sessions in label order).
Reported: balanced accuracy over the classes (chance = 1 / n_classes for any constant predictor) and the AUC of
absence vs present from out-of-fold probabilities. CONTROL: the same attack after permuting labels across sessions must
give chance; if it does not, there is a leak in the pipeline.

A PROXY, NOT A BOUND: accuracy of one classifier family on one dataset lower-bounds what a stronger adversary extracts.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_CLIP = re.compile(r"^DevNode(\d+)_ex(\d+)_(\d+)$")


COUNT_EDGES = (1, 2, 3, 4, 6, 9, 13, 21)  # bins of the per-window count: 0, 1, 2, 3, 4-5, 6-8, 9-12, 13-20, 21+


def view_features(view, mode: str, m: int | None = None) -> np.ndarray:
    O = view.opened
    if mode == "count_hist":  # averaged over a chunk this is the distribution of counts the server saw
        return np.eye(len(COUNT_EDGES) + 1, dtype=np.float32)[np.digitize(O.sum(1), COUNT_EDGES)]
    if mode == "alarm":
        return (O.sum(1, keepdims=True) >= m).astype(np.float32)
    if mode == "count":
        return O.sum(1, keepdims=True).astype(np.float32)
    if mode == "open":
        return O.astype(np.float32)
    if mode == "open+codes":
        bits = ((view.codes[..., None] >> np.arange(view.k)) & 1).astype(np.float32) * O[..., None]
        return np.concatenate([O.astype(np.float32), bits.reshape(len(O), -1)], 1)
    raise ValueError(mode)


def chunk_ids(clip_id: np.ndarray, context) -> np.ndarray:
    """Which windows the server can pool: 'window', 'clip', an int C (C consecutive clips of one device) or 'session'."""
    clip = np.asarray(clip_id).astype(str)
    if context == "window":
        return np.arange(len(clip)).astype(str)
    if context == "clip":
        return clip
    parsed = [_CLIP.match(c) for c in clip]
    if not all(parsed):
        raise ValueError("clip ids are not DevNode<n>_ex<session>_<i>: only the 'window' and 'clip' contexts exist")
    unit = np.array([int(p.group(1)) * 1_000_000 + int(p.group(2)) for p in parsed])
    if context == "session":
        return unit.astype(str)
    idx = np.array([int(p.group(3)) for p in parsed])
    rank = pd.DataFrame({"u": unit, "i": idx}).groupby("u")["i"].rank(method="dense").astype(int).values - 1
    return np.array([f"{u}_{r // int(context)}" for u, r in zip(unit, rank)])


def aggregate(F: np.ndarray, y: np.ndarray, groups: np.ndarray, chunks: np.ndarray):
    codes, _ = pd.factorize(chunks)
    Fa = pd.DataFrame(F).groupby(codes).mean().values.astype(np.float32)
    first = pd.Series(np.arange(len(codes))).groupby(codes).first().values
    return Fa, np.asarray(y)[first], np.asarray(groups)[first]


def attack(F: np.ndarray, y, groups, n_splits: int = 5, seed: int = 0, permute: bool = False, pca: int | None = None,
           n_jobs: int = 1) -> dict:
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y, groups = np.asarray(y).astype(str), np.asarray(groups).astype(str)
    if permute:  # permute the labels across SESSIONS, which keeps every session's internal structure
        ug, first = np.unique(groups, return_index=True)
        mapping = dict(zip(ug, np.random.default_rng(seed + 1).permutation(y[first])))
        y = np.array([mapping[g] for g in groups])
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))

    def fold(tr, te):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)  # an unconverged fit only weakens the adversary
        nc = min(pca, F.shape[1], len(tr) - 1) if pca else 0
        steps = [StandardScaler()] + ([PCA(nc, random_state=seed)] if nc else []) + [
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)]
        clf = make_pipeline(*steps).fit(F[tr], y[tr])
        return te, clf.classes_, clf.predict_proba(F[te])

    splits = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(F, y, groups))
    for te, cl, pr in Parallel(n_jobs=n_jobs)(delayed(fold)(tr, te) for tr, te in splits):
        proba[np.ix_(te, np.searchsorted(classes, cl))] = pr
    out = dict(bal_acc=float(balanced_accuracy_score(y, classes[proba.argmax(1)])), chance=1.0 / len(classes),
               n=int(len(y)), n_sessions=int(len(np.unique(groups))), n_classes=int(len(classes)))
    if "absence" in classes and len(classes) > 1:
        out["occupancy_auc"] = float(roc_auc_score(y == "absence", proba[:, list(classes).index("absence")]))
    return out
