"""What the PSI server sees, dummy traffic, and an epoch replay of the reporting decision (PLAN §7, §10; RQ3, RQ4).

SERVER VIEW. In the Apple protocol as implemented (apple_psi/server.py: `ingest` returns match / non-match and keeps a
record whose share coordinate is a keyed PRF of a random id) the server learns, per voucher, only whether it opened. It
does NOT learn which table entry matched, so neither the band nor the k-bit code. Grouped by window, its view is the
COUNT of opened vouchers: `opened.sum(1)`, which equals the matcher's score (`LSHExact.score`).
`ServerView` additionally carries `opened` per band and the matching `codes`. Those are WHAT-IF information: they
describe a stream sent in band order, or a design whose server learns the matched entry. Use them as upper bounds on
weaker designs, never as the leakage of the protocol itself.

DUMMY TRAFFIC. A synthetic voucher opens like a real match and carries no usable share, so the server cannot tell it
from a real match below the threshold. `add_synthetic` spreads `s_max` synthetic matches per epoch uniformly over the
epoch's closed vouchers, each opening a random entry of its band's table. The price is voucher size: 260 + 8 (s_max+1)
+ ad_len bytes (checked against the prototype). Synthetic vouchers never count towards the reporting threshold, so the
replay's decisions use real opens only; only the server's view of the stream changes.

EPOCH REPLAY. Benign epochs are built by block bootstrap from benign videos: `epoch_windows` windows from blocks of
`block` consecutive windows of one video, resampled with replacement. A positive epoch replaces one block by a
violent event. Every rule reduces an epoch to a sum over its windows, calibrated to a false-alarm probability `alpha`
per epoch on benign epochs, then read as recall:
    A     alarm when the epoch's opened vouchers exceed t        (the unmodified protocol: m = 1, no per-window grouping)
    B     a window is flagged when >= m of its bands open (m set at window FMR f); alarm when flagged windows exceed t
          (needs the per-window grouping / banded evidence key)
    D     a window is flagged by a local classifier at window FMR f; same count rule (threshold encryption + policy)
    COS   as D with the maximum cosine to any reference (what a fuzzy PSI with a cosine threshold computes)
The bootstrap resamples benign MOVIES (group_id) and events and recomputes every threshold in each resample.
Benign data here are XD normal videos, which open more vouchers than household audio does, so thresholds calibrated on
them are conservative for a home. SINS has no usable chronology, so an epoch is a synthetic arrangement of blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .evaluate import calibrate
from .matching import Prepared, simhash_codes

AD_LEN = 64
HOP_S = 2.4


def voucher_bytes(s_max: int, ad_len: int = AD_LEN) -> int:
    """Size of one voucher (measured on apple_psi: 332 / 412 / 1132 / 8332 bytes for s_max 0 / 10 / 100 / 1000)."""
    return 260 + 8 * (s_max + 1) + ad_len


def bytes_per_day(L: int, s_max: int, hop_s: float = HOP_S, ad_len: int = AD_LEN) -> float:
    """Upload per day: one voucher per band per window."""
    return (86400.0 / hop_s) * L * voucher_bytes(s_max, ad_len)


# ----------------------------------------------------------------------------- the server's view
@dataclass
class ServerView:
    k: int
    L: int
    seed: int
    opened: np.ndarray  # (N, L) bool
    codes: np.ndarray   # (N, L) int64; the revealed band code where the voucher opened, -1 where it did not
    tables: list        # per band: sorted array of the reference codes (the server's table)


def server_view(P: Prepared, idx: np.ndarray, ref: np.ndarray, k: int, L: int, seed: int = 0) -> ServerView:
    ref_codes = simhash_codes(P.unit[ref], k, L, seed)
    tables = [np.unique(ref_codes[:, b]) for b in range(L)]
    codes = simhash_codes(P.unit[idx], k, L, seed)
    opened = np.stack([np.isin(codes[:, b], tables[b]) for b in range(L)], 1)
    return ServerView(k, L, seed, opened, np.where(opened, codes, -1), tables)


def add_synthetic(view: ServerView, s_max: int, epoch_windows: int, rng: np.random.Generator) -> ServerView:
    """Dummy traffic: s_max synthetic matches per epoch of `epoch_windows` windows, uniformly over closed vouchers."""
    if s_max <= 0:
        return view
    N, L = view.opened.shape
    q = min(1.0, s_max / (epoch_windows * L))
    flip = (~view.opened) & (rng.random((N, L)) < q)
    codes = view.codes.copy()
    for b in range(L):
        rows = np.where(flip[:, b])[0]
        if len(rows):
            codes[rows, b] = rng.choice(view.tables[b], len(rows))
    return replace(view, opened=view.opened | flip, codes=codes)


# ----------------------------------------------------------------------------- per-window rule statistics
def rule_stats(W, P: Prepared, masks: dict, cells, seeds, f_ws=(0.01, 0.001)):
    """Per-window quantity that each rule sums over an epoch (NaN outside the windows used). Window-level thresholds
    (m for B, the score threshold for D and COS) are set on benign calibration windows at window FMR f."""
    from .fusion import _scores
    idx = np.where(masks["calib"] | masks["check"] | masks["violent"])[0]
    cal = masks["calib"][idx]
    X, thr = {}, {}

    def put(name, vals):
        full = np.full(len(W), np.nan)
        full[idx] = vals
        X[name] = full

    for k, L in cells:
        for sd in seeds:
            cnt = server_view(P, idx, masks["ref"], k, L, sd).opened.sum(1).astype(float)
            tag = f"k{k}L{L}s{sd}"
            put(f"A|{tag}", cnt)
            for f in f_ws:
                m = calibrate(cnt[cal], f, True)
                thr[f"B{f:g}|{tag}"] = float(m)
                put(f"B{f:g}|{tag}", (cnt >= m).astype(float))
    for name, short in (("logreg", "D"), ("cosine_top1", "COS")):
        s = _scores(W, P, masks, {"name": name})[idx]
        for f in f_ws:
            tau = calibrate(s[cal], f, False)
            thr[f"{short}{f:g}"] = float(tau)
            put(f"{short}{f:g}", (s >= tau).astype(float))
    return X, thr


# ----------------------------------------------------------------------------- epochs
def make_blocks(W, mask: np.ndarray, block: int):
    """Full blocks of `block` consecutive windows of one clip: list of (window indices, movie)."""
    i = np.where(mask)[0]
    d = pd.DataFrame({"i": i, "clip": W.clip_id.values[i], "t0": W.t0.values[i], "g": W.group_id.values[i]})
    d = d.sort_values(["clip", "t0"])
    d["blk"] = d.groupby("clip").cumcount() // block
    out = []
    for _, g in d.groupby(["clip", "blk"], sort=False):
        if len(g) == block:
            out.append((g["i"].values, g["g"].iloc[0]))
    return out


def make_events(W, mask: np.ndarray, cap: int):
    """Violent events: window indices in time order, at most `cap` windows."""
    i = np.where(mask)[0]
    d = pd.DataFrame({"i": i, "ev": W.event_id.values[i], "t0": W.t0.values[i]}).sort_values(["ev", "t0"])
    return [g["i"].values[:cap] for _, g in d.groupby("ev", sort=False)]


def _rep(bstat, estat, pool, ev_idx, n_benign, per_event, n_blk, alphas, rng):
    R = bstat.shape[1]
    ben = bstat[rng.choice(pool, (n_benign, n_blk))].sum(1)                        # (n_benign, R)
    srt = np.sort(ben, axis=0)
    thr = np.stack([srt[min(int(np.ceil((1 - a) * n_benign)) - 1, n_benign - 1)] for a in alphas])  # (A, R)
    far = (ben[None] > thr[:, None, :]).mean(1)                                    # (A, R) measured benign alarm rate
    bg = bstat[rng.choice(pool, (len(ev_idx), per_event, n_blk - 1))].sum(2)       # (E, per_event, R)
    pos = bg + estat[ev_idx][:, None, :]
    rec = (pos[None] > thr[:, None, None, :]).mean((1, 2))                         # (A, R)
    return rec, far, thr


def epoch_replay(W, X: dict, masks: dict, alphas=(1e-2, 1e-3), epoch_windows: int = 250, block: int = 10,
                 event_cap: int = 30, n_benign: int = 4000, per_event: int = 20, B: int = 100, seed: int = 0) -> dict:
    """Recall of every rule in X at false-alarm probability alpha per epoch; bootstrap over benign movies and events."""
    names = list(X)
    Xm = np.stack([np.nan_to_num(X[n]) for n in names], 1)
    blocks = make_blocks(W, masks["calib"] | masks["check"], block)
    events = make_events(W, masks["violent"], event_cap)
    bstat = np.stack([Xm[ix].sum(0) for ix, _ in blocks])
    estat = np.stack([Xm[ix].sum(0) for ix in events])
    grp = np.array([g for _, g in blocks])
    by_group = {g: np.where(grp == g)[0] for g in np.unique(grp)}
    n_blk, rng = epoch_windows // block, np.random.default_rng(seed)
    rec0, far0, thr0 = _rep(bstat, estat, np.arange(len(blocks)), np.arange(len(events)), 4 * n_benign, per_event,
                            n_blk, alphas, rng)
    boot = np.empty((B, len(alphas), len(names)))
    gl = list(by_group)
    for b in range(B):
        pool = np.concatenate([by_group[gl[j]] for j in rng.integers(0, len(gl), len(gl))])
        boot[b] = _rep(bstat, estat, pool, rng.integers(0, len(events), len(events)), n_benign, per_event, n_blk,
                       alphas, rng)[0]
    return dict(names=names, alphas=list(alphas), recall=rec0, benign_alarm_rate=far0, threshold=thr0, boot=boot,
                n_blocks=len(blocks), n_movies=len(by_group), n_events=len(events), epoch_windows=epoch_windows,
                epochs_per_week=7 * 86400 / (epoch_windows * HOP_S))
