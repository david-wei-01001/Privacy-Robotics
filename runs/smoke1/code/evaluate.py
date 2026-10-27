"""Evaluation (PLAN §7): calibration on val_calib, metrics on val_check, bootstrap CIs, comparison grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import build
from .matching import Prepared

FMRS = (0.01, 0.001)


def calibrate(s_benign: np.ndarray, fmr: float, discrete: bool) -> float:
    """Threshold such that the benign false-match rate on the calibration windows is at most `fmr`."""
    if discrete:
        for m in range(1, int(s_benign.max(initial=0)) + 2):
            if np.mean(s_benign >= m) <= fmr:
                return float(m)
        return float(s_benign.max(initial=0) + 1)
    s = np.sort(s_benign)[::-1]
    k = int(np.floor(fmr * len(s)))
    # the next representable score above s[k], so windows tied with s[k] never count as matches
    return float(np.nextafter(s[k], np.inf)) if k < len(s) else float(s[-1])


def _event_recall(hit: np.ndarray, events: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    ev, inv = np.unique(events, return_inverse=True)
    per = np.zeros(len(ev), bool)
    np.logical_or.at(per, inv, hit)
    return float(per.mean()) if len(ev) else float("nan"), per, ev


def _boot(values: np.ndarray, groups: np.ndarray | None = None, B: int = 1000, seed: int = 0):
    """95% bootstrap CI of a mean; resampling whole groups when groups are given."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(values), (B, len(values)))
        stats = values[idx].mean(1)
    else:
        g, inv = np.unique(groups, return_inverse=True)
        sums, cnts = np.bincount(inv, values, len(g)), np.bincount(inv, minlength=len(g))
        pick = rng.integers(0, len(g), (B, len(g)))
        stats = sums[pick].sum(1) / np.maximum(cnts[pick].sum(1), 1)
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def track_masks(W: pd.DataFrame, track: str, modality: str) -> dict[str, np.ndarray]:
    xd, sins, tv = (W.dataset == "xd").values, (W.dataset == "sins").values, (W.dataset == "tv_mix").values
    has_a = W.has_audio.values.astype(bool)
    viol = xd & (W.label == "violent").values
    if modality == "audio":
        viol = viol & has_a
    split = W.split.values
    if track == "home":
        benign = sins
        hard = tv
    else:  # "xd": benign windows from normal XD videos
        benign = xd & (W.label == "benign").values & (has_a if modality == "audio" else True)
        hard = np.zeros(len(W), bool)
    return dict(ref=viol & (split == "reference"), benign_train=benign & (split == "benign_train"),
                calib=benign & (split == "val_calib"), check=benign & (split == "val_check"),
                violent=viol & (split == "val_violent"), hard=hard,
                tv_sins=sins & (split == "val_check") & (W.subclass == "watching_tv").values)


def evaluate_one(W, P: Prepared, masks, matcher_spec: dict, new_id, fmrs=FMRS) -> tuple[list[dict], np.ndarray]:
    """One matcher on one embedding: a row per FMR target, each with its own run id from `new_id()`."""
    spec = dict(matcher_spec)
    m = build("matcher", spec)
    m.fit(P, masks["ref"], masks["benign_train"])
    evalset = masks["calib"] | masks["check"] | masks["violent"] | masks["hard"] | masks["tv_sins"]
    idx = np.where(evalset)[0]
    s_all = np.full(len(W), np.nan, np.float32)
    s_all[idx] = m.score(P, idx)
    out = []
    for fmr in fmrs:
        tau = calibrate(s_all[masks["calib"]], fmr, m.discrete)
        hit = s_all >= tau
        chk = masks["check"]
        b_vals, b_groups = hit[chk].astype(float), W.group_id.values[chk]
        v = masks["violent"]
        rec_e, per_event, _ = _event_recall(hit[v], W.event_id.values[v])
        y = np.r_[np.ones(v.sum()), np.zeros(chk.sum())]
        sc = np.r_[s_all[v], s_all[chk]]
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, sc)) if 0 < y.sum() < len(y) else float("nan")
        except ValueError:
            auc = float("nan")
        row = dict(run_id=new_id(), matcher=spec.get("name"),
                   params={k: v for k, v in matcher_spec.items() if k != "name"}, fmr_target=fmr, tau=tau,
                   fmr_calib=float(np.mean(hit[masks["calib"]])), fmr_check=float(b_vals.mean()) if len(b_vals) else float("nan"),
                   fmr_check_ci=_boot(b_vals, b_groups), recall_window=float(hit[v].mean()) if v.any() else float("nan"),
                   recall_event=rec_e, recall_event_ci=_boot(per_event.astype(float)), auc=auc,
                   tv_fmr=float(hit[masks["hard"]].mean()) if masks["hard"].any() else float("nan"),
                   tv_sins_fmr=float(hit[masks["tv_sins"]].mean()) if masks["tv_sins"].any() else float("nan"),
                   n_events=int(len(per_event)), n_check=int(chk.sum()), n_calib=int(masks["calib"].sum()),
                   n_ref=int(masks["ref"].sum()), lsh_coverage=getattr(m, "coverage", None))
        out.append(row)
    return out, s_all


# Dataset settings (PLAN §12.3, section 1). The encoder, matcher, and fitted post-processing stay fixed; only the
# source of the reference events, the violent test events, and the benign windows (calibration / measurement) varies.
DATASET_SETTINGS = {
    "all_sources": "references and test events from all XD sources; benign windows from SINS",
    "events_movie": "all references; test events from movies only",
    "events_youtube": "all references; test events from YouTube only",
    "ref_movie__events_movie": "references from movies only; test events from movies",
    "ref_movie__events_youtube": "references from movies only; test events from YouTube",
    "benign_xd_normal": "benign windows (calibration and measurement) from normal XD videos instead of SINS",
    "calib_sins__check_xd_normal": "threshold calibrated on SINS, FMR measured on normal XD videos",
    "calib_xd_normal__check_sins": "threshold calibrated on normal XD videos, FMR measured on SINS",
}


def dataset_comparisons(W, P: Prepared, masks_home, masks_xd, matcher_spec: dict, new_id) -> dict:
    """Each setting in DATASET_SETTINGS is one evaluate_one() run per FMR target, on the home-track embedding."""
    src = W.source.values
    movie, youtube = src == "movie", src == "youtube"
    mh, mx = masks_home, masks_xd
    variants = {
        "all_sources": mh,
        "events_movie": {**mh, "violent": mh["violent"] & movie},
        "events_youtube": {**mh, "violent": mh["violent"] & youtube},
        "ref_movie__events_movie": {**mh, "ref": mh["ref"] & movie, "violent": mh["violent"] & movie},
        "ref_movie__events_youtube": {**mh, "ref": mh["ref"] & movie, "violent": mh["violent"] & youtube},
        "benign_xd_normal": {**mh, "calib": mx["calib"], "check": mx["check"], "hard": np.zeros(len(W), bool),
                             "tv_sins": np.zeros(len(W), bool)},
        "calib_sins__check_xd_normal": {**mh, "check": mx["check"], "hard": np.zeros(len(W), bool),
                                        "tv_sins": np.zeros(len(W), bool)},
        "calib_xd_normal__check_sins": {**mh, "calib": mx["calib"]},
    }
    res = {"settings": {}, "skipped": {}}
    for name, mk in variants.items():
        need = {"ref": 2, "calib": 10, "check": 10, "violent": 1}
        short = [k for k, n in need.items() if mk[k].sum() < n]
        if short:
            res["skipped"][name] = "too few windows in " + ", ".join(short)
            continue
        rows, s_all = evaluate_one(W, P, mk, matcher_spec, new_id)
        for r in rows:
            r.update(setting=name, description=DATASET_SETTINGS[name])
            if name == "all_sources":  # hard negatives broken down at the same threshold
                hit = s_all >= r["tau"]
                hard, chk = mh["hard"], mh["check"]
                r["tv_mix_by_snr"] = {s: float(np.mean(hit[hard & (W.subclass.values == s)]))
                                      for s in sorted(W.subclass[hard].unique())} if hard.any() else {}
                r["sins_fmr_by_activity"] = {c: float(np.mean(hit[chk & (W.subclass.values == c)]))
                                             for c in sorted(W.subclass[chk].unique())}
        res["settings"][name] = rows
    return res
