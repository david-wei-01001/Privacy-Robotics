"""Reconciling the audio decision and the video decision (PLAN §6.4).

Each modality has its own matcher and its own score per window. This module (1) puts the two scores on one scale,
(2) combines them with a rule, and (3) compares methods at the SAME MEASURED false-match rate, because a method that
spends more false matches recalls more, and calibrating several methods to one nominal target does not give them one
measured FMR (a discrete LSH score cannot hit the target, and a threshold set on a few benign videos need not transfer).

Scale. A matcher score (a cosine, or the number of matching LSH bands) means different things per modality, so each is
mapped to z = -log P(benign score >= s), with P estimated on the benign-train windows (not on the windows used to
compare methods; for a learned matcher such as logreg these scores are in-sample, which inflates z a little). z is
monotone in the score, so it does not change a single modality's decision; it makes the two scores commensurable. Rules, with their protocol
analogue:

  max  alarm when either modality is unusual enough        OR of two per-modality server decisions
  sum  evidence accumulates across modalities (Fisher)     both modalities' vouchers count toward one threshold
  min  alarm only when both are unusual                    AND; needs per-window grouping or nested secret sharing
  early  one matcher on the concatenated embedding         one PSI run, one key per window

Two comparisons are reported.
  matched   recall at a fixed FMR measured on the benign val windows (calib + check, so more benign videos). The
            threshold is chosen on those windows for every method alike, with a time-sharing mixture of the two
            nearest deterministic rules where scores tie, so the FMR is exactly the stated one. It measures how well
            a method separates, not how well a threshold transfers. The bootstrap resamples benign VIDEOS and events,
            and recomputes the threshold in every resample.
  deployed  the threshold calibrated on val_calib and measured on val_check, as evaluate_one does. Rows carry the
            counts (false-matching windows and videos) that show how few benign units a measured FMR rests on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import build
from .evaluate import FMRS, _boot, _event_recall, calibrate
from .matching import Prepared, l2n

RULES = {"max": np.maximum, "min": np.minimum, "sum": np.add}


def tail_z(s: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """-log P(benign score >= s), the benign score distribution taken from the windows selected by `ref`."""
    z = np.full(len(s), np.nan)
    ok = np.isfinite(s)
    b = np.sort(s[ref & ok])
    p = (len(b) - np.searchsorted(b, s[ok], side="left") + 1) / (len(b) + 1)  # add-one: never exactly 0
    z[ok] = -np.log(p)
    return z


# ----------------------------------------------------------------------------- matched-FMR comparison
def recall_at_fmr(b: np.ndarray, e: np.ndarray, f: float) -> float:
    """Event recall at exactly FMR `f`. `b`: benign window scores; `e`: the maximum score of each violent event (an
    event is detected when any of its windows is). Where scores tie at the threshold t, the answer is the mixture of
    the strict rule (score > t) and the lenient rule (score >= t) that gives the FMR exactly `f`."""
    n, c = len(b), f * len(b)
    srt = np.sort(b)[::-1]
    i = int(np.floor(c))
    if i >= n:
        return float(np.mean(e >= srt[-1]))
    t = srt[i]
    gt, ge = np.count_nonzero(b > t), np.count_nonzero(b >= t)  # gt <= c < ge by construction
    q = (c - gt) / (ge - gt)
    return float((1 - q) * np.mean(e > t) + q * np.mean(e >= t))


def _event_max(W, s: np.ndarray, viol: np.ndarray) -> np.ndarray:
    ev, inv = np.unique(W.event_id.values[viol], return_inverse=True)
    out = np.full(len(ev), -np.inf)
    np.maximum.at(out, inv, s[viol])
    return out


def matched_fmr(W, scores: dict, masks: dict, fmrs=FMRS, B: int = 500, seed: int = 0, pairs=None) -> list[dict]:
    """Per FMR: recall of every method at that measured FMR (95% bootstrap CI over benign videos and events), and
    paired differences (same resamples). `pairs` lists (method, versus); by default every combined method against
    each single modality ("audio", "video")."""
    pool = masks["calib"] | masks["check"]
    groups, ginv = np.unique(W.group_id.values[pool], return_inverse=True)
    by_group = [np.where(ginv == i)[0] for i in range(len(groups))]
    n_ev = len(np.unique(W.event_id.values[masks["violent"]]))
    sb = {k: s[pool] for k, (s, _) in scores.items()}
    se = {k: _event_max(W, s, masks["violent"]) for k, (s, _) in scores.items()}
    rng = np.random.default_rng(seed)
    draws = [(np.concatenate([by_group[i] for i in rng.integers(0, len(groups), len(groups))]),
              rng.integers(0, n_ev, n_ev)) for _ in range(B)]
    out = []
    for f in fmrs:
        point = {k: recall_at_fmr(sb[k], se[k], f) for k in scores}
        boot = {k: np.array([recall_at_fmr(sb[k][bi], se[k][ei], f) for bi, ei in draws]) for k in scores}
        ci = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))  # noqa: E731
        pr = pairs if pairs is not None else [(k, ref) for k in scores if k not in ("audio", "video")
                                              for ref in ("audio", "video")]
        paired = [dict(method=k, versus=ref, diff=point[k] - point[ref], diff_ci=ci(boot[k] - boot[ref]))
                  for k, ref in pr]
        out.append(dict(fmr=f, n_benign_videos=int(len(groups)), n_benign_windows=int(pool.sum()), n_events=n_ev,
                        rows=[dict(method=k, recall=point[k], ci=ci(boot[k])) for k in scores], paired=paired))
    return out


# ----------------------------------------------------------------------------- deployed (calibrated) comparison
def _row(W, s: np.ndarray, masks: dict, fmr: float, discrete: bool):
    tau = calibrate(s[masks["calib"]], fmr, discrete)
    hit = s >= tau
    chk, v = masks["check"], masks["violent"]
    b = hit[chk].astype(float)
    rec, per, _ = _event_recall(hit[v], W.event_id.values[v])
    fm = hit & chk
    row = dict(fmr_target=fmr, tau=float(tau), fmr_calib=float(hit[masks["calib"]].mean()), fmr_check=float(b.mean()),
               fmr_check_ci=_boot(b, W.group_id.values[chk]), recall_event=rec,
               recall_event_ci=_boot(per.astype(float)), recall_window=float(hit[v].mean()), n_events=int(len(per)),
               n_fm_windows=int(fm.sum()), n_fm_videos=int(len(np.unique(W.group_id.values[fm]))),
               n_check_videos=int(len(np.unique(W.group_id.values[chk]))))
    return row, per


def evaluate_methods(W, scores: dict, masks: dict, fmrs=FMRS) -> dict:
    """`scores`: method -> (score per window, matcher is discrete). One calibrated row per method and FMR target."""
    return dict(rows=[dict(method=name, **_row(W, s, masks, fmr, disc)[0])
                      for fmr in fmrs for name, (s, disc) in scores.items()])


def threshold_setters(W, s: np.ndarray, masks: dict, fmr: float, top: int = 3) -> dict:
    """Which benign calibration videos supply the top-`fmr` windows, i.e. set the threshold? If a few videos supply
    nearly all of them, the calibrated FMR rests on those few videos and need not transfer to others."""
    c = masks["calib"]
    g = W.group_id.values[c]
    k = max(1, int(np.floor(fmr * c.sum())))
    vc = pd.Series(g[np.argsort(-s[c])[:k]]).value_counts()
    return dict(fmr=fmr, top_windows=k, videos_supplying=int(len(vc)), calib_videos=int(len(np.unique(g))),
                share_of_top_videos=float(vc.head(top).sum() / k), top=[(str(a), int(b)) for a, b in vc.head(top).items()])


def agreement(W, s_a: np.ndarray, s_v: np.ndarray, masks: dict, fmr: float, disc_a: bool, disc_v: bool) -> dict:
    """Each modality alone at its own calibrated threshold: which violent events does each detect, and do the benign
    false matches coincide? Counts are given, because with a handful of false-matching windows the rates mean little."""
    ta = calibrate(s_a[masks["calib"]], fmr, disc_a)
    tv = calibrate(s_v[masks["calib"]], fmr, disc_v)
    ha, hv = s_a >= ta, s_v >= tv
    v, c = masks["violent"], masks["check"]
    _, pa, _ = _event_recall(ha[v], W.event_id.values[v])
    _, pv, _ = _event_recall(hv[v], W.event_id.values[v])
    fa, fv = ha[c], hv[c]
    ra, rv, both = float(fa.mean()), float(fv.mean()), float((fa & fv).mean())
    return dict(fmr_target=fmr, tau_audio=float(ta), tau_video=float(tv),
                events=dict(both=int((pa & pv).sum()), audio_only=int((pa & ~pv).sum()),
                            video_only=int((~pa & pv).sum()), neither=int((~pa & ~pv).sum())),
                benign=dict(audio=ra, video=rv, both=both, both_if_independent=ra * rv,
                            n_audio=int(fa.sum()), n_video=int(fv.sum()), n_both=int((fa & fv).sum()),
                            p_video_given_audio=both / ra if ra > 0 else float("nan")))


def _scores(W, P: Prepared, masks: dict, matcher_spec: dict) -> np.ndarray:
    """Matcher score of every window that is evaluated or that defines the benign scale (benign_train as well, which
    evaluate_one leaves unscored). NaN elsewhere."""
    mt = build("matcher", dict(matcher_spec))
    mt.fit(P, masks["ref"], masks["benign_train"])
    idx = np.where(masks["calib"] | masks["check"] | masks["violent"] | masks["benign_train"])[0]
    out = np.full(len(W), np.nan, np.float32)
    out[idx] = mt.score(P, idx)
    return out


def run(W, E_a: np.ndarray, E_v: np.ndarray, masks: dict, matcher_spec: dict, fmrs=FMRS,
        rules=("max", "sum", "min"), B: int = 500) -> dict:
    """Audio only, video only, early fusion and the late rules on the windows that have both embeddings."""
    ok = np.isfinite(E_a[:, 0]) & np.isfinite(E_v[:, 0])
    m = {k: v & ok for k, v in masks.items()}
    clean_a, clean_v = np.nan_to_num(E_a), np.nan_to_num(E_v)
    spaces = {"audio": clean_a, "video": clean_v, "early": np.concatenate([l2n(clean_a), l2n(clean_v)], 1)}
    s = {}
    for key, E in spaces.items():
        s[key] = _scores(W, Prepared(E, m["ref"], m["benign_train"]), m, matcher_spec)
    disc = bool(build("matcher", dict(matcher_spec)).discrete)
    z_a, z_v = tail_z(s["audio"], m["benign_train"]), tail_z(s["video"], m["benign_train"])
    scores = {"audio": (z_a, False), "video": (z_v, False), "early": (s["early"], disc)}
    for r in rules:
        scores[f"late_{r}"] = (RULES[r](z_a, z_v), False)
    return dict(matcher=matcher_spec, windows={k: int(v.sum()) for k, v in m.items()},
                matched=matched_fmr(W, scores, m, fmrs, B), deployed=evaluate_methods(W, scores, m, fmrs),
                agreement=[agreement(W, s["audio"], s["video"], m, f, disc, disc) for f in fmrs],
                setters={k: threshold_setters(W, s[k], m, fmrs[0]) for k in ("audio", "video")})


def format_result(res: dict) -> str:
    L = [f"matcher {res['matcher']}", f"windows with both modalities: {res['windows']}", ""]
    for mm in res["matched"]:
        L += [f"MATCHED FMR {mm['fmr']:.2%} (measured on {mm['n_benign_videos']} benign videos, "
              f"{mm['n_benign_windows']} windows; {mm['n_events']} events):"]
        for r in mm["rows"]:
            L.append(f"  {r['method']:<9} recall {r['recall']:>6.1%}  [{r['ci'][0]:>4.0%},{r['ci'][1]:>4.0%}]")
        for p in mm["paired"]:
            lo, hi = p["diff_ci"]
            L.append(f"  {p['method']:<9} minus {p['versus']:<5} {p['diff']:+6.1%}  [{lo:+.1%},{hi:+.1%}]"
                     + ("   <- CI excludes 0" if lo > 0 or hi < 0 else ""))
        L.append("")
    L += ["DEPLOYED (threshold calibrated on val_calib, measured on val_check; not comparable across rows unless the",
          "measured FMRs are):", f"  {'method':<9} {'target':>7} {'measured':>9} {'FM wins/vids':>13} {'recall':>7}  95% CI"]
    for r in res["deployed"]["rows"]:
        lo, hi = r["recall_event_ci"]
        L.append(f"  {r['method']:<9} {r['fmr_target']:>7.2%} {r['fmr_check']:>9.3%} "
                 f"{r['n_fm_windows']:>6d}/{r['n_fm_videos']:<2d} of {r['n_check_videos']:<3d} {r['recall_event']:>6.1%}  "
                 f"[{lo:.0%},{hi:.0%}]")
    L += ["", "WHICH BENIGN CALIBRATION VIDEOS SET THE THRESHOLD (top windows at the first FMR target):"]
    for k, d in res["setters"].items():
        L.append(f"  {k:<6}: the top {d['top_windows']} windows come from {d['videos_supplying']} of {d['calib_videos']} "
                 f"videos; the top 3 supply {d['share_of_top_videos']:.0%}  {d['top']}")
    for a in res["agreement"]:
        e, b = a["events"], a["benign"]
        L += ["", f"AGREEMENT at FMR target {a['fmr_target']:.2%} (each modality at its own calibrated threshold):",
              f"  violent events: both {e['both']}, audio only {e['audio_only']}, video only {e['video_only']}, "
              f"neither {e['neither']}",
              f"  benign false-matching windows: audio {b['n_audio']}, video {b['n_video']}, both {b['n_both']}"
              f"  (rates {b['audio']:.3%} / {b['video']:.3%} / {b['both']:.3%}; independent would give "
              f"{b['both_if_independent']:.3%})"]
    return "\n".join(L)
