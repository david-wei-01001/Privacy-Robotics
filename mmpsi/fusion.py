"""Reconciling the audio decision and the video decision (PLAN §6.4).

Each modality has its own matcher and its own score per window. This module (1) puts the two scores on one scale,
(2) combines them with a rule, (3) calibrates ONE threshold on benign windows at the target false-match rate, so that
every method is compared at the same FMR, and (4) reports where the modalities agree and disagree.

Scale. A matcher score (a cosine, or the number of matching LSH bands) means different things per modality, so each is
mapped to z = -log P(benign score >= s), with P estimated on the benign calibration windows. z is monotone in the
score, so it does not change a single modality's decision; it makes the two scores commensurable. Rules, with their
protocol analogue:

  max  alarm when either modality is unusual enough        OR of two per-modality server decisions
  sum  evidence accumulates across modalities (Fisher)     both modalities' vouchers count toward one threshold
  min  alarm only when both are unusual                    AND; needs per-window grouping or nested secret sharing
  early  one matcher on the concatenated embedding         one PSI run, one key per window (the `fusion:` rows)

The threshold is calibrated on val_calib windows and measured on val_check (as in evaluate_one); a method is better
only if its recall is higher at the same measured FMR, which `paired` tests on the same events.
"""
from __future__ import annotations

import itertools

import numpy as np

from .core import build
from .evaluate import FMRS, _boot, _event_recall, calibrate, evaluate_one
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


def _row(W, s: np.ndarray, masks: dict, fmr: float, discrete: bool):
    tau = calibrate(s[masks["calib"]], fmr, discrete)
    hit = s >= tau
    chk, v = masks["check"], masks["violent"]
    b = hit[chk].astype(float)
    rec, per, _ = _event_recall(hit[v], W.event_id.values[v])
    row = dict(fmr_target=fmr, tau=float(tau), fmr_calib=float(hit[masks["calib"]].mean()), fmr_check=float(b.mean()),
               fmr_check_ci=_boot(b, W.group_id.values[chk]), recall_event=rec,
               recall_event_ci=_boot(per.astype(float)), recall_window=float(hit[v].mean()), n_events=int(len(per)))
    return row, per


def evaluate_methods(W, scores: dict, masks: dict, fmrs=FMRS) -> dict:
    """`scores`: method -> (score per window, matcher is discrete). Rows per method and FMR, plus the paired
    difference in event recall of every combined method against each single modality (same events, so the
    confidence interval is far tighter than comparing two marginal intervals)."""
    rows, per_all = [], {}
    for fmr in fmrs:
        for name, (s, disc) in scores.items():
            row, per = _row(W, s, masks, fmr, disc)
            rows.append(dict(method=name, **row))
            per_all[(name, fmr)] = per
    paired = []
    for fmr in fmrs:
        for name in scores:
            if name in ("audio", "video"):
                continue
            for ref in ("audio", "video"):
                d = per_all[(name, fmr)].astype(float) - per_all[(ref, fmr)].astype(float)
                paired.append(dict(method=name, versus=ref, fmr_target=fmr, diff=float(d.mean()), diff_ci=_boot(d)))
    return dict(rows=rows, paired=paired)


def agreement(W, s_a: np.ndarray, s_v: np.ndarray, masks: dict, fmr: float, disc_a: bool, disc_v: bool) -> dict:
    """Each modality alone at its own threshold (both calibrated at `fmr`): which violent events does each detect, and
    do the benign false matches of the two modalities coincide?

    events: detected by both / audio only / video only / neither. Large audio-only and video-only counts mean the
    modalities are complementary (OR helps). benign: if P(video false match | audio false match) is far above the
    video rate, the false matches share a cause (a TV in view and in earshot) and AND helps little."""
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
                            p_video_given_audio=both / ra if ra > 0 else float("nan"),
                            p_audio_given_video=both / rv if rv > 0 else float("nan")))


def run(W, E_a: np.ndarray, E_v: np.ndarray, masks: dict, matcher_spec: dict, fmrs=FMRS,
        rules=("max", "sum", "min")) -> dict:
    """Audio only, video only, early fusion and the late rules on the windows that have both embeddings."""
    ok = np.isfinite(E_a[:, 0]) & np.isfinite(E_v[:, 0])
    m = {k: v & ok for k, v in masks.items()}
    ids = itertools.count()
    new_id = lambda: f"fx{next(ids):05d}"  # noqa: E731
    clean_a, clean_v = np.nan_to_num(E_a), np.nan_to_num(E_v)
    spaces = {"audio": clean_a, "video": clean_v, "early": np.concatenate([l2n(clean_a), l2n(clean_v)], 1)}
    s = {}
    for key, E in spaces.items():
        _, s[key] = evaluate_one(W, Prepared(E, m["ref"], m["benign_train"]), m, matcher_spec, new_id, fmrs)
    disc = bool(build("matcher", dict(matcher_spec)).discrete)
    z_a, z_v = tail_z(s["audio"], m["calib"]), tail_z(s["video"], m["calib"])
    scores = {"audio": (z_a, False), "video": (z_v, False), "early": (s["early"], disc)}
    for r in rules:
        scores[f"late_{r}"] = (RULES[r](z_a, z_v), False)
    out = evaluate_methods(W, scores, m, fmrs)
    out["agreement"] = [agreement(W, s["audio"], s["video"], m, f, disc, disc) for f in fmrs]
    out["windows"] = {k: int(v.sum()) for k, v in m.items()}
    out["matcher"] = matcher_spec
    return out


def format_result(res: dict) -> str:
    L = [f"matcher {res['matcher']}   windows with both modalities: {res['windows']}", ""]
    L.append(f"{'method':<10} {'FMR tgt':>8} {'FMR chk':>8} {'recall':>7}  {'95% CI':<13} events")
    for r in res["rows"]:
        lo, hi = r["recall_event_ci"]
        L.append(f"{r['method']:<10} {r['fmr_target']:>8.3%} {r['fmr_check']:>8.3%} {r['recall_event']:>7.1%}  "
                 f"[{lo:>4.0%},{hi:>4.0%}]    {r['n_events']}")
    L += ["", "paired difference in event recall (same events), method minus single modality:"]
    for p in res["paired"]:
        lo, hi = p["diff_ci"]
        flag = "  <- CI excludes 0" if lo > 0 or hi < 0 else ""
        L.append(f"  FMR {p['fmr_target']:.3%}  {p['method']:<9} vs {p['versus']:<5} {p['diff']:+6.1%}  "
                 f"[{lo:+.1%},{hi:+.1%}]{flag}")
    for a in res["agreement"]:
        e, b = a["events"], a["benign"]
        L += ["", f"agreement at FMR target {a['fmr_target']:.3%} (each modality at its own threshold):",
              f"  violent events: both {e['both']}, audio only {e['audio_only']}, video only {e['video_only']}, "
              f"neither {e['neither']}",
              f"  benign false matches: audio {b['audio']:.3%}, video {b['video']:.3%}, both {b['both']:.3%} "
              f"(independent would be {b['both_if_independent']:.3%}); P(video|audio) {b['p_video_given_audio']:.1%}"]
    return "\n".join(L)
