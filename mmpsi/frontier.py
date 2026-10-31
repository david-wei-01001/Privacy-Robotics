"""Utility versus a leakage proxy for one embedding: every matcher at the same measured FMR (PLAN §7, RQ3, RQ4).

For one embedding this puts side by side
  * the matchers that need no secret list (logreg: a classifier the device could hold, i.e. plain threshold encryption
    with a policy check) and the matchers that match against the reference list (cosine top-1 / top-5, LSH);
  * several LSH settings (k bits per band, L bands), each over several hash seeds, because one seed is one random draw
    and the seed-to-seed spread is as large as many differences between settings.
Recall is read at a fixed measured FMR (see fusion.matched_fmr), with a bootstrap over benign videos and events.

Leakage proxy. In the Apple protocol as implemented the server learns, for every voucher, only whether it opened. For a
benign window that is the number of its L band-vouchers that open (`opened_benign`): a graded similarity score of every
benign window to the reference list, visible once the vouchers are grouped by window. `bits_per_benign_window` = k x
opened_benign counts the code bits that WOULD be exposed by a design whose server learns the matched entry; the protocol
as implemented does not expose them, so read it as a what-if bound. It depends only on (k, L, seed) and the benign
windows, not on the decision threshold m.
"""
from __future__ import annotations

import numpy as np

from .fusion import _scores, matched_fmr
from .matching import Prepared

DEFAULT_GRID = ((12, 32), (16, 16), (16, 32), (24, 16))


def frontier(W, E: np.ndarray, masks: dict, grid=DEFAULT_GRID, seeds=(0, 1, 2, 3, 4), fmrs=(0.01, 0.001),
             B: int = 500) -> dict:
    ok = np.isfinite(E[:, 0])
    m = {k: v & ok for k, v in masks.items()}
    P = Prepared(np.nan_to_num(E), m["ref"], m["benign_train"])
    scores, leak = {}, {}
    for name in ("cosine_top1", "cosine_top5", "logreg"):
        scores[name] = (_scores(W, P, m, {"name": name}), False)
    for k, L in grid:
        for sd in seeds:
            key = f"lsh_k{k}_L{L}_s{sd}"
            s = _scores(W, P, m, {"name": "lsh", "k": k, "L": L, "seed": sd})
            scores[key] = (s, True)
            b, v = s[m["check"]], s[m["violent"]]
            leak[key] = dict(k=k, L=L, seed=sd, opened_benign=float(b.mean()), p_any_open=float((b >= 1).mean()),
                             opened_violent=float(v.mean()), bits_per_benign_window=float(k * b.mean()))
    lsh_keys = [k for k in scores if k.startswith("lsh_")]
    pairs = [("logreg", "cosine_top1"), ("logreg", "cosine_top5")] + [("logreg", k) for k in lsh_keys]
    matched = matched_fmr(W, scores, m, fmrs, B, pairs=pairs)
    return dict(windows={k: int(v.sum()) for k, v in m.items()}, matched=matched, leak=leak, grid=[list(g) for g in grid],
                seeds=list(seeds))


def format_frontier(res: dict, name: str = "") -> str:
    L = [f"{name}  windows: {res['windows']}", ""]
    for mm in res["matched"]:
        rec = {r["method"]: r for r in mm["rows"]}
        L += [f"MATCHED FMR {mm['fmr']:.2%}  ({mm['n_benign_videos']} benign videos, {mm['n_events']} events)",
              f"  {'method':<22}{'recall':>8}  {'95% CI':<12}  {'LSH seed spread / vouchers opened per benign window'}"]
        for k in ("cosine_top1", "cosine_top5", "logreg"):
            r = rec[k]
            L.append(f"  {k:<22}{r['recall']:>8.1%}  [{r['ci'][0]:>4.0%},{r['ci'][1]:>4.0%}]")
        for k_, l_ in res["grid"]:
            ks = [f"lsh_k{k_}_L{l_}_s{sd}" for sd in res["seeds"]]
            v = np.array([rec[k]["recall"] for k in ks])
            lk = [res["leak"][k] for k in ks]
            L.append(f"  {f'lsh k={k_} L={l_}':<22}{v.mean():>8.1%}  sd {v.std(ddof=1):4.1%} over {len(ks)} seeds"
                     f"   opens {np.mean([x['opened_benign'] for x in lk]):4.1f} of {l_} on a benign window "
                     f"(P(any) {np.mean([x['p_any_open'] for x in lk]):.0%}; {np.mean([x['bits_per_benign_window'] for x in lk]):.0f} bits only if matched entries were exposed)")
        best = max((k for k in rec if k.startswith("lsh_")), key=lambda k: rec[k]["recall"])
        for p in mm["paired"]:
            if p["versus"] in ("cosine_top1", best):
                lo, hi = p["diff_ci"]
                L.append(f"  logreg minus {p['versus']:<20} {p['diff']:+6.1%}  [{lo:+.1%},{hi:+.1%}]"
                         + ("   <- CI excludes 0" if lo > 0 or hi < 0 else "") + ("   (best LSH seed: favours LSH)" if p["versus"] == best else ""))
        L.append("")
    return "\n".join(L)
