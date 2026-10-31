#!/usr/bin/env python
"""One-screen summary of a finished night, from reports/<night>/facts.json.

    .venv/bin/python tools/summarize_night.py 2026-10-29_n2            # XD track, 1% target
    .venv/bin/python tools/summarize_night.py 2026-10-29_n2 --track home --fmr 0.001

Per encoder: cosine top-1 (AUC does not depend on any threshold; recall and measured FMR do), the LSH setting with the
best SEED-MEAN recall (the best single seed would overstate it), and logistic regression. Then the Apple PSI
equivalence checks, the encoder timings, and where the logs and reports are.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def pct(x):
    return "   n/a" if pd.isna(x) else f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("night")
    ap.add_argument("--track", default="xd")
    ap.add_argument("--fmr", type=float, default=0.01)
    a = ap.parse_args()
    f = json.loads((REPO / "reports" / a.night / "facts.json").read_text())
    R = pd.DataFrame(f["runs"])
    R = R[(R.track == a.track) & (R.fmr_target == a.fmr)]
    mod = {k: v["modality"] for k, v in f.get("encoders", {}).items()}
    print(f"night {a.night}: {f['meta'].get('minutes')} min; track {a.track}, target FMR {a.fmr:.2%}; counts {len(R)} runs\n")
    print(f"{'encoder':<26}{'mod':<6}| {'cos1 AUC':>8} {'recall':>7} {'FMR':>7} | {'LSH (k,L) seed-mean':<21}{'recall':>7} {'+-sd':>6} {'FMR':>7} | {'logreg':>7} {'FMR':>7}")
    for enc, g in R.groupby("encoder", sort=False):
        c = g[g.matcher == "cosine_top1"]
        c = c.iloc[0] if len(c) else None
        L = g[g.matcher == "lsh"].copy()
        l_txt, l_rec, l_sd, l_fmr = "-", float("nan"), float("nan"), float("nan")
        if len(L):
            L["k"], L["L"] = L.params.map(lambda p: p["k"]), L.params.map(lambda p: p["L"])
            agg = L.groupby(["k", "L"]).agg(rec=("recall_event", "mean"), sd=("recall_event", "std"), fmr=("fmr_check", "mean"))
            (k, l), b = agg.rec.idxmax(), agg.loc[agg.rec.idxmax()]
            l_txt, l_rec, l_sd, l_fmr = f"k{k} L{l}", b.rec, b.sd, b.fmr
        lg = g[g.matcher == "logreg"]
        lg = lg.iloc[0] if len(lg) else None
        m = mod.get(enc, "fusion" if enc.startswith("fusion:") else "?")
        print(f"{enc:<26}{m:<6}| {c.auc if c is not None else float('nan'):>8.3f} {pct(c.recall_event if c is not None else float('nan'))} "
              f"{pct(c.fmr_check if c is not None else float('nan'))} | {l_txt:<21}{pct(l_rec)} {pct(l_sd)} {pct(l_fmr)} | "
              f"{pct(lg.recall_event if lg is not None else float('nan'))} {pct(lg.fmr_check if lg is not None else float('nan'))}")
    print("\nApple PSI equivalence (server decision == plaintext LSH decision):")
    eq = {"home:" + f["equivalence"]["encoder"]: f["equivalence"]} if f.get("equivalence") else {}
    eq.update({"xd:" + k: v for k, v in (f.get("equivalence_xd") or {}).items()})
    for k, v in eq.items():
        print(f"  {k:<30} k={v['k']} L={v['L']} m={v['m']}  rate {v['equivalence_rate']:.3f} ({v['n_disagree']} disagree, "
              f"{v['n_disagree_explained_by_drops']} explained by cuckoo drops)  voucher {v['voucher_bytes']} B  "
              f"client {v['client_ms_per_voucher']} ms  server {v['server_ms_per_voucher']} ms  setup {v['setup_s']} s")
    ms = f.get("scale", {}).get("ms_per_window", {})
    print("\nprobe ms/window:", ", ".join(f"{k} {v}" for k, v in ms.items()))
    print(f"logs: data/logs/{a.night}/pilot.log   reports: reports/{a.night}/ (comparison.md, feasibility.md, facts.json)")


if __name__ == "__main__":
    main()
