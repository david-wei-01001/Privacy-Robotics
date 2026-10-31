#!/usr/bin/env python
"""Utility and leakage of the PSI stream on the cached embeddings of a finished night. No encoding.

  attack   what a server-view adversary learns about household activity (SINS) from the benign vouchers, as a function
           of (k, L) and of the dummy-traffic budget s_max, with what each budget costs in upload.
  utility  event recall at a fixed number of false alarms per home-week, by epoch replay (XD in-domain), for the
           unmodified protocol (A), the grouped m-of-L variant (B), a public classifier (D) and a cosine threshold (COS).

    .venv/bin/python tools/replay.py attack  --night 2026-10-29_n2 --encoder ast --grid 12,32 --grid 16,32
    .venv/bin/python tools/replay.py utility --night 2026-10-29_n2 --encoder ast --encoder clip_frames \
        --encoder fusion:ast+clip_frames --grid 12,32 --grid 16,32 --seeds 3

Read mmpsi/replay.py and mmpsi/adversary.py first: they state what is modelled, what is not, and the caveats.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_eval import load  # noqa: E402
from mmpsi import adversary as adv  # noqa: E402
from mmpsi import replay as rp  # noqa: E402
from mmpsi.core import REPO, build  # noqa: E402
from mmpsi.evaluate import calibrate, track_masks  # noqa: E402
from mmpsi.matching import Prepared, l2n  # noqa: E402


def parse_grid(g):
    return [tuple(int(x) for x in s.split(",")) for s in g] if g else [(12, 32), (16, 32)]


def embedding(E, name):
    if name.startswith("fusion:"):
        x, y = name[len("fusion:"):].split("+")
        emb = np.concatenate([l2n(np.nan_to_num(E[x])), l2n(np.nan_to_num(E[y]))], 1)
        emb[~(np.isfinite(E[x][:, 0]) & np.isfinite(E[y][:, 0]))] = np.nan
        return emb, "video"
    return E[name], build("encoder", name).modality


def parts(names):
    return sorted({p for e in names for p in (e[len("fusion:"):].split("+") if e.startswith("fusion:") else [e])})


def out_dir(night, sub):
    d = REPO / "runs" / night / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- attack
def cmd_attack(a):
    W, E = load(a.night, [a.encoder])
    emb, modality = embedding(E, a.encoder)
    if modality != "audio":
        sys.exit("the household data (SINS) have audio only: use an audio encoder")
    masks = track_masks(W, "home", "audio")
    ok = np.isfinite(emb[:, 0])
    masks = {k: v & ok for k, v in masks.items()}
    P = Prepared(np.nan_to_num(emb), masks["ref"], masks["benign_train"])
    idx = np.where((W.dataset == "sins").values & ok)[0]
    sub = W.iloc[idx]
    y, g = sub.subclass.values, sub.group_id.values
    cal = np.isin(idx, np.where(masks["calib"])[0])
    contexts = [c if c in ("window", "clip", "session") else int(c) for c in a.context]
    chunks = {c: adv.chunk_ids(sub.clip_id.values, c) for c in contexts}
    rows, base = [], []

    def run(F, ctx, **kw):
        Fa, ya, ga = adv.aggregate(F, y, g, chunks[ctx])
        return adv.attack(Fa, ya, ga, n_jobs=a.jobs, **kw)

    for ctx in contexts:  # reference lines: the non-private upper bound and the permuted-label control
        base.append(dict(context=ctx, what="raw embedding (non-private upper bound)", **run(P.unit[idx], ctx, pca=64)))
        base.append(dict(context=ctx, what="control: raw embedding, labels permuted across sessions",
                         **run(P.unit[idx], ctx, pca=64, permute=True)))
    for k, L in parse_grid(a.grid):
        for seed in range(a.seeds):
            v0 = rp.server_view(P, idx, masks["ref"], k, L, seed)
            m = int(calibrate(v0.opened.sum(1)[cal].astype(float), 0.01, True))
            for s_max in a.s_max:
                v = rp.add_synthetic(v0, s_max, a.epoch_windows, np.random.default_rng(seed * 100003 + s_max))
                for mode in a.mode + (["alarm"] if s_max == 0 else []):
                    F = adv.view_features(v, mode, m)
                    for ctx in contexts:
                        rows.append(dict(k=k, L=L, seed=seed, s_max=s_max, mode=mode, context=ctx,
                                         opened_per_window=float(v.opened.sum(1).mean()),
                                         real_opened_per_window=float(v0.opened.sum(1).mean()),
                                         upload_mb_per_day=rp.bytes_per_day(L, s_max) / 1e6, **run(F, ctx)))
                        print(f"  k={k} L={L} seed={seed} s_max={s_max} {mode} {ctx}: bal.acc {rows[-1]['bal_acc']:.1%}",
                              file=sys.stderr, flush=True)
    R = pd.DataFrame(rows)
    chance = R.chance.iloc[0]
    print(f"\n{a.encoder}: {R.n_sessions.iloc[0]} sessions, {R.n_classes.iloc[0]} activities; chance (balanced accuracy) {chance:.1%}")
    print("columns: count_hist/count = what the protocol as implemented shows the server (match or not, per voucher, grouped by window);\n"
          "         open = band order visible, open+codes = matched entry visible: WHAT-IFS for weaker designs, not this protocol.")
    for ctx in contexts:
        print(f"\n=== server pools features over: {ctx}")
        for b in [x for x in base if x["context"] == ctx]:
            print(f"  {b['what']:<62} {b['bal_acc']:>6.1%}   occupancy AUC {b.get('occupancy_auc', float('nan')):.2f}")
        for (k, L), g_ in R[R.context == ctx].groupby(["k", "L"]):
            print(f"  --- k={k} L={L}  (mean over {a.seeds} hash seed(s); balanced accuracy / occupancy AUC)")
            head = f"    {'s_max':>6} {'upload MB/day':>14} {'opened/window':>14} " + "".join(f"{m:>16}" for m in sorted(g_["mode"].unique()))
            print(head)
            for s_max, h in g_.groupby("s_max"):
                cells = []
                for mode in sorted(g_["mode"].unique()):
                    q = h[h["mode"] == mode]
                    cells.append("-".rjust(16) if q.empty else f"{q.bal_acc.mean():>8.1%} /{q.occupancy_auc.mean():>5.2f}")
                print(f"    {s_max:>6} {h.upload_mb_per_day.iloc[0]:>14.0f} {h.opened_per_window.mean():>14.1f} " + "".join(cells))
    (out_dir(a.night, "replay") / f"attack_{a.encoder.replace(':', '_').replace('+', '_')}.json").write_text(
        json.dumps(dict(rows=rows, baselines=base), indent=2, default=str))


# ----------------------------------------------------------------------------- utility
def family(name):
    head, _, tag = name.partition("|")
    return head + "|" + tag.rsplit("s", 1)[0] if tag else head


LABELS = {"A": "A   unmodified: opened vouchers", "B": "B   >= m bands per window", "D": "D   public classifier",
          "COS": "COS cosine top-1 threshold"}


def family_label(fam):
    head, _, tag = fam.partition("|")
    letter = "COS" if head.startswith("COS") else head[0]
    fw = head[len(letter):]
    return LABELS[letter] + (f", window FMR {float(fw):.1%}" if fw else "") + (f"  {tag}" if tag else "")


def cmd_utility(a):
    W, E = load(a.night, parts(a.encoder))
    cells = parse_grid(a.grid)
    for enc in a.encoder:
        emb, modality = embedding(E, enc)
        masks = track_masks(W, "xd", modality)
        ok = np.isfinite(emb[:, 0])
        masks = {k: v & ok for k, v in masks.items()}
        P = Prepared(np.nan_to_num(emb), masks["ref"], masks["benign_train"])
        X, thr = rp.rule_stats(W, P, masks, cells, list(range(a.seeds)), tuple(a.fw))
        res = rp.epoch_replay(W, X, masks, tuple(a.alpha), a.epoch_windows, a.block, B=a.boot)
        fams = {}
        for i, n in enumerate(res["names"]):
            fams.setdefault(family(n), []).append(i)
        wk = res["epochs_per_week"]
        print("=" * 110, f"\n{enc}: epoch = {a.epoch_windows} windows ({a.epoch_windows * rp.HOP_S / 60:.0f} min), {wk:.0f} epochs/week; "
              f"benign {res['n_movies']} movies / {res['n_blocks']} blocks of {a.block} windows; {res['n_events']} events")
        print(f"{'rule':<62}" + "".join(f"{f'false alarms {al * wk:.1f}/week':>34}" for al in res["alphas"]))
        for fam_, ix in fams.items():
            label = family_label(fam_)
            cells_txt = ""
            for ai in range(len(res["alphas"])):
                pt = res["recall"][ai, ix]
                lo, hi = np.quantile(res["boot"][:, ai, ix].mean(1), [0.025, 0.975])
                sd = f" sd {pt.std(ddof=1):.0%}" if len(ix) > 1 else ""
                cells_txt += f"{pt.mean():>7.1%} [{lo:>4.0%},{hi:>4.0%}]{sd}".rjust(34)
            print(f"{label:<62}{cells_txt}")
        (out_dir(a.night, "replay") / f"utility_{enc.replace(':', '_').replace('+', '_')}.json").write_text(json.dumps(
            dict(names=res["names"], alphas=res["alphas"], recall=res["recall"].tolist(),
                 benign_alarm_rate=res["benign_alarm_rate"].tolist(), threshold=res["threshold"].tolist(),
                 window_thresholds=thr, n_movies=res["n_movies"], n_events=res["n_events"],
                 epochs_per_week=wk, boot_ci_note="95% bootstrap over movies and events, per family seed-mean"), indent=2))
        print()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("attack", "utility"):
        p = sp.add_parser(name)
        p.add_argument("--night", required=True)
        p.add_argument("--grid", action="append", help="k,L (repeatable); default 12,32 16,32")
        p.add_argument("--epoch-windows", type=int, default=250, help="window count of an epoch; 250 = 10 min at the 2.4 s hop")
        if name == "attack":
            p.add_argument("--encoder", required=True)
            p.add_argument("--seeds", type=int, default=1)
            p.add_argument("--s-max", type=int, nargs="+", default=[0, 10, 100, 1000, 8000])
            p.add_argument("--mode", nargs="+", default=["count_hist", "count"],
                           help="count_hist = the protocol as implemented; open and open+codes are WHAT-IFS (see mmpsi/adversary.py)")
            p.add_argument("--context", nargs="+", default=["clip", "session"], help="window | clip | session | C (clips)")
            p.add_argument("--jobs", type=int, default=5)
        else:
            p.add_argument("--encoder", action="append", required=True)
            p.add_argument("--seeds", type=int, default=3)
            p.add_argument("--alpha", type=float, nargs="+", default=[1e-2, 1e-3])
            p.add_argument("--fw", type=float, nargs="+", default=[0.01, 0.001], help="window-level FMR for rules B, D, COS")
            p.add_argument("--block", type=int, default=10)
            p.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    {"attack": cmd_attack, "utility": cmd_utility}[a.cmd](a)


if __name__ == "__main__":
    main()
