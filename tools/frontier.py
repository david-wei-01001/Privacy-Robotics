#!/usr/bin/env python
"""Matchers at the same measured FMR, plus a leakage proxy, on the cached embeddings of a finished night. No encoding.

    .venv/bin/python tools/frontier.py --night 2026-10-29_n2 --encoder videomae_k400 --encoder xclip_b32 --encoder clip_frames
    .venv/bin/python tools/frontier.py --night 2026-10-29_n2 --encoder ast --grid 12,32 --grid 8,64 --seeds 8

`--encoder fusion:ast+clip_frames` concatenates the L2-normalised embeddings, as the pilot does. Writes
runs/<night>/frontier/<encoder>.json. See mmpsi/frontier.py for what is compared and what the leakage proxy means.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_eval import load  # noqa: E402
from mmpsi.core import REPO, build  # noqa: E402
from mmpsi.evaluate import track_masks  # noqa: E402
from mmpsi.frontier import DEFAULT_GRID, format_frontier, frontier  # noqa: E402
from mmpsi.matching import l2n  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", required=True)
    ap.add_argument("--encoder", action="append", required=True)
    ap.add_argument("--track", default="xd", choices=["xd", "home"])
    ap.add_argument("--grid", action="append", help="k,L (repeatable); default 12,32 16,16 16,32 24,16")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    grid = tuple(tuple(int(x) for x in g.split(",")) for g in a.grid) if a.grid else DEFAULT_GRID
    parts = sorted({p for e in a.encoder for p in (e[len("fusion:"):].split("+") if e.startswith("fusion:") else [e])})
    W, E = load(a.night, parts)
    out_dir = REPO / "runs" / a.night / "frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.encoder:
        if name.startswith("fusion:"):
            x, y = name[len("fusion:"):].split("+")
            emb = np.concatenate([l2n(np.nan_to_num(E[x])), l2n(np.nan_to_num(E[y]))], 1)
            emb[~(np.isfinite(E[x][:, 0]) & np.isfinite(E[y][:, 0]))] = np.nan
            modality = "video"
        else:
            emb, modality = E[name], build("encoder", name).modality
        masks = track_masks(W, a.track, modality)
        res = frontier(W, emb, masks, grid=grid, seeds=tuple(range(a.seeds)), B=a.boot)
        print("=" * 100, "\n" + format_frontier(res, name), "\n")
        (out_dir / (name.replace(":", "_").replace("+", "_") + ".json")).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
