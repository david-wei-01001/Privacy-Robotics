#!/usr/bin/env python
"""Audio/video reconciliation on the cached embeddings of a finished night (PLAN §6.4). No encoding, no GPU: seconds.

    .venv/bin/python tools/fusion_eval.py --night 2026-10-29_n2 --audio ast --video videomae_k400
    .venv/bin/python tools/fusion_eval.py --night 2026-10-29_n2 --audio ast --video xclip_b32 \
        --matcher '{"name": "cosine_top1"}' --matcher '{"name": "lsh", "k": 20, "L": 16, "seed": 1}'

Prints, per matcher: audio only, video only, early fusion (concatenated embedding) and the late rules max / sum / min,
all calibrated on val_calib at the same FMR targets and measured on val_check; the paired recall differences on the
same events; and the audio/video agreement table. Results are also written to runs/<night>/fusion/*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmpsi import features, features_video, matching  # noqa: E402,F401  (register encoders and matchers)
from mmpsi.core import REPO, build, data_root, short_hash  # noqa: E402
from mmpsi.evaluate import track_masks  # noqa: E402
from mmpsi.fusion import format_result, run  # noqa: E402

DEFAULT_MATCHERS = ['{"name": "cosine_top1"}', '{"name": "lsh", "k": 16, "L": 16, "seed": 0}']


def load(night: str, names: list[str]):
    stages = json.loads((REPO / "runs" / night / "stages.json").read_text())
    facts_path = REPO / "reports" / night / "facts.json"
    if facts_path.exists():
        frac = json.loads(facts_path.read_text()).get("scale", {}).get("benign_fraction_kept", 1.0)
        if frac != 1.0:
            sys.exit(f"night {night} dropped benign windows to fit its time budget (kept {frac}); the cached embeddings "
                     f"belong to a window table that is not on disk, so this tool cannot rebuild it")
    W = pd.read_csv(stages["data"]["file"], keep_default_na=False)
    whash = short_hash(W.window_id.tolist())
    E = {}
    for n in names:
        p = data_root() / "cache/emb" / f"{whash}_{build('encoder', n).fingerprint()}.npy"
        if not p.exists():
            sys.exit(f"no cached embedding for {n}: {p}\n(run the pilot for this night with {n} in `encoders` first)")
        E[n] = np.load(p)
    return W, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--matcher", action="append", help="JSON matcher spec; may be repeated")
    a = ap.parse_args()
    W, E = load(a.night, [a.audio, a.video])
    masks = track_masks(W, "xd", "video")  # XD only: SINS has no video
    out_dir = REPO / "runs" / a.night / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec_s in a.matcher or DEFAULT_MATCHERS:
        spec = json.loads(spec_s)
        res = run(W, E[a.audio], E[a.video], masks, spec)
        print("=" * 100, f"\n{a.audio} + {a.video}\n", format_result(res), "\n")
        tag = "_".join(f"{k}{v}" for k, v in spec.items()).replace(" ", "")
        (out_dir / f"{a.audio}+{a.video}__{tag}.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
