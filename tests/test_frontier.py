"""mmpsi.frontier: matchers at one measured FMR plus the leakage proxy. Run from the repository root."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmpsi.evaluate import track_masks  # noqa: E402
from mmpsi.frontier import format_frontier, frontier  # noqa: E402
from test_video_fusion import synth  # noqa: E402


@pytest.fixture(scope="module")
def res():
    W, EA, _ = synth(0)
    return frontier(W, EA, track_masks(W, "xd", "audio"), grid=((6, 8), (10, 8)), seeds=(0, 1), fmrs=(0.01,), B=60)


def test_every_matcher_and_seed_is_reported(res):
    rows = {r["method"] for r in res["matched"][0]["rows"]}
    assert {"cosine_top1", "cosine_top5", "logreg"} <= rows
    assert {f"lsh_k{k}_L8_s{s}" for k in (6, 10) for s in (0, 1)} <= rows
    assert res["matched"][0]["n_benign_videos"] == 50


def test_leakage_proxy_is_consistent_with_the_score(res):
    for v in res["leak"].values():
        assert v["bits_per_benign_window"] == pytest.approx(v["k"] * v["opened_benign"])
        assert 0 <= v["p_any_open"] <= 1 and 0 <= v["opened_benign"] <= v["L"]


def test_fewer_bits_per_band_opens_more_vouchers(res):
    lo = np.mean([v["opened_benign"] for v in res["leak"].values() if v["k"] == 6])
    hi = np.mean([v["opened_benign"] for v in res["leak"].values() if v["k"] == 10])
    assert lo > hi  # a shorter code collides with the reference table more often


def test_a_classifier_beats_reference_matching_on_this_easy_case(res):
    rec = {r["method"]: r["recall"] for r in res["matched"][0]["rows"]}
    assert rec["logreg"] >= max(v for k, v in rec.items() if k.startswith("lsh")) - 0.05
    assert any(p["versus"] == "cosine_top1" for p in res["matched"][0]["paired"])


def test_report_names_the_best_seed_as_the_conservative_comparison(res):
    txt = format_frontier(res, "synthetic")
    assert "MATCHED FMR 1.00%" in txt and "opens" in txt and "best LSH seed" in txt
