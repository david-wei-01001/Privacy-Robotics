"""mmpsi.replay and mmpsi.adversary on synthetic worlds with a known structure. Run from the repository root."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmpsi import adversary as adv  # noqa: E402
from mmpsi import replay as rp  # noqa: E402
from mmpsi.evaluate import track_masks  # noqa: E402
from mmpsi.matching import LSHExact, Prepared  # noqa: E402


def _row(**kw):
    base = dict(dataset="xd", label="benign", has_audio=True, split="", subclass="", group_id="", event_id="",
                clip_id="", t0=0.0, t1=4.8, source="movie", video_path="", window_id="")
    base.update(kw)
    return base


def home_world(seed=0, D=32, per_act=12):
    """SINS-like sessions of four activities whose embeddings differ; the reference list resembles two of them, so the
    number and the codes of opened vouchers depend on the activity (a real leak, to be found by the adversary)."""
    rng = np.random.default_rng(seed)
    acts = ["absence", "cooking", "vacuum", "tv"]
    centre = {a: rng.normal(size=D) * 2 for a in acts}
    ref_dir = 0.8 * centre["vacuum"] + 0.2 * centre["tv"]
    rows, E = [], []
    for i in range(60):
        rows.append(_row(label="violent", split="reference", group_id=f"r{i // 5}", event_id=f"re{i // 5}",
                         clip_id=f"rv{i // 5}", t0=2.4 * (i % 5), window_id=f"w{len(rows)}"))
        E.append(ref_dir + 0.5 * rng.normal(size=D))
    s = 0
    for a in acts:
        for _ in range(per_act):
            s += 1
            split = ["benign_train", "val_calib", "val_check"][s % 3]
            for node in (1, 2):
                for c in range(1, 7):
                    for w in range(3):
                        cid = f"DevNode{node}_ex{s}_{c}"
                        rows.append(_row(dataset="sins", split=split, subclass=a, group_id=f"s{s}", event_id=cid,
                                         clip_id=cid, t0=2.4 * w, source="home", window_id=f"w{len(rows)}"))
                        E.append(centre[a] + rng.normal(size=D))
    return pd.DataFrame(rows), np.array(E, np.float32)


def xd_world(seed=0, D=32):
    """Benign movies (long clips; 1% of the windows of 30% of the movies look like the violent class) and violent events."""
    rng = np.random.default_rng(seed)
    v_dir = 5 * rng.normal(size=D) / np.sqrt(D)
    rows, E = [], []

    def add(n, **kw):
        for j in range(n):
            rows.append(_row(clip_id=kw.pop("clip_id") if False else kw["clip_id"], t0=2.4 * j, window_id=f"w{len(rows)}",
                             **{k: v for k, v in kw.items() if k != "clip_id"}))

    for i in range(24):  # references
        for j in range(5):
            rows.append(_row(label="violent", split="reference", group_id=f"r{i}", event_id=f"re{i}", clip_id=f"rv{i}",
                             t0=2.4 * j, window_id=f"w{len(rows)}"))
            E.append(v_dir + 0.5 * rng.normal(size=D))
    for i in range(30):  # test events, ten windows each
        for j in range(10):
            rows.append(_row(label="violent", split="val_violent", group_id=f"v{i}", event_id=f"ve{i}", clip_id=f"vv{i}",
                             t0=2.4 * j, window_id=f"w{len(rows)}"))
            E.append(v_dir * (1 if i % 5 else 0) + 0.5 * rng.normal(size=D))  # one event in five is invisible
    for split, n_movies in (("benign_train", 10), ("val_calib", 20), ("val_check", 20)):
        for g in range(n_movies):
            lookalike = rng.random() < 0.3
            for j in range(60):
                rows.append(_row(split=split, group_id=f"{split}{g}", event_id="", clip_id=f"{split}{g}", t0=2.4 * j,
                                 window_id=f"w{len(rows)}"))
                E.append(v_dir * (rng.random() < 0.01 and lookalike) + 0.5 * rng.normal(size=D))
    return pd.DataFrame(rows), np.array(E, np.float32)


# ----------------------------------------------------------------------------- server view and dummy traffic
@pytest.fixture(scope="module")
def home():
    W, E = home_world()
    m = track_masks(W, "home", "audio")
    P = Prepared(E, m["ref"], m["benign_train"])
    idx = np.where((W.dataset == "sins").values)[0]
    return W, P, m, idx


def test_server_view_equals_the_matcher_score(home):
    W, P, m, idx = home
    v = rp.server_view(P, idx, m["ref"], 8, 12, seed=3)
    assert np.array_equal(v.opened.sum(1), LSHExact(8, 12, 3).fit(P, m["ref"], m["benign_train"]).score(P, idx))
    assert (v.codes[~v.opened] == -1).all() and (v.codes[v.opened] >= 0).all()


def test_synthetic_matches_add_opens_without_touching_real_ones(home):
    W, P, m, idx = home
    v = rp.server_view(P, idx, m["ref"], 8, 12)
    noisy = rp.add_synthetic(v, s_max=1000, epoch_windows=250, rng=np.random.default_rng(0))
    assert (noisy.opened[v.opened]).all() and np.array_equal(noisy.codes[v.opened], v.codes[v.opened])
    added = (noisy.opened.sum(1) - v.opened.sum(1)).mean()
    expected = 1000 / 250 * (1 - v.opened.mean())          # s_max / epoch_windows per window, over closed vouchers
    assert added == pytest.approx(expected, rel=0.15)
    new = noisy.opened & ~v.opened
    assert all(np.isin(noisy.codes[:, b][new[:, b]], v.tables[b]).all() for b in range(12))  # random table entries
    assert rp.add_synthetic(v, 0, 250, np.random.default_rng(0)) is v


def test_voucher_bytes_match_the_prototype_and_cost_scales_with_the_budget():
    assert [rp.voucher_bytes(s) for s in (0, 10, 100, 1000)] == [332, 412, 1132, 8332]
    assert rp.bytes_per_day(32, 10) == pytest.approx(36000 * 32 * 412)
    assert rp.bytes_per_day(32, 1000) / rp.bytes_per_day(32, 10) == pytest.approx(8332 / 412)


def test_context_chunks(home):
    W, P, m, idx = home
    clip = W.clip_id.values[idx]
    n_clip, n_unit = len(set(clip)), len({c.rsplit("_", 1)[0] for c in clip})
    assert len(set(adv.chunk_ids(clip, "window"))) == len(clip)
    assert len(set(adv.chunk_ids(clip, "clip"))) == n_clip
    assert len(set(adv.chunk_ids(clip, "session"))) == n_unit
    assert len(set(adv.chunk_ids(clip, 3))) == 2 * n_unit  # six clips per device and session
    with pytest.raises(ValueError):
        adv.chunk_ids(np.array(["rv1", "rv2"]), "session")


# ----------------------------------------------------------------------------- the adversary
def _attack(home, s_max, mode="open+codes", context="clip", **kw):
    W, P, m, idx = home
    v = rp.add_synthetic(rp.server_view(P, idx, m["ref"], 8, 16), s_max, 250, np.random.default_rng(1))
    F = adv.view_features(v, mode, m=3)
    sub = W.iloc[idx]
    F, y, g = adv.aggregate(F, sub.subclass.values, sub.group_id.values, adv.chunk_ids(sub.clip_id.values, context))
    return adv.attack(F, y, g, **kw)


def test_the_adversary_finds_the_activity_in_an_unprotected_stream(home):
    r = _attack(home, 0)
    assert r["chance"] == 0.25 and r["bal_acc"] > 0.5 and r["occupancy_auc"] > 0.7
    assert r["n_sessions"] == 48


def test_the_control_with_permuted_labels_is_at_chance(home):
    r = _attack(home, 0, permute=True)
    assert abs(r["bal_acc"] - 0.25) < 0.15


def test_dummy_traffic_lowers_what_the_adversary_learns(home):
    clean, noisy = _attack(home, 0)["bal_acc"], _attack(home, 8000)["bal_acc"]
    assert noisy < clean


def test_the_decision_bit_alone_carries_far_less_than_the_opened_vouchers(home):
    assert _attack(home, 0, mode="alarm")["bal_acc"] < _attack(home, 0, mode="open+codes")["bal_acc"]


def test_longer_context_is_a_stronger_adversary(home):
    assert _attack(home, 0, mode="count", context="session")["bal_acc"] >= _attack(home, 0, mode="count", context="clip")["bal_acc"] - 0.05


# ----------------------------------------------------------------------------- epoch replay
@pytest.fixture(scope="module")
def replay_world():
    W, E = xd_world()
    m = track_masks(W, "xd", "video")
    P = Prepared(E, m["ref"], m["benign_train"])
    X, thr = rp.rule_stats(W, P, m, cells=[(10, 16)], seeds=[0, 1], f_ws=(0.01,))
    return W, m, X, thr


def test_rule_statistics_exist_for_every_rule(replay_world):
    W, m, X, thr = replay_world
    assert {"A|k10L16s0", "A|k10L16s1", "B0.01|k10L16s0", "D0.01", "COS0.01"} <= set(X)
    assert set(np.unique(X["D0.01"][np.isfinite(X["D0.01"])])) <= {0.0, 1.0}


def test_epoch_replay_is_calibrated_and_detects_visible_events(replay_world):
    W, m, X, thr = replay_world
    res = rp.epoch_replay(W, X, m, alphas=(1e-2,), epoch_windows=100, block=10, n_benign=1500, per_event=10, B=20)
    assert res["n_events"] == 30 and res["n_movies"] == 40 and res["epochs_per_week"] == pytest.approx(7 * 86400 / (100 * 2.4))
    r = dict(zip(res["names"], res["recall"][0]))
    far = dict(zip(res["names"], res["benign_alarm_rate"][0]))
    assert all(v <= 0.01 + 1e-9 for v in far.values())
    assert r["D0.01"] > 0.6 and r["A|k10L16s0"] > 0.3          # the classifier sees the 4 in 5 visible events
    assert r["D0.01"] <= 1.0 and res["boot"].shape == (20, 1, len(res["names"]))
    assert 0 <= np.quantile(res["boot"][:, 0, res["names"].index("D0.01")], 0.025) <= r["D0.01"] + 0.2


def test_a_stricter_false_alarm_budget_costs_recall(replay_world):
    W, m, X, thr = replay_world
    res = rp.epoch_replay(W, X, m, alphas=(1e-1, 1e-3), epoch_windows=100, block=10, n_benign=1500, per_event=10, B=2)
    i = res["names"].index("D0.01")
    assert res["recall"][0, i] >= res["recall"][1, i]


def test_the_replay_is_reproducible(replay_world):
    W, m, X, thr = replay_world
    a = rp.epoch_replay(W, X, m, alphas=(1e-2,), epoch_windows=100, n_benign=500, per_event=5, B=3, seed=4)
    b = rp.epoch_replay(W, X, m, alphas=(1e-2,), epoch_windows=100, n_benign=500, per_event=5, B=3, seed=4)
    assert np.array_equal(a["recall"], b["recall"]) and np.array_equal(a["boot"], b["boot"])


def test_the_attack_does_not_crash_when_the_features_are_fewer_than_the_pca_components():
    rng = np.random.default_rng(0)
    F, y, g = rng.normal(size=(200, 12)), np.repeat(["a", "b"], 100), np.repeat(np.arange(20), 10)
    r = adv.attack(F, y, g, pca=64, n_splits=4)
    assert 0 <= r["bal_acc"] <= 1 and r["chance"] == 0.5


def test_count_histogram_is_a_distribution_over_the_pooled_windows(home):
    W, P, m, idx = home
    v = rp.server_view(P, idx, m["ref"], 8, 16)
    F = adv.view_features(v, "count_hist")
    assert F.shape == (len(idx), len(adv.COUNT_EDGES) + 1) and (F.sum(1) == 1).all()
    sub = W.iloc[idx]
    Fa, ya, ga = adv.aggregate(F, sub.subclass.values, sub.group_id.values, adv.chunk_ids(sub.clip_id.values, "clip"))
    assert np.allclose(Fa.sum(1), 1)
    cnt = v.opened.sum(1)
    bins = np.digitize(cnt, adv.COUNT_EDGES)
    assert (bins[cnt == 0] == 0).all() and (bins[cnt == 4] == 4).all() and (bins[cnt == 5] == 4).all()


def test_the_protocol_view_is_the_count_and_the_histogram_attack_finds_the_activity(home):
    r = _attack(home, 0, mode="count_hist")
    assert r["bal_acc"] > 0.4 and r["occupancy_auc"] > 0.7
