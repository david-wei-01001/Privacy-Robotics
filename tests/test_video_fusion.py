"""Video encoders and audio/video fusion. Run from the repository root: python -m pytest tests -q

The video fixture is synthetic: the grey level of a frame encodes its time (0 -> 255 over the clip), and after 10 s a
square moves. The neural encoders are not exercised here (they need weights); BlockVideoEncoder carries all the logic
they share, and a stub subclass tests it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmpsi import features_video as fv  # noqa: E402
from mmpsi.evaluate import track_masks  # noqa: E402
from mmpsi.fusion import RULES, format_result, recall_at_fmr, run, tail_z  # noqa: E402

SECONDS, FPS = 20, 25


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    av = pytest.importorskip("av")
    path = str(tmp_path_factory.mktemp("vid") / "t.mp4")
    c = av.open(path, "w")
    st = c.add_stream("mpeg4", rate=FPS)
    st.width, st.height, st.pix_fmt = 160, 120, "yuv420p"
    for i in range(SECONDS * FPS):
        t = i / FPS
        img = np.full((120, 160, 3), int(255 * t / SECONDS), np.uint8)
        if t >= 10:
            x = int((t * 60) % 140)
            img[40:60, x:x + 20] = 255 if img[0, 0, 0] < 128 else 0
        for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()
    return path


def windows(path, n=7, w=4.8, hop=2.4):
    t0 = np.round(np.arange(n) * hop, 3)
    return pd.DataFrame(dict(video_path=path, t0=t0, t1=np.round(t0 + w, 3)))


class GreyStub(fv.BlockVideoEncoder):
    """The base-class logic with a trivial model: the embedding of a block is its mean grey level."""
    dim, frames, batch = 1, 4, 3
    registry_name = "grey_stub"

    def embed_blocks(self, clips):
        return clips.mean((1, 2, 3, 4))[:, None].astype(np.float32)


# ----------------------------------------------------------------------------- decoding and windows
def test_block_times_are_sub_interval_centres():
    t = fv.block_times([0, 2], 2.4, 4)
    assert np.allclose(t[0], [0.3, 0.9, 1.5, 2.1]) and np.allclose(t[1], 4.8 + np.array([0.3, 0.9, 1.5, 2.1]))


def test_decoded_frames_follow_the_requested_times(video):
    clips = fv.decode_blocks(video, [0, 1, 2, 3, 4], 2.4, 4)
    assert clips.shape == (5, 4, 224, 224, 3) and clips.dtype == np.uint8
    expected = 255 * fv.block_times([0, 1, 2, 3, 4], 2.4, 4) / SECONDS
    got = clips[:, :, :20, :20].mean((2, 3, 4))  # a corner the square never reaches
    assert np.abs(got - expected).max() < 8


def test_seek_path_agrees_with_sequential_decode(video):
    seq = fv.decode_blocks(video, [0, 1, 2, 3, 4], 2.4, 4)[3:]
    seeked = fv.decode_blocks(video, [3, 4], 2.4, 4)  # first target > 2 s, so the container is seeked
    assert np.abs(seq.astype(float) - seeked.astype(float)).mean() < 2


def test_targets_past_the_end_reuse_the_last_frame(video):
    clips = fv.decode_blocks(video, [8], 2.4, 4)  # block 8 covers 19.2 - 21.6 s of a 20 s clip
    assert clips.shape[0] == 1 and np.isfinite(clips).all()


def test_window_is_the_mean_of_its_blocks(video):
    enc, df = GreyStub(), windows(video, n=3)
    out = enc.encode(df)
    per_block = fv.decode_blocks(video, [0, 1, 2, 3], 2.4, 4).mean((1, 2, 3, 4))
    for i in range(3):
        assert out[i, 0] == pytest.approx(per_block[i:i + 2].mean(), rel=1e-4)
    assert enc.coverage == 1.0


def test_a_misaligned_window_is_an_error_not_a_skip(video):
    df = windows(video, n=1)
    df.loc[0, ["t0", "t1"]] = [1.0, 5.8]
    with pytest.raises(ValueError, match="whole blocks"):
        GreyStub().encode(df)


def test_an_unreadable_video_yields_nan_and_does_not_stop_the_rest(video, tmp_path):
    df = pd.concat([windows(str(tmp_path / "missing.mp4"), n=2), windows(video, n=2)], ignore_index=True)
    out = GreyStub().encode(df)
    assert np.isnan(out[:2]).all() and np.isfinite(out[2:]).all()


def test_budget_overrun_marks_coverage(video):
    enc = GreyStub()
    out = enc.encode(windows(video, n=3), budget_s=1e-9)
    assert enc.coverage < 1.0 and np.isnan(out).all()


def test_motion_stats_sees_motion_and_ignores_appearance(video):
    enc = fv.MotionStats()
    out = enc.encode(windows(video, n=7))
    assert out.shape == (7, 35) and np.isfinite(out).all()
    still, moving = out[0], out[5]  # 0 - 4.8 s vs 12 - 16.8 s
    assert moving[:16].mean() > 2 * still[:16].mean()
    assert enc.fingerprint() == "motion_stats-b2.4-t16-v1" and not enc.neural


def test_registry_has_the_three_new_encoders():
    from mmpsi.core import available
    assert {"motion_stats", "videomae_k400", "xclip_b32"} <= set(available("encoder"))


# ----------------------------------------------------------------------------- fusion
def test_tail_z_is_monotone_and_finite():
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    z = tail_z(s, np.ones(500, bool))
    assert np.isfinite(z).all() and z.min() >= 0 and (np.diff(z[np.argsort(s)]) >= 0).all()
    assert z[np.argmax(s)] == pytest.approx(np.log(501 / 2))  # in sample the maximum counts itself: P = 2/(n+1)
    big = np.r_[s, s.max() + 1]
    assert tail_z(big, np.r_[np.ones(500, bool), False])[-1] == pytest.approx(np.log(501))  # above every benign: 1/(n+1)
    assert np.isnan(tail_z(np.array([np.nan, 1.0]), np.r_[True, True])[0])


def synth(seed=0, D=24, contaminated=0.003):
    """Benign windows with modality-specific false matches (a 'TV' that only the microphone hears, a 'sports field'
    that only the camera sees), drawn per window, and violent events that carry the signal in one modality or both."""
    rng = np.random.default_rng(seed)
    a_dir, v_dir = rng.normal(size=D), rng.normal(size=D)
    a_dir, v_dir = 6 * a_dir / np.linalg.norm(a_dir), 6 * v_dir / np.linalg.norm(v_dir)
    rows, EA, EV = [], [], []

    def add(split, label, n, group, event="", ca=lambda: False, cv=lambda: False):
        for _ in range(n):
            rows.append(dict(dataset="xd", label=label, has_audio=True, split=split, subclass="", group_id=group,
                             event_id=event, source="movie", video_path="", window_id=f"w{len(rows)}"))
            EA.append(rng.normal(size=D) * 0.5 + (a_dir if ca() else 0))
            EV.append(rng.normal(size=D) * 0.5 + (v_dir if cv() else 0))

    yes = lambda: True  # noqa: E731
    for e in range(12):  # references: both modalities informative
        add("reference", "violent", 5, f"r{e}", f"re{e}", yes, yes)
    for e in range(60):  # test events: 40% carry both, 30% only the audio, 30% only the video
        kind = e % 10
        add("val_violent", "violent", 5, f"v{e}", f"ve{e}", (yes if kind < 7 else (lambda: False)),
            (yes if (kind < 4 or kind >= 7) else (lambda: False)))
    for split, n in (("benign_train", 800), ("val_calib", 2500), ("val_check", 2500)):
        for g in range(25):
            add(split, "benign", n // 25, f"{split}{g}", "", lambda: rng.random() < contaminated,
                lambda: rng.random() < contaminated)
    W = pd.DataFrame(rows)
    return W, np.array(EA, np.float32), np.array(EV, np.float32)


@pytest.mark.parametrize("spec", [{"name": "cosine_top1"}, {"name": "lsh", "k": 8, "L": 8, "seed": 0}])
def test_fusion_run_is_calibrated_and_complementary_modalities_help(spec):
    W, EA, EV = synth()
    first_benign = int(np.where(W.split == "benign_train")[0][0])
    EV[first_benign:first_benign + 5] = np.nan  # windows without a video embedding are left out of both modalities
    res = run(W, EA, EV, track_masks(W, "xd", "video"), spec, fmrs=(0.01,), B=100)
    by = {r["method"]: r for r in res["deployed"]["rows"]}
    assert set(by) == {"audio", "video", "early", "late_max", "late_sum", "late_min"}
    for r in by.values():
        assert r["fmr_calib"] <= 0.01 + 1e-9 and r["n_events"] == 60 and r["n_fm_videos"] <= r["n_check_videos"]
    mm = {r["method"]: r["recall"] for r in res["matched"][0]["rows"]}
    # events carried by only one modality: OR beats either single modality; AND is below OR
    assert mm["late_max"] > max(mm["audio"], mm["video"])
    assert mm["late_min"] < mm["late_max"]
    ag = res["agreement"][0]
    assert sum(ag["events"].values()) == 60 and ag["events"]["audio_only"] > 0 and ag["events"]["video_only"] > 0
    assert res["windows"]["ref"] == 60  # references keep their windows; only the 5 NaN windows (benign_train) are dropped
    assert "MATCHED FMR" in format_result(res) and "DEPLOYED" in format_result(res)


def test_matched_paired_difference_excludes_zero_for_a_real_gain():
    W, EA, EV = synth()
    res = run(W, EA, EV, track_masks(W, "xd", "video"), {"name": "cosine_top1"}, fmrs=(0.01,), B=200)
    p = {(x["method"], x["versus"]): x for x in res["matched"][0]["paired"]}
    d = p[("late_max", "audio")]
    assert d["diff"] > 0 and d["diff_ci"][0] > 0


def test_recall_at_fmr_is_exact_with_ties():
    b = np.r_[np.zeros(90), np.ones(10)]  # 10% of benign windows tie at the top score
    e = np.array([1.0, 1.0, 0.0])         # two of three events reach the top score
    # FMR 5%: half of the lenient rule (FMR 10%, recall 2/3) and half of the strict rule (FMR 0%, recall 0)
    assert recall_at_fmr(b, e, 0.05) == pytest.approx(1 / 3)
    assert recall_at_fmr(b, e, 0.10) == pytest.approx(2 / 3)
    assert recall_at_fmr(b, e, 0.0) == pytest.approx(0.0)
    c = np.linspace(0, 1, 1000)  # no ties: the plain quantile
    assert recall_at_fmr(c, np.array([0.995, 0.5]), 0.01) == pytest.approx(0.5)


def test_matched_fmr_removes_the_advantage_of_spending_more_false_matches():
    """A discrete matcher calibrated to a target it cannot hit measures a lower FMR than a finer-grained fusion rule;
    at the same MEASURED FMR the two are compared fairly."""
    W, EA, EV = synth()
    res = run(W, EA, EV, track_masks(W, "xd", "video"), {"name": "lsh", "k": 8, "L": 8, "seed": 0}, fmrs=(0.01,), B=50)
    chk = {r["method"]: r["fmr_check"] for r in res["deployed"]["rows"]}
    assert len(set(round(v, 4) for v in chk.values())) > 1  # the deployed rows differ in measured FMR ...
    assert res["matched"][0]["n_benign_videos"] == 50       # ... the matched comparison pools all benign val videos


def test_threshold_setters_report_how_concentrated_the_calibration_is():
    W, EA, EV = synth()
    res = run(W, EA, EV, track_masks(W, "xd", "video"), {"name": "cosine_top1"}, fmrs=(0.01,), B=20)
    d = res["setters"]["audio"]
    assert d["top_windows"] == 25 and 0 < d["share_of_top_videos"] <= 1 and d["videos_supplying"] <= d["calib_videos"]


def test_rules_are_the_documented_ones():
    a, b = np.array([1.0, 5.0]), np.array([4.0, 2.0])
    assert RULES["max"](a, b).tolist() == [4, 5] and RULES["min"](a, b).tolist() == [1, 2] and RULES["sum"](a, b).tolist() == [5, 7]
