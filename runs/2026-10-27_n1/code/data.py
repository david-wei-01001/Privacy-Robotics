"""Dataset module (PLAN §4): XD-Violence, SINS (DCASE 2018 Task 5), TV mix; windows and group splits."""
from __future__ import annotations

import hashlib
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
XD_CODES = {"A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot", "B5": "abuse",
            "B6": "car_accident", "G": "explosion"}
HF_REPO = "jherng/xd-violence"


def _h(key: str, salt: str = "") -> float:
    """Deterministic number in [0, 1) from a key (split assignment by group)."""
    return int(hashlib.sha1((salt + key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign(key: str, parts: list[tuple[str, float]], salt: str) -> str:
    u, acc = _h(key, salt), 0.0
    for name, p in parts:
        acc += p
        if u < acc:
            return name
    return parts[-1][0]


# ============================================================================ XD-Violence
def parse_xd_name(stem: str):
    group = stem.split("__#")[0]
    source = "youtube" if group.startswith("v=") else "movie"
    codes = [c for c in stem.split("_label_")[-1].split("-") if c not in ("0", "")]
    return group, source, [XD_CODES.get(c, c) for c in codes]


def read_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            nums = list(map(int, parts[1:]))
            name = parts[0][:-4] if parts[0].endswith(".mp4") else parts[0]  # 14 official lines keep the suffix
            out[name] = list(zip(nums[0::2], nums[1::2]))
    return out


def is_locked_test(group: str, frac: float) -> bool:
    return _h(group, "locked-test") < frac


def find_local_videos(raw: Path, log) -> dict[str, Path]:
    """Videos under raw/xd_violence. Zips are extracted only when no video is found, and never those in official/
    (the official feature archives; i3d-features.zip alone is 41 GB)."""
    vids = {p.stem: p for ext in ("*.mp4", "*.avi", "*.mkv") for p in raw.rglob(ext)}
    if not vids:
        zips = [z for z in raw.rglob("*.zip") if "official" not in z.relative_to(raw).parts and zipfile.is_zipfile(z)]
        for z in zips:
            log.info(f"extracting {z.name}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw / "extracted")
        vids = {p.stem: p for ext in ("*.mp4", "*.avi", "*.mkv") for p in raw.rglob(ext)}
    return vids


def hf_test_stems() -> list[str]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF_REPO, "data/test_list.txt", repo_type="dataset")
    return [Path(l.strip()).name for l in Path(p).read_text().splitlines() if l.strip()]


def hf_download_videos(stems: list[str], dest: Path, log) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)

    def one(stem):
        try:
            p = hf_hub_download(HF_REPO, f"data/video/test_videos/{stem}.mp4", repo_type="dataset", local_dir=dest)
            return stem, Path(p)
        except Exception as e:  # noqa: BLE001
            log.warning(f"HF download failed for {stem}: {e}")
            return stem, None

    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(one, stems))
    return {k: v for k, v in res.items() if v is not None}


def probe_video(path: Path):
    import av
    with av.open(str(path)) as c:
        fps = float(c.streams.video[0].average_rate) if c.streams.video else 24.0
        dur = float(c.duration) / 1e6 if c.duration else None
        has_audio = bool(c.streams.audio)
    return fps, dur, has_audio


def extract_audio(video: Path, out_wav: Path) -> float | None:
    """Decode the first audio stream to 16 kHz mono float32 WAV with PyAV. Returns duration or None."""
    import av
    if out_wav.exists():
        return sf.info(str(out_wav)).duration
    with av.open(str(video)) as c:
        if not c.streams.audio:
            return None
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in c.decode(c.streams.audio[0]):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    y = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), y, SR)
    return len(y) / SR


def windows_for_clip(duration: float, w: float, hop: float):
    t = 0.0
    while t + w <= duration + 1e-6:
        yield round(t, 3), round(t + w, 3)
        t += hop


def build_xd(cfg: dict, root: Path, log) -> pd.DataFrame:
    raw, der = root / "raw/xd_violence", root / "derived/xd_violence"
    ann = read_annotations(raw / "annotations.txt")
    local = find_local_videos(raw, log)
    stems = sorted(local) if local else hf_test_stems()
    log.info(f"XD: {len(local)} local videos; {len(stems)} candidate test stems")
    lock_frac = cfg["xd"]["locked_test_frac"]
    themes = set(cfg["xd"]["themes"])
    violent, normal = [], []
    for s in stems:
        group, source, classes = parse_xd_name(s)
        if is_locked_test(group, lock_frac):
            continue
        if s in ann and themes & set(classes):
            violent.append(s)
        elif classes == ["normal"]:
            normal.append(s)
    violent = sorted(violent, key=lambda s: _h(s, "pick"))[: cfg["xd"]["max_violent"]]
    normal = sorted(normal, key=lambda s: _h(s, "pick"))[: cfg["xd"]["max_normal"]]
    need = violent + normal
    missing = [s for s in need if s not in local]
    if missing:
        log.info(f"XD: downloading {len(missing)} test videos from the Hugging Face mirror")
        local.update(hf_download_videos(missing, raw / "hf_mirror", log))
    rows = []
    w, hop = cfg["window_s"], cfg["hop_s"]
    for s in need:
        if s not in local:
            continue
        vpath = local[s]
        try:
            fps, vdur, has_audio = probe_video(vpath)
            adur = extract_audio(vpath, der / "audio16k" / f"{s}.wav") if has_audio else None
        except Exception as e:  # noqa: BLE001
            log.warning(f"XD: cannot read {s}: {e}")
            continue
        dur = adur or vdur
        if not dur:
            continue
        group, source, classes = parse_xd_name(s)
        segs = [(a / fps, b / fps) for a, b in ann.get(s, [])]
        is_violent = s in violent
        split = (assign(group, [("reference", 0.5), ("val_violent", 0.5)], "xd-v") if is_violent
                 else assign(group, [("benign_train", 0.3), ("val_calib", 0.35), ("val_check", 0.35)], "xd-n"))
        for t0, t1 in windows_for_clip(dur, w, hop):
            if is_violent:
                ov = [max(0.0, min(t1, b) - max(t0, a)) for a, b in segs]
                k = int(np.argmax(ov)) if ov else -1
                if k < 0 or ov[k] < 0.5 * w:
                    continue  # unannotated part of a violent video: not used
                label, event = "violent", f"{s}#{k}"
            else:
                label, event = "benign", f"{s}"
            rows.append(dict(dataset="xd", clip_id=s, event_id=event, group_id=group, source=source,
                             t0=t0, t1=t1, label=label, subclass="+".join(classes), split=split,
                             audio_path=str(der / "audio16k" / f"{s}.wav") if adur else "",
                             video_path=str(vpath), has_audio=bool(adur)))
    df = pd.DataFrame(rows)
    log.info(f"XD: {df.clip_id.nunique() if len(df) else 0} videos, {len(df)} windows")
    return df


# ============================================================================ SINS
def build_sins(cfg: dict, root: Path, log) -> pd.DataFrame:
    raw, der = root / "raw/sins", root / "derived/sins/audio16k"
    der.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw / "DCASE2018-task5-dev.meta.zip") as z:
        meta = z.read("DCASE2018-task5-dev/meta.txt").decode().splitlines()
    lab = {}
    for line in meta:
        f, label, session = line.split("\t")[:3]
        lab[Path(f).stem] = (label, session)
    done = {p.stem for p in der.glob("*.wav")}
    for zp in sorted(raw.glob("DCASE2018-task5-dev.audio.*.zip")):
        if not zipfile.is_zipfile(zp):
            log.warning(f"SINS: {zp.name} is incomplete; skipped")
            continue
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                if not m.endswith(".wav") or Path(m).stem in done:
                    continue
                y, sr = sf.read(io.BytesIO(z.read(m)), dtype="float32")
                y = y[:, 0] if y.ndim > 1 else y
                if sr != SR:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                sf.write(str(der / f"{Path(m).stem}.wav"), y, SR)
    files = sorted(der.glob("*.wav"))
    files = sorted(files, key=lambda p: _h(p.stem, "sins-pick"))[: cfg["sins"]["max_files"]]
    rows = []
    w, hop = cfg["window_s"], cfg["hop_s"]
    for p in files:
        if p.stem not in lab:
            continue
        label, session = lab[p.stem]
        dur = sf.info(str(p)).duration
        split = assign(session, [("benign_train", 0.3), ("val_calib", 0.35), ("val_check", 0.35)], "sins")
        for t0, t1 in windows_for_clip(dur, w, hop):
            rows.append(dict(dataset="sins", clip_id=p.stem, event_id=p.stem, group_id=session, source="home",
                             t0=t0, t1=t1, label="benign", subclass=label, split=split, audio_path=str(p),
                             video_path="", has_audio=True))
    df = pd.DataFrame(rows)
    log.info(f"SINS: {len(files)} files, {len(df)} windows")
    return df


# ============================================================================ TV mix (hard negatives)
def load_segment(path: str, t0: float, t1: float) -> np.ndarray:
    a, b = int(t0 * SR), int(t1 * SR)
    y, _ = sf.read(path, start=a, stop=b, dtype="float32")
    if len(y) < b - a:
        y = np.pad(y, (0, b - a - len(y)))
    return y


def build_tv_mix(cfg: dict, root: Path, windows: pd.DataFrame, log) -> pd.DataFrame:
    out_dir = root / "derived/tv_mix"
    out_dir.mkdir(parents=True, exist_ok=True)
    tv = windows[(windows.dataset == "xd") & (windows.split == "val_violent") & windows.has_audio]
    room = windows[(windows.dataset == "sins") & (windows.split == "val_check") & (windows.subclass != "watching_tv")]
    n = min(cfg["tv_mix"]["n"], len(tv), len(room))
    if n == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(0)
    tv_s, room_s = tv.sample(n, random_state=1), room.sample(n, random_state=2)
    rows = []
    for i, ((_, a), (_, b)) in enumerate(zip(tv_s.iterrows(), room_s.iterrows())):
        snr = float(rng.choice(cfg["tv_mix"]["snr_db"]))
        ya, yb = load_segment(a.audio_path, a.t0, a.t1), load_segment(b.audio_path, b.t0, b.t1)
        ra, rb = np.sqrt(np.mean(ya ** 2)) + 1e-8, np.sqrt(np.mean(yb ** 2)) + 1e-8
        y = yb + ya * (rb / ra) * 10 ** (snr / 20)
        y = y / max(1.0, np.abs(y).max())
        p = out_dir / f"tvmix_{i:04d}.wav"
        sf.write(str(p), y.astype(np.float32), SR)
        rows.append(dict(dataset="tv_mix", clip_id=p.stem, event_id=p.stem, group_id=b.group_id, source="home",
                         t0=0.0, t1=round(len(y) / SR, 3), label="hard_negative", subclass=f"tv_mix_{snr:+.0f}dB",
                         split="val_check", audio_path=str(p), video_path="", has_audio=True))
    log.info(f"TV mix: {len(rows)} windows")
    return pd.DataFrame(rows)
