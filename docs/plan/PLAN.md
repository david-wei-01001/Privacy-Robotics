# Plan: an autoresearch pipeline for multimodal PSI (Aim 2)

*Draft, 2026-10-26, for Elise and David. It builds on the Apple PSI prototype in `apple_psi_prototype/`.
Scope: Aim 2, privacy-preserving reporting. Incident detection and the robot's physical response are
treated as design and scenario analysis.*

**Plan version 0.8 (2026-10-27).** History: `docs/plan/CHANGELOG.md` (summary in §13.4). Decisions:
`docs/decisions/`. Update this file only together with a CHANGELOG entry (`python3 tools/plan_log.py new ...`).

Companion figures (Obsidian vault): `Research/Images/Robot PSI - architecture.pdf` (protocol) and
`Research/Images/Robot PSI - experiment pipeline.pdf` (experiments).

---

## 0. Summary

1. **Code structure.** Add a new package, `mmpsi/`, next to the crypto core `apple_psi/`, and leave the
   core unchanged. Each stage is an interface with interchangeable implementations: dataset, preprocessing,
   encoder, fusion, post-processing, hashing, matcher, private protocol, evidence builder, and metric. This is
   the **Strategy pattern**. A registry plus YAML configs selects the implementations (see §2).
2. **Search in plaintext, confirm in crypto.** All design-space search runs on cached embeddings with
   plaintext matchers, which takes seconds per run. Only the best configurations go through the private
   protocols, which check equivalence and measure leakage and cost. The protocols are Apple PSI, a banded
   variant of it, a threshold-encryption baseline, and (optionally) CKKS fuzzy PSI.
3. **Data.** The core datasets are:
   - XD-Violence: aggression with audio and video.
   - DCASE 2018 Task 5 (SINS): continuous benign home audio, including "watching TV" and "absence".
   - NTU RGB+D: skeletons of violent and benign two-person interactions.

   Optional: AudioSet, RWF-2000, HOMAGE.
4. **Metric, fixed before running.** The main metric is event-level recall at a fixed benign false-match rate
   (Anwar's first question), subject to a false-alarm budget per home-week. Privacy and cost break ties. What
   the server learns from benign uploads (Anwar's second question) is measured for every protocol.
5. **The autoresearch loop** follows Karpathy's *autoresearch*:
   - a frozen evaluation harness;
   - one editable experiment file;
   - a fixed time budget per run;
   - one keep/revert metric;
   - an append-only `results.tsv`;
   - a `program.md` written by a human that tells the agent what to explore.
6. **First step: an overnight pilot (§9.2).** A small end-to-end run on this laptop that finishes by morning.
   It answers two questions: 1) whether the pipeline and the evaluation metrics are feasible, and 2) which
   dataset setting, embedding, and similarity look promising. Larger runs follow in nightly cycles (§9.3).
7. **Final output: a report** that only compares results across datasets, embeddings, and matching
   methods. It is written as bullet points in Chenxinran's academic voice. Frozen code computes every
   number, and each number in the text must trace back to a run (see §12).

---

## 1. Goal, questions, and definition of done

**RQ2 (operational).** A robot cuts its audio and video stream into windows. The server holds a hidden set of
reference encodings of aggressive incidents. Questions:
- **Q1. Utility.** At a fixed per-window false-match rate (FMR) on benign home windows, what fraction of
  *unseen* violent events is detected? How does this depend on dataset, encoder, similarity, and the private
  matching route?
- **Q2. Privacy.** What does the server learn from benign uploads:
  - below the threshold;
  - above the threshold (benign false matches);
  - under a malicious reference set?
- **Q3. Cost.** What are voucher bytes, robot-side latency, and server opening time, as functions of the
  threshold t, the synthetic budget s_max, the number of bands L, and the evidence size?
- **Q4. Ablation.** Where is the PSI lock necessary, and where is threshold encryption with a public policy
  enough? This is answered partly empirically (Q2, Q3) and partly analytically (threat model).

**Done by Dec 3 (presentation) and Dec 10 (paper):**
1. Recall@FMR curves for at least 3 audio encoders, 2 video encoders, and 1 fused configuration, on validation
   data, with one locked test run at the end.
2. The *discretization gap*: plaintext cosine versus LSH codes versus the private protocol, compared at the
   same FMR.
3. An end-to-end demo: windows → encoder → LSH → Apple-PSI vouchers (keys renewed every epoch, coarsened
   evidence) → server opens evidence only above t.
4. Leakage experiments: occupancy inference from the timing of opened vouchers, with and without synthetic
   matches; and how many benign windows a malicious reference set can open.
5. Cost tables, and an analytical comparison of PSI against threshold encryption.
6. `reports/final_report.md`: the bullet-point comparison of datasets, embeddings, and matching (§12),
   plus one short report per stage.

---

## 2. Design principle: Strategy or interface?

Both, at different levels.

| Level | Name | Role here |
|---|---|---|
| Design pattern | **Strategy** (Gang of Four) | Each stage is a family of interchangeable algorithms (e.g. `HuBERTEncoder`, `BEATsEncoder`, `MFCCStats` are all `Encoder`s). The pipeline picks one at run time. |
| How the contract is written | **Interface**: `typing.Protocol` (structural) or `abc.ABC` | The interface states what a strategy must provide (`encode(batch) -> (n, d)`). In Python, `Protocol` lets third-party wrappers conform without inheriting from anything. |
| How strategies are chosen | **Registry + factory**, driven by config (a light form of dependency injection) | `@register("encoder", "beats")`; YAML names the components; `build()` constructs them. The runner never imports a concrete encoder. |
| Principles behind it | **Dependency inversion** and **open/closed** (SOLID) | High-level code (runner, evaluator) depends only on interfaces. A new encoder is a new file, not an edit to the pipeline. |
| Data flow | **Pipes and filters** | Typed stages, cached between stages (raw → windows → embeddings → codes → scores → metrics). |
| Crypto backends | **Ports and adapters** (hexagonal architecture) | `PrivateProtocol` is the port. Adapters wrap `apple_psi`, the threshold-encryption baseline, and CKKS. |

In one sentence: *each swappable component is a Strategy, written against an interface (Protocol), and wired
in through a registry and config files.*

---

## 3. Architecture

### 3.1 Data flow

```
DatasetAdapter ─► Windower ─► Preprocessor* ─► Encoder (per modality) ─► Fusion ─► PostProcessor*
     │ manifest + splits (by event / source)            │ cached, content-addressed         │
     ▼                                                    ▼                                    ▼
  WindowRef ───────────────────────────────────► EmbeddingSet (n × d) ─────────────┬─► Matcher (plaintext) ─► Calibrator ─► Metrics
                                                                                    │
                                                                                    └─► Hasher ─► items (bytes) ─► PrivateProtocol ─► ServerView ─► Metrics
                                                                                                     ▲
                                                                     EvidenceBuilder ─► ad (bytes) ─┘
```

`*` means zero or more, applied in order.

- The **plaintext branch** is used for search; it is fast.
- The **private branch** confirms that the private protocol makes the same decisions as its plaintext twin,
  and measures leakage and cost.

### 3.2 Repository layout (proposed)

```
privacy-robotics/
├── apple_psi_prototype/            # existing; unchanged; `pip install -e apple_psi_prototype`
│   ├── apple_psi/ …                #   crypto core (client, server, dhf, shamir, ec, …)
│   └── tests/ …
├── mmpsi/
│   ├── core/       types.py  interfaces.py  registry.py  cache.py  config.py
│   ├── data/       xd_violence.py  sins.py  ntu.py  audioset.py  rwf2000.py  homage.py
│   │               tv_mix.py  rir_augment.py  manifest.py  splits.py  windowing.py
│   ├── preprocess/ resample.py  speech_mask.py  pose_extract.py  face_blur.py
│   ├── features/
│   │   ├── audio/  mfcc_stats.py  logmel_stats.py  chromaprint.py  vggish_xd.py  yamnet.py
│   │   │           panns.py  beats.py  ast.py  clap.py  hubert.py  wavlm.py
│   │   ├── video/  frame_phash.py  motion_stats.py  pose_kinematics.py  i3d_xd.py
│   │   │           clip_frames.py  dinov2_frames.py  videomae.py  xclip.py  stgcn.py
│   │   ├── joint/  imagebind.py  cav_mae.py
│   │   ├── fusion.py               # concat, weighted, late (AND/OR/k-of-n), joint
│   │   └── post.py                 # l2norm, center, pca, whiten, dim_cap
│   ├── hashing/    simhash.py  itq.py  sign_bits.py  bit_sampling.py  pstable.py  phash_bits.py
│   ├── matching/
│   │   ├── plain/   cosine.py  l2.py  hamming.py  lsh_exact.py  classifier.py
│   │   └── private/ apple_psi_exact.py  apple_psi_banded.py  threshold_enc.py  ckks_fuzzy.py
│   ├── evidence/   coarsen.py      # builds fixed-length `ad`
│   ├── eval/       calibrate.py  metrics.py  deployment.py  leakage.py  cost.py  report.py  report_lint.py   # FROZEN for agents
│   └── run.py                      # python -m mmpsi.run <config|experiment.py>
├── docs/           plan/ (PLAN.md, CHANGELOG.md)  decisions/  reviews/   (§13)
├── configs/                        # YAML: one file per named experiment
├── autoresearch/   program.md  experiment.py  search_space.yaml  results.tsv
├── splits/         pinned split assignment + TEST_LOCK
├── tools/          plan_log.py  githooks/  night.sh  sync_obsidian.py
├── paths.yaml      data_root: data (resolved against the repo root; §13)
├── data/           git-ignored: raw datasets, derived audio, caches, model weights, logs (§13.2)
├── .venv/          git-ignored: the uv environment (Python 3.12)
├── runs/           <run_id>/{config.yaml, metrics.json, plots/}
├── reports/        TEMPLATE.md  stage_<X>.md  final_report.md  facts.json  tables/  figures/
└── tests/          test_interfaces.py  test_equivalence.py  test_splits.py  test_cache.py
```

Housekeeping: the repo root currently holds a byte-identical flat copy of `apple_psi/` and its tests. Keep only
`apple_psi_prototype/`; its imports are `apple_psi.…`. Confirm with David before deleting anything.

### 3.3 Core types and interfaces (sketch)

```python
# mmpsi/core/types.py
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

Label = Literal["violent", "benign", "hard_negative"]
Split = Literal["reference", "val_calib", "val_check", "test"]

@dataclass(frozen=True)
class WindowRef:                      # metadata only; payload is loaded lazily
    window_id: str                    # f"{dataset}/{clip_id}/{t0:.2f}"
    dataset: str
    clip_id: str
    event_id: str                     # split unit: never crosses splits
    group_id: str                     # coarser leakage unit: source movie / home / subject
    t0: float
    t1: float
    label: Label
    subclass: str                     # "fighting", "abuse", "watching_tv", "hug", …
    modalities: frozenset[str]        # {"audio", "video", "pose"}
    split: Split

@dataclass
class WindowData:
    ref: WindowRef
    audio: Optional[np.ndarray] = None    # mono float32
    sr: Optional[int] = None
    frames: Optional[np.ndarray] = None   # (T, H, W, 3) uint8
    fps: Optional[float] = None
    pose: Optional[np.ndarray] = None     # (T, persons, joints, 3)

@dataclass
class EmbeddingSet:
    refs: list[WindowRef]
    X: np.ndarray                         # (n, d) float32
    fingerprint: str                      # hash of everything upstream (cache key)
```

```python
# mmpsi/core/interfaces.py
from typing import Protocol, Sequence, Iterable
import numpy as np

class DatasetAdapter(Protocol):
    name: str
    def manifest(self) -> "pd.DataFrame": ...                       # one row per clip / annotated segment
    def windows(self, window_s: float, hop_s: float) -> list["WindowRef"]: ...
    def load(self, ref: "WindowRef", modalities: set[str]) -> "WindowData": ...

class Preprocessor(Protocol):         # resample, speech masking, room acoustics, pose extraction, …
    name: str
    def __call__(self, w: "WindowData") -> "WindowData": ...

class Encoder(Protocol):
    name: str
    modality: str                     # "audio" | "video" | "pose" | "av"
    dim: int
    neural: bool
    def fingerprint(self) -> str: ...                               # model id + layer + pooling + preprocessing
    def encode(self, batch: Sequence["WindowData"]) -> np.ndarray: ...   # (n, dim)

class Fusion(Protocol):
    def __call__(self, per_modality: dict[str, np.ndarray]) -> np.ndarray: ...

class PostProcessor(Protocol):        # fit ONLY on reference + benign-train windows
    def fit(self, X: np.ndarray) -> "PostProcessor": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...

class Hasher(Protocol):               # embedding -> PSI items (bytes); the parameters are public
    def fit(self, X_ref: np.ndarray) -> "Hasher": ...               # no-op for data-independent LSH
    def items(self, X: np.ndarray) -> list[list[bytes]]: ...        # per window: L items (band_id || code)

class Matcher(Protocol):              # plaintext decision rule (research loop)
    def fit(self, reference: "EmbeddingSet") -> "Matcher": ...
    def score(self, queries: "EmbeddingSet") -> np.ndarray: ...     # higher = more suspicious

class Calibrator(Protocol):
    def threshold(self, benign_scores: np.ndarray, fmr: float) -> float: ...

class EvidenceBuilder(Protocol):      # the `ad` carried inside a voucher
    ad_len: int
    def build(self, w: "WindowData", local_info: dict) -> bytes: ...  # exactly ad_len bytes

class PrivateProtocol(Protocol):      # PORT; adapters wrap apple_psi, CKKS, …
    name: str
    def setup(self, reference: Iterable[bytes] | np.ndarray) -> bytes: ...              # public data
    def robot_epoch(self, robot_id: str, epoch_id: int, windows: list) -> list[bytes]: ...  # messages for one epoch
    def server_epoch(self, robot_id: str, epoch_id: int, msgs: list[bytes]) -> "ServerView": ...
    def plaintext_twin(self) -> Matcher: ...                        # the decision rule it must reproduce

class Metric(Protocol):
    name: str
    def __call__(self, run: "RunArtifacts") -> float | dict: ...
```

### 3.4 Registry and config

```python
# mmpsi/core/registry.py
_REG: dict[str, dict[str, type]] = {}

def register(kind: str, name: str):
    def deco(cls):
        _REG.setdefault(kind, {})[name] = cls
        return cls
    return deco

def build(kind: str, spec: dict):
    spec = dict(spec)
    return _REG[kind][spec.pop("name")](**spec)
```

```yaml
# configs/a0_xd_prefeat.yaml   (Stage A0: no GPU)
run:      {stage: A0, seed: 0, budget_min: 5}
data:
  violent:       {name: xd_violence, subset: test_annotated, classes: [fighting, abuse]}
  benign:        {name: sins, channel: 0}
  hard_negative: [{name: sins_class, classes: [watching_tv, social_activity]}]
  window:        {seconds: 4.8, hop: 2.4}      # a multiple of VGGish's 0.96 s frame
  splits:        {reference: 0.5, val: 0.3, test: 0.2, group_by: [event_id, group_id]}
preprocess: []
encoders:
  audio: {name: vggish_xd}                     # XD pre-extracted; same VGGish model run on SINS
  video: {name: i3d_xd, stream: rgb}           # XD pre-extracted; video-only track
fusion:   {name: late_or}
post:     [{name: l2norm}, {name: center, on: benign_train}, {name: pca, dim: 128}]
hasher:   null                                 # plaintext only in A0
matcher:  {name: cosine_max, top_k: 1}
calibrate: {fmr: [0.01, 0.001]}
metrics:  [recall_event_at_fmr, recall_window_at_fmr, auc, fmr_check]
```

### 3.5 Caching

- Embeddings: `<data_root>/cache/emb/<dataset>/<window_spec>/<encoder_fingerprint>.npy`, plus `index.parquet`
  (window_id → row).
- Codes: `<data_root>/cache/codes/<emb_fingerprint>/<hasher_fingerprint>.npy`.
- Keys are content hashes of the upstream config and the dataset version. Never recompute what is cached.
  Neural extraction runs once per (dataset, window, encoder), outside the agent's time budget (§8).

### 3.6 What changes around `apple_psi` (in wrappers, not in the core)

| Need | Why | Where |
|---|---|---|
| `y` = LSH item bytes (`band_id ‖ code`) | `hash_item(y, nonce)` accepts any bytes, so no protocol change is needed | `matching/private/apple_psi_exact.py` |
| **Fresh `ClientState` per (robot, epoch)** | (a) Apple's threshold accumulates over an account's lifetime: at 0.1 % per-window FMR and 8,640 ten-second windows a day, t = 30 would be crossed by benign false matches in about 3–4 days. (b) `s_max` caps synthetic vouchers per *client lifetime*, so a continuous stream would use it up within minutes. | adapter: `ClientState.generate(params)` per epoch |
| Server sessions keyed by (robot_id, epoch_id), sharing α and pdata | `Server` keeps one account's matches in `_matches` | thin `SessionServer` wrapper |
| `ad` = coarsened evidence, fixed `ad_len` | the evidence size is a privacy knob | `evidence/coarsen.py` |
| One voucher per window per band, at a constant rate | metadata must not depend on the data | `robot_epoch()` scheduler |
| Optional faster P-256 backend for timing on robot-class hardware | pure-Python EC is the reference, not representative of real speed | later; keep the reference as the test oracle |

Voucher size with 2 cuckoo hashes is `260 + 8·(s_max+1) + ad_len` bytes (412 B at `ad_len=64, s_max=10`, which
matches the README). Bytes per window = L × voucher size. With L = 4, `ad_len` = 1 KiB, and 5 s windows, that is
about 1.1 KB/s, roughly 95 MB per day.

---

## 4. Dataset module

### 4.1 Requirements

1. Aggressive behavior with **audio and video** (both modalities are needed for multimodal matching).
2. Violent segments annotated in time, so windows can be labelled.
3. **Benign home activities** for calibrating the FMR, ideally continuous recordings (to count false alarms per
   day).
4. **Hard negatives**: TV violence, play-fighting, arguing without contact, sports.
5. A different source for cross-dataset tests ("unseen events").
6. Research-usable licence; reasonable size.

### 4.2 Candidates

| Dataset | Modalities | Content | Role | Notes |
|---|---|---|---|---|
| **XD-Violence** (ECCV 2020) | audio + video | 4,754 untrimmed videos, 217 h, from movies and YouTube; 6 violence types (abuse, car accident, explosion, fighting, riot, shooting) plus normal; weak video-level labels (1–3 per video); test set annotated in time | **Core**: reference set, validation, test (split by source) | File names encode the source (movie title or `v=<YouTube id>`) and the labels: A normal, B1 fighting, B2 shooting, B4 riot, B5 abuse, B6 car accident, G explosion. Official pre-extracted **VGGish** and **I3D** features exist, but the official OneDrive links require a Microsoft login; an unofficial Hugging Face mirror (`jherng/xd-violence`) serves the videos, annotations, and I3D/Swin/C3D features without login. Some videos may be silent or music-only, so run an audio QC step. Filter to the theme: keep fighting and abuse (shooting optional). The other classes are not domestic; keep them as "out-of-theme". |
| **DCASE 2018 Task 5 / SINS** | audio (multichannel) | One person living in a vacation home for one week; 10 s segments; daily activities (cooking, dishwashing, eating, social activity, vacuum cleaning, **watching TV**, working, **absence**, …); dev set ≈ 72,984 segments ≈ 200 h | **Core benign**: FMR calibration, false alarms per day, occupancy-leakage ground truth (absence) | "Watching TV" is a natural hard negative. Use one channel. Only one home, so the per-home-week statistics have a sample size of one; report that limitation. |
| **NTU RGB+D (60 / 120)** | RGB, depth, IR, 3D skeleton (25 joints) | Lab-recorded actions. Mutual actions include punching/slapping, kicking, pushing (120 adds hitting with an object and wielding a knife) versus hugging, handshaking, patting someone's back, giving an object, … | **Core for the pose track**: violent versus benign interactions; cross-dataset video tests | No audio. Access is by request (ROSE Lab). |
| AudioSet | audio | 2M+ ten-second YouTube clips, 527 classes (screaming, shout, slap/smack, crying, glass breaking, gunshot; plus domestic sounds) | Audio cross-dataset violent set; audio hard negatives (laughter, children playing) | Weak labels; some YouTube IDs have disappeared. |
| RWF-2000 | video | 2,000 surveillance clips, fight vs non-fight | Video cross-dataset test | Video only in the paper's usage. |
| HOMAGE (Home Action Genome, CVPR 2021) | multi-view video, microphones, other sensors | Daily home activities | Optional benign home audio + video | Check access and the audio format. |
| RLVS (Real Life Violence Situations) | video | 1,000 violent and 1,000 non-violent YouTube clips | Not recommended for audio | Most videos have no sound. |

As of 2022, the literature calls XD-Violence "by far the only available large-scale audio-visual dataset for
violence detection". So a cross-dataset *audio+video* test has to be built:
- either split XD-Violence by **source** (e.g. movies for the reference set, YouTube for test);
- or test each modality separately: AudioSet for audio, NTU and RWF-2000 for video.

**Recommended set: XD-Violence + SINS + NTU RGB+D.** Add AudioSet when audio-only cross-dataset tests are
needed.

### 4.3 Manifest schema (one parquet per dataset)

`dataset, clip_id, path_audio, path_video, path_pose, duration_s, has_audio, audio_qc (ok|silent|music),
label, subclass, segments ([t0, t1] list), event_id, group_id (movie / home / subject), source (movie|youtube|lab|home),
license, version`

### 4.4 Windowing and labels

- Fixed windows: `window_s ∈ {2.4, 4.8, 9.6}` (multiples of VGGish's 0.96 s), with 50 % hop.
- A window is *violent* if at least 50 % of it overlaps an annotated violent segment, *benign* if it has no
  overlap, and is dropped otherwise (ambiguous).
- XD-Violence: use the **time-annotated test videos** for clean window labels. Training videos have only
  video-level labels. Use them only for a larger, noisier reference set, labelled as such.
- Room acoustics: convolve XD audio with simulated room impulse responses (e.g. `pyroomacoustics`) to
  approximate far-field audio heard by a robot. This is an optional preprocessing strategy; compare runs with
  and without it.

### 4.5 Splits

- **Group by `event_id` and by `group_id` (source movie / home / subject).** Windows from the same event or the
  same movie never land in two splits. Otherwise near-duplicate windows inflate recall.
- Splits:
  - `reference`: the server's set X;
  - `val_calib`: benign windows used to set τ;
  - `val_check`: benign + unseen violent windows used to measure FMR and recall;
  - `test`: locked.
- Splitting `val_calib` from `val_check` matters: calibrating and measuring on the same windows makes the
  reported FMR too good.
- **Test lock.** The test manifest is hashed into `mmpsi/eval/TEST_LOCK`. `evaluate(split="test")` runs only
  with `MMPSI_UNLOCK_TEST=1`, which a human sets once at the end.

### 4.6 Hard negatives and augmentation

- `tv_mix`: violent XD audio (never from X) mixed into SINS at −10 to −25 dB relative to the room, labelled
  `hard_negative`. Report its false-match rate separately; it is the clearest common-cause case.
- `play`: NTU benign mutual actions; AudioSet laughter / children playing / cheering.
- `argue`: speech with high arousal but no contact. The source is still to be chosen (e.g. an acted
  emotional-speech corpus).

### 4.7 Access checklist

- [ ] XD-Violence test videos (800 files, 10.9 GB) + frame-level test annotations: Hugging Face mirror
      `jherng/xd-violence` (no login). The official OneDrive links redirect to a Microsoft login. Confirm the
      dataset's terms on the official release before publishing.
- [ ] DCASE 2018 Task 5 dev set (Zenodo).
- [ ] NTU RGB+D request form (ROSE Lab). This can take days; start now.
- [ ] AudioSet class subsets (YouTube IDs → download script; expect missing videos).
- [ ] RWF-2000 (GitHub).

---

## 5. Embedding module

### 5.1 Interfaces

`Preprocessor* → Encoder (per modality) → Fusion → PostProcessor*`. Every encoder declares:
- `modality`, `dim`, `neural`;
- `fingerprint()`;
- a **speech-content risk** tag (high / medium / low), used by the privacy metrics.

### 5.2 Candidates

**Audio**

| id | type | dim | trained on | speech risk | notes |
|---|---|---|---|---|---|
| `mfcc_stats` | non-neural | ~80 | n/a | medium | mean/std of MFCCs + RMS + spectral flux; cheap baseline |
| `logmel_stats` | non-neural | ~128 | n/a | medium | |
| `chromaprint` | non-neural fingerprint | 32-bit sub-prints | n/a | low | near-duplicate oriented (an "Apple-style" exact baseline); expected to fail on unseen events |
| `vggish_xd` | neural, pre-extracted | 128 | YouTube-8M-style audio | low–medium | **zero-cost first baseline** |
| `yamnet` | neural | 1024 | AudioSet | low–medium | light, plausible on a robot |
| `panns_cnn14` | neural | 2048 | AudioSet | low–medium | |
| `beats` | neural SSL | 768 | AudioSet | low–medium | strong sound-event encoder |
| `ast` | neural | 768 | AudioSet | low–medium | |
| `clap` | neural audio–text | 512 | audio–caption pairs | low–medium | reference set can include *text* prompts ("a person screaming in pain") |
| `hubert_L{k}` | neural speech SSL | 768 | LibriSpeech | **high** | layer choice matters; the professor flagged content leakage; keep as a speech baseline |
| `wavlm_base_plus` | neural speech SSL | 768 | speech | **high** | |

**Video**

| id | type | dim | notes |
|---|---|---|---|
| `frame_phash` | non-neural | 64 bit / frame | image hash per frame (`imagehash`), aggregated by majority bit or kept as a set of items; the closest analogue of last year's pHash |
| `motion_stats` | non-neural | ~32–64 | frame-difference energy, histograms of optical-flow magnitude (OpenCV Farnebäck) |
| `pose_kinematics` | non-neural features on top of a neural pose estimator | ~64 | joint speeds, wrist acceleration, minimum distance between people, contact events (pose from RTMPose / MediaPipe) |
| `i3d_xd` | neural, pre-extracted | 1024 (RGB) + 1024 (flow) | **zero-cost first baseline** |
| `clip_frames` | neural image (ViT-B/16) | 512 | mean over frames; text references possible |
| `dinov2_frames` | neural image | 384–768 | |
| `videomae` | neural video (ViT) | 768 | Kinetics-pretrained |
| `xclip` | neural video–text | 512 | |
| `stgcn` / `posec3d` | neural skeleton | 256 | NTU-pretrained (PYSKL / MMAction2); privacy-friendlier than RGB |

**Joint audio + video**: `imagebind` (one space for audio, video, and text; CC-BY-NC licence),
`cav_mae`.

### 5.3 Privacy-oriented preprocessing (a knob to measure, not a fixed choice)

- `speech_mask`: a voice-activity detector (Silero / WebRTC) zeroes or attenuates speech frames before
  encoding. This lowers content leakage but may hurt detection of verbal aggression. Measure both.
- `pose_only`: video reduced to skeletons; no RGB is encoded.
- `face_blur` for RGB encoders (frame-level).

### 5.4 Fusion

- `concat(w_a, w_v)` after per-modality L2 normalization.
- Late fusion: `late_and`, `late_or`, `late_k_of_n`, `late_weighted` (per-modality matchers whose decisions or
  scores are combined).
- `joint` (ImageBind): a single space.

Late fusion maps naturally onto PSI: one item stream per modality, with evidence released only if both or
k-of-n modalities match. This **constrains the server**. It is not an "independent witness" (see the
proposal feedback).

### 5.5 Post-processing (fit on reference + benign-train only)

- `l2norm`;
- `center` (subtract the benign mean, so cosine is computed against background, not raw);
- `pca(d)`, `whiten`;
- `dim_cap(512)`: CKKS fuzzy-PSI cost grows linearly with dimension, and the published protocol is
  benchmarked up to 512.

### 5.6 First set to run

- Audio: `vggish_xd`, `beats`, `clap`, `hubert_L9`, `mfcc_stats`.
- Video: `i3d_xd`, `clip_frames`, `pose_kinematics`.
- Fusion: `late_or`, `concat`.

This spans neural versus non-neural, speech versus sound-event, and RGB versus pose.

---

## 6. Matching module

### 6.1 Two layers

| Layer | Purpose | Speed | Used by |
|---|---|---|---|
| **Plaintext matchers** | Design-space search; upper bounds | ms–s per run | autoresearch inner loop |
| **Private protocols** (adapters) | Realize a chosen plaintext rule; measure leakage and cost | s–min | Stage B onwards, on finalists only |

Each private adapter declares its **plaintext twin**. A unit test checks that on a sample, the private
decisions equal the twin's decisions:
- exactly for the exact routes;
- up to the approximation band around τ for CKKS.

### 6.2 Hashers: embedding → exact-match items

| id | family | approximates | params |
|---|---|---|---|
| `simhash` | random hyperplane LSH (Charikar 2002) | cosine | k bits per band, L bands, public seed |
| `itq` | learned rotation + sign (Gong & Lazebnik 2011) | cosine | k bits; fit on reference + benign-train |
| `sign_bits` | sign of each (PCA) dimension | cosine / Hamming | d bits |
| `bit_sampling` | FLPSI-style random subsamples of a bit vector | Hamming | T subsamples of k bits |
| `pstable` | p-stable LSH (Datar et al. 2004) | L2 | bucket width, L |
| `phash_bits` | image / audio perceptual hash | near-duplicate | 64 / 96 bits |

Items are `band_id ‖ code` so that codes from different bands never collide. All hasher parameters are
**public** (they are part of the robot's firmware). Only the reference set X is secret.

### 6.3 Plaintext matchers

| id | score | twin of |
|---|---|---|
| `cosine_max(top_k)` | max (or top-k mean) cosine similarity to X | `ckks_fuzzy` |
| `l2_min` | −min L2 distance | (p-stable LSH) |
| `hamming_min` | −min Hamming distance on bits | `bit_sampling` + exact PSI |
| `lsh_exact(m)` | number of bands whose code is in X's band tables (decision: ≥ m) | `apple_psi_exact` (m = 1), `apple_psi_banded` (m > 1) |
| `classifier` | logistic regression on embeddings (local, supervised) | `threshold_enc` (public-policy baseline); also a reference point (§7.2) |

### 6.4 Private protocol adapters

| Adapter | Built from | Who learns what | Status / effort |
|---|---|---|---|
| **A. `apple_psi_exact`** | `apple_psi` as is: `Server.setup(X_items)`; `Client(params, pdata, state=fresh per epoch).create_voucher(y_band, ad)`; one voucher per (window, band) | The server learns which vouchers open (real or synthetic) and, once more than t real matches arrive in an epoch, their `ad`. The robot learns nothing. | Small: wrapper + sessions |
| **B. `apple_psi_banded`** | A, plus an inner m-of-L Shamir sharing per window (`shamir.py`): each band voucher's `ad` carries one share of a per-window key; the window's evidence is encrypted under that key | Evidence of a window opens only if (outer) the epoch threshold is crossed **and** (inner) at least m bands of that window matched. This is the FLPSI idea, built on Apple's protocol. | Medium. Note: the outer threshold counts band-matches, so t must be scaled or counting changed. |
| **C. `ckks_fuzzy`** | CKKS (TenSEAL or OpenFHE-Python), following Son et al. 2025: encrypted inner products, an approximate sign/threshold, noise flooding | In the paper, the *client* learns which of its items are close. Here the **authority must act as client**: it encrypts the references and the robot computes on its plaintext window embeddings. So the robot does the heavy work, and evidence release needs the labelled variant (or masking an evidence-key share with the indicator). | Large; optional; can be a cost-only prototype |
| **D. `threshold_enc`** | Shamir over windows, no PSI: the robot decides locally with a **public** matcher or classifier; above t positives the server reconstructs the epoch key | Same below-threshold privacy as A. But the criteria are on the device, and the robot knows which windows matched, so it can withhold selectively. | Small. This is the Aim 2 comparison. |
| (E. distance-aware PSI) | Chakraborti, Fanti, Reiter (USENIX Sec 2023) | n/a | Related work only |

### 6.5 Leakage accounting (fill per adapter; feeds Q2)

| Server observes | A | B | C | D |
|---|---|---|---|---|
| Number, size, and timing of messages | constant by construction | constant | constant | constant |
| Which windows "opened" below threshold | yes, mixed with synthetics | yes (per band), mixed with synthetics | depends on output design | no, if dummy shares look exactly like real ones (but then, above threshold, finding the real shares needs a decoding step like Apple's DHF) |
| Evidence of benign false matches above threshold | yes: FMR × `ad` content | yes, but needs m bands | yes | yes |
| Partial code bits of matched windows | band code that matched | m band codes | none (if thresholded) | none |
| What a malicious reference set can reach | any code it inserts; short codes cover more benign windows | fewer (m bands) | any region it inserts | n/a (criteria public) |

### 6.6 Evidence builder (`ad`)

`ad` is a fixed-length, padded encoding of coarsened fields. Its granularity is an experimental knob:
coarser evidence leaks less per false match.
- epoch id;
- time bucket;
- coarse zone (room type);
- top-3 sound-event tags from a local tagger;
- a quantized loudness envelope;
- an optional low-resolution skeleton keyframe;
- the local L1 check-in outcome.

---

## 7. Evaluation

### 7.1 Metrics

| Group | Metric | Definition |
|---|---|---|
| Detection | `recall_event_at_fmr(f, q)` | τ is set on `val_calib` so that benign per-window FMR = f. An event counts as detected if at least q of its windows score ≥ τ. Report q = 1 and q = t + 1. |
| | `fmr_check` | FMR actually measured on `val_check` at that τ (it drifts; report it) |
| | `auc`, `roc` | threshold-free view |
| Deployment | `false_alarms_per_home_week` | Run the full epoch logic on continuous SINS. A false alarm is an epoch with more than t opened benign windows. |
| | `delay_s` | time from event onset to the (t+1)-th match |
| | `tv_fmr` | FMR on `tv_mix` and SINS "watching TV" windows |
| Privacy | `opened_benign_per_day` | benign windows whose `ad` the server decrypts per day |
| | `timing_leak_auc` | An adversary predicts occupancy (SINS "absence" vs presence, per hour) from the timestamps of opened vouchers. Report for `synthetic_prob ∈ {0, 0.05, 0.1}`. This is Aim 2 measure (i). |
| | `scope_creep_coverage(M)` | Fraction of benign windows opened when the server adds M references aimed at benign activity (e.g. centroids of "cooking" or "watching TV"). Report as a function of code length k and m. This is Aim 2 measure (ii). |
| | `speech_probe` | Linear-probe accuracy for speaker identity and keyword recognition from embeddings and from released codes (e.g. LibriSpeech speakers, Speech Commands keywords). Compare encoders. |
| Cost | `encode_ms_per_window` | per encoder; CPU laptop and, if available, robot-class hardware (Raspberry Pi 5 / Jetson) |
| | `voucher_bytes_per_window`, `voucher_ms`, `server_ingest_ms`, `reveal_ms` | as functions of t, s_max, L, `ad_len` (extends `benchmark.py`). This is Aim 2 measure (iii). |
| | `ckks_ms`, `ckks_bytes` | as functions of dimension and \|X\| (adapter C) |
| Correctness | `equivalence_rate` | private decisions equal to plaintext-twin decisions |

### 7.2 Reference points: read results as gaps

- **Classifier vs chance.** Does the encoder separate violence at all?
- **Cosine vs classifier.** The cost of matching against a hidden list instead of classifying.
- **LSH (plaintext) vs cosine.** The cost of discretizing so that exact PSI works.
- **Private vs LSH plaintext.** Should be zero for A and B (cuckoo drops can only cause false negatives).

### 7.3 Decision rule (fix before running; edit only with both of you agreeing)

1. Keep configurations with `fmr_check ≤ 1.5 × target` (target 0.1 %) and `false_alarms_per_home_week ≤ 1`.
2. Among them, choose the highest `recall_event_at_fmr` on validation.
3. Break ties by lower `speech_probe` and `opened_benign_per_day`, then by lower cost.
4. Run test once. Report test FMR and recall with bootstrap confidence intervals.

The numbers (0.1 %, 1 per week) are placeholders. Set them with David before Stage C.

---

## 8. The autoresearch loop

This follows Karpathy's *autoresearch* pattern:
- an agent edits a single experiment file;
- each experiment has a fixed time budget;
- a frozen harness computes one metric;
- the agent keeps the change (git commit) if the metric improves and reverts it otherwise;
- results go to an append-only `results.tsv`;
- the human "programs" the agent through `program.md`.

### 8.1 Two loops

| Loop | Who | What changes | Cost per run | Gate |
|---|---|---|---|---|
| **Inner** | agent (e.g. Claude Code) | post-processing, fusion, hasher (k, L, m, ITQ), matcher (top-k, centering), window aggregation (q, window length), evidence coarsening; new light components in `mmpsi/**/contrib/` | ≤ 5 min CPU, on cached embeddings | automatic keep/revert |
| **Outer** | human (agent may propose) | new datasets, new neural encoders (GPU extraction), new private adapters, changes to metrics, constraints, or splits, and unlocking test | hours | human approval |

### 8.2 Files

- `autoresearch/program.md`: the goal, current stage, metric, constraints, allowed and forbidden edits, idea
  backlog, and stop rules. Written by humans.
- `autoresearch/experiment.py`: the only file the inner loop edits. It builds a pipeline from registry
  components and calls `mmpsi.run.execute(...)`.
- `mmpsi/eval/**`: **frozen**. The runner checks a checksum and refuses to run if it changed.
- `autoresearch/results.tsv`: append-only. Columns:
  `run_id, utc, git_sha, stage, config_hash, datasets, encoders, preprocess, fusion, post, hasher, matcher, window_s, q, t, fmr_target, fmr_check, recall_event, recall_window, auc, fa_home_week, opened_benign_day, speech_probe, cost_ms, status(keep|discard|reject|crash), note`.
- Git: one branch per session (`autoresearch/2026-11-03`). "keep" means commit; "discard" means
  `git checkout -- autoresearch/experiment.py`.
- `reports/stage_<X>.md`: written at the end of each stage, following the rules and structure in §12.

### 8.3 Objective and constraints (single scalar for keep/revert)

`objective = recall_event_at_fmr(0.001, q=1)` on `val_check`, using the decision rule of the **current track**:
- LSH track: `lsh_exact`;
- cosine track: `cosine_max`.

A run is `reject` (never kept) if:
- `fmr_check > 1.5 × target`;
- or, from Stage C on, `fa_home_week > 1`;
- or, from Stage D on, `speech_probe` exceeds the VGGish baseline.

"Keep" requires an improvement of at least 0.5 recall points, so the agent does not chase noise. Report
bootstrap confidence intervals at stage boundaries.

### 8.4 Guardrails

- Test is locked (§4.5). The agent never sets `MMPSI_UNLOCK_TEST`.
- τ comes only from the frozen `calibrate()` on `val_calib`. The agent cannot pass thresholds by hand.
- Post-processors and ITQ fit only on reference + benign-train.
- Every run records the seed, the config hash, and the cache fingerprints. Negative results stay in
  `results.tsv`.
- A wall-clock budget per run; a crash is logged and counts as a run.
- Stop after N = 15 consecutive runs without improvement, then write a short summary for the humans.

### 8.5 `program.md` skeleton

```markdown
# program.md: multimodal PSI, Stage B (matcher and LSH)
Goal: maximize recall_event_at_fmr(0.001, q=1) on val_check for the LSH track,
with encoders fixed to the Stage A finalists: {beats, clap, i3d_xd}.
You may edit: autoresearch/experiment.py, mmpsi/**/contrib/*.py
You may not edit: mmpsi/eval/**, mmpsi/data/splits.py, manifests, cache/, apple_psi_prototype/**
Run: python -m mmpsi.run autoresearch/experiment.py --budget 300
Log: the runner appends to results.tsv; never edit past rows.
Ideas to try first:
- center by benign mean before simhash; ITQ vs simhash at equal bits
- k ∈ {8, 12, 16, 24}, L ∈ {4, 8, 16, 32}, m ∈ {1, 2, 3}
- late_and vs concat fusion under LSH
- window 2.4 vs 4.8 s; q = 1 vs 2
Stop: 15 runs without improvement → write summary.md (what helped, what didn't, best config).
```

### 8.6 Stages (the funnel)

| Stage | Question | Varies | Fixed | Output |
|---|---|---|---|---|
| **A0** | Are off-the-shelf embeddings usable at all? | `vggish_xd`, `i3d_xd`, `mfcc_stats`; cosine | window 4.8 s; XD + SINS | first ROC; recall@FMR; classifier gap |
| **A** | Which encoders? | the audio and video candidates in §5.6, speech masking | cosine_max | top 3 per modality |
| **B** | How much does discretization cost? | hasher, k, L, m; fusion | finalists | LSH vs cosine gap; Apple-PSI equivalence; voucher bytes |
| **C** | Does it hold over time, at home? | window length, q, t, epoch length, synthetic_prob, `ad` coarsening | best B config | false alarms per home-week, delay, timing leak, opened benign per day, threshold-encryption comparison |
| **D** | Does it generalize, and what does it leak? | cross-dataset tests (XD split by source; AudioSet; NTU / RWF), malicious references, speech probes, cost (optionally CKKS) | frozen pipeline | final test run (once), cost tables, leakage tables |

---

## 9. Schedule: an overnight pilot first, then nightly cycles

### 9.1 Why the first version was spread over weeks

The first schedule ran each stage at full scale before starting the next:
- all of XD-Violence (217 h) through every neural encoder;
- NTU RGB+D only after access is approved;
- every private adapter built before any evaluation.

That order is right for the final numbers, but it delays the first answer by a week. The schedule below starts
instead with a small end-to-end run that finishes overnight on this laptop. Machine, measured on Oct 27: Apple M5,
32 GB RAM, no CUDA GPU, about 450 Mbit/s download, which means 50 GB takes about 15 minutes.

### 9.2 Night 1: overnight pilot (small scale, laptop only)

**Purpose.** By morning, answer two questions:
1. **Feasibility.** Does every stage run end to end? Are the metrics in §7 computable, with confidence
   intervals usable at this scale? What breaks?
2. **First comparison.** At small scale, which dataset setting, embedding, and similarity look promising?

**Scope**

| Dimension | Pilot choice | Left for later |
|---|---|---|
| Violent events | XD-Violence time-annotated test videos from the Hugging Face mirror (800 videos, 10.9 GB, no login): fighting (B1) and abuse (B5), up to 150 videos, plus up to 100 normal (A) videos; audio extracted with ffmpeg (16 kHz, mono). 40 % of the annotated videos are set aside, untouched, as the future locked test. | XD training videos; shooting |
| Benign home | SINS: 2 of the 23 audio archives (about 6,300 ten-second segments, about 17 h) plus metadata (includes "watching TV" and "absence") | the other 21 archives |
| Hard negatives | SINS "watching TV" windows; 300 "TV mix" windows (violent XD audio from validation only, mixed into SINS at −10 dB or 0 dB relative to the room recording; `tv_mix.snr_db` in the config) | play-fights, arguing |
| Cross-dataset | none (Night 4) | AudioSet features, NTU RGB+D, RWF-2000 |
| Windows | 4.8 s, hop 2.4 s | 2.4 s and 9.6 s |
| Audio encoders | `mfcc_stats`, `vggish` (run locally on both datasets), `hubert` (base, layer 9), `ast`, `clap`: the encoders implemented in `mmpsi/features.py` | `panns_cnn14` and `beats` (not implemented yet; Night 2), WavLM, speech masking; the official VGGish features as a check of the local VGGish |
| Video encoders (XD only; SINS has no video) | `clip_frames` (CLIP ViT-B/16 on keyframes, averaged within each window) | `i3d_xd` from the official I3D features in `data/raw/xd_violence/official/` (not implemented yet; Night 2), VideoMAE, pose |
| Post-processing | l2norm; centering on the benign-train mean; PCA to 128 for the bit-based methods | whitening, ITQ |
| Similarity | cosine (top-1; top-5 mean), L2, Hamming on 128 sign bits, SimHash LSH exact match (k ∈ {8, 12, 16}, L ∈ {8, 16}; m, the number of matching bands, is the smallest value that meets the FMR target on val_calib); a logistic-regression classifier as the reference | p-stable LSH, CKKS |
| Protocol | Apple PSI prototype on one epoch (150 windows, half violent and half benign, × L bands) for the best LSH setting: decision equivalence and time per voucher | per-epoch sessions at scale, banded variant, threshold encryption |
| Metrics | recall at FMR 1 % and 0.1 % (window and event level); measured FMR on val_check; AUC; TV FMR; classifier gap; discretization gap; equivalence rate; bootstrap 95 % CIs (1,000 resamples by event; where every event or none is detected, the bootstrap CI collapses to a point, so the report gives the exact Clopper-Pearson interval, ADR 0008); runtime per stage | false alarms per home-week (needs continuous recordings), leakage, cost on robot hardware |

**Timeline.** T is the start time. Each budget is a cap: a stage that overruns is cut off and logged.

| T+ | Stage | Budget |
|---|---|---|
| 0:00 | Environment (uv venv, ffmpeg), prototype tests, start downloads | 30 min |
| 0:30 | Downloads (SINS: 2 archives ≈ 4 GB; XD: test videos 10.9 GB, annotations, I3D test features), audio extraction | 60 min |
| 1:30 | Manifests, windows, splits. A timing probe runs each encoder on 2 minutes of audio, then shrinks the subsets so the remaining budget suffices. | 20 min |
| 2:00 | Embedding extraction, one encoder at a time, cached and resumable | 4 h |
| 6:00 | Similarity grid, calibration, bootstrap | 30 min |
| 6:30 | Apple PSI equivalence and timing | 15 min |
| 6:45 | `facts.json`, tables, figures, `feasibility.md` | 15 min |
| 7:00 | Buffer | 60 min |

**Robustness**
- Everything runs under `caffeinate`, so the laptop does not sleep.
- Every stage writes to the cache and can resume.
- A failing encoder is logged and skipped; it does not stop the night.
- A heartbeat file records the current stage every minute.

**Fallback.** If the mirror fails, the official links need Elise to log in once. Until then, the pilot runs
the feasibility checks and the similarity comparison on whatever XD data is present; the embedding comparison
moves to Night 2.

**Outputs** (in `reports/<night_id>/`, e.g. `reports/2026-10-27_n1/`; `python -m mmpsi.pilot --night <night_id> --report-only`
re-renders them from `facts.json`):
1. `feasibility.md`:
   - each stage: whether it ran, how long it took, and its errors;
   - data counts per split;
   - each metric: whether it was computable, and the width of its CI;
   - the scale the full run needs (e.g. how many benign windows it takes to estimate FMR = 0.1 % within a
     given CI).
2. `comparison.md`: the comparison-only report of §12 (datasets, embeddings, matching), written from
   `facts.json` in Chenxinran's bullet style.
3. `facts.json`, `tables/`, `figures/`: ROC per encoder, recall at FMR by encoder, an encoder × similarity
   heatmap.

**Go / no-go in the morning**
- **The pipeline is feasible if** at least 3 audio encoders and every similarity family completed, and the
  Apple PSI decisions match plaintext LSH, except for windows dropped from the cuckoo table.
- **The metrics are feasible if:**
  - recall at FMR 1 % has a CI no wider than ±10 points;
  - val_check holds at least 3,000 benign windows, so that FMR = 0.1 % is at least estimable (about 3
    expected false matches).

  If not, `feasibility.md` states how much more data is needed.
- **The components are promising if:**
  - an encoder's recall at FMR 1 % clearly exceeds the MFCC baseline;
  - at least one LSH setting has a small "discretization gap".
- The classifier gap shows whether the bottleneck is the encoder or the matching.

**Needed from Elise before she sleeps**
1. Approve the environment (created with uv in `.venv/` at the repo root, Python 3.12). Video and audio
   decoding uses PyAV inside the environment, so no system-wide ffmpeg install is needed.
2. XD-Violence is in `data/raw/xd_violence/`: the official test annotations, 799 of the 800 test videos from
   the Hugging Face mirror (`hf_mirror/`; one normal video is missing), and Elise's download of the official
   feature archives (`official/vggish-features.zip`, 242 MB, and `official/i3d-features.zip`, 41 GB; both pass
   a zip check and stay zipped). The pilot never extracts the archives in `official/`.
3. Approve the SINS download: 2 archives, CC BY-NC 4.0. Using the data means accepting its EULA.
4. Keep the laptop on power, with Wi-Fi on.
5. Accept the defaults: fighting and abuse only; FMR 1 % and 0.1 %; 4.8 s windows.
6. Approve the data root `data/` inside the repo, which git ignores. The project folder sits under
   `~/Desktop/Research-Projects.nosync/`, which iCloud does not sync (§13).

### 9.3 After the pilot: nightly cycles

Each night runs one larger step. Each morning starts with a 30-minute review of `feasibility.md` and
`comparison.md`.

| Night | Step | Stages (§8.6) | Adds |
|---|---|---|---|
| 1 | Pilot (§9.2) | A0, A, and a first look at B, at small scale | first answers to both questions |
| 2 | Scale up | A, B | more SINS archives (towards continuous days); all annotated XD videos except the locked test; encoders dropped on Night 1; LSH grid around the best settings. The autoresearch inner loop starts on the caches. |
| 3 | Protocol and leakage | C | Apple PSI with per-epoch sessions at scale, banded variant, threshold-encryption baseline, timing-leak experiment, TV-mix variants, false alarms per home-week |
| 4 | Generalization | D | AudioSet features for audio cross-dataset tests; NTU RGB+D and RWF-2000 once access arrives; pose encoders |
| 5 | Freeze and test | D | the decision rule (§7.3) picks the final configuration; the locked test runs once; final report (§12) |

Course deadlines are unchanged: slides by Dec 3, paper by Dec 10. A failed night can simply be rerun; there
is ample slack before Dec 3.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Embeddings do not separate unseen violence from benign home audio (what happened to last year's team) | A0 / A first, with the classifier gap as a diagnostic; sound-event encoders and pose; centering; report negative results honestly |
| Domain mismatch: movie / YouTube audio vs far-field home audio | RIR augmentation; benign data from a real home (SINS); report the in-domain vs cross-domain gap |
| XD audio is silent or music-only for some clips | QC filter (energy plus a music/speech tagger); report how many were dropped |
| Weak labels on XD training videos | use the time-annotated test videos for clean windows; label the larger reference set "noisy" |
| Data access delays (NTU) | XD + SINS are enough for A0–C |
| GPU time for neural encoders | extract once, cache, and run the agent loop on caches only |
| CKKS is hard to get right | keep it optional; cost-only prototype; otherwise cite published numbers |
| The agent overfits validation | split val_calib / val_check, lock test, require a minimum improvement, bootstrap CIs |
| Ethics / licences | public research datasets only; no real household recordings; check licences (e.g. ImageBind is CC-BY-NC) |

---

## 11. Open decisions for the team

1. The FMR target and the false-alarm budget (placeholders: 0.1 % per window; ≤ 1 per home-week).
2. Window length and epoch length (they set t's meaning and the delay).
3. Theme filter for XD (fighting + abuse only, or include shooting?).
4. What `ad` contains (sets the per-false-match leakage).
5. Whether multiple robots in one home share an account. The prototype already supports Apple §5
   multi-device accounts, which add up matches across devices.
6. Whether CKKS (adapter C) is in scope, or only a cost discussion.

---

## 12. Final output: the report

The pipeline ends with a short report that **only compares results across three dimensions: datasets,
embeddings, and matching methods**. It is written as bullet points in Chenxinran's academic voice (the
`chenxinran-writing-style` skill). Frozen code computes every number; the agent writes only the wording,
and every number traces back to a run. Leakage, cost, and protocol results are still recorded in
`results.tsv` and `facts.json`, but they are not part of this report.

### 12.1 What is produced, and when

| When | File | Content |
|---|---|---|
| End of each stage | `reports/stage_<X>.md` | the same three sections, limited to what the stage varied (e.g. Stage A: embeddings only) |
| After the single test run | `reports/final_report.md` | the three sections in §12.3 |
| Both | `reports/facts.json`, `reports/tables/*.md`, `reports/figures/*.pdf` | generated by the frozen `mmpsi/eval/report.py` |

Steps:
1. `python -m mmpsi.eval.report --stage D` (frozen) writes:
   - `facts.json`, every reportable number with its `run_id`;
   - one comparison table per section;
   - the figures: recall at FMR by dataset setting, an encoder × matcher heatmap, ROC curves.
2. The agent fills `reports/TEMPLATE.md` using the writing-style skill. It may cite only numbers that exist in
   `facts.json`.
3. `python -m mmpsi.eval.report_lint reports/final_report.md` checks that:
   - every number appears in `facts.json`, and the bullet names its run id;
   - every comparison bullet compares at least two options at the same FMR target and on the same split;
   - there are no em-dashes;
   - none of the banned words appear (`prove`, `proves`, `conclusively`, `I`, `you`), matched as whole words;
   - each of the three sections has its table and at least two bullets.
4. Elise and David review the report before it goes to Anwar.

### 12.2 Writing rules (her voice, adapted to bullets)

- **One comparison per bullet.** State the claim first (which option did better, and under what condition),
  then the evidence: both numbers, the confidence interval, and the run ids.
- **Compare like with like.** Use the same FMR target and the same split, with every other component fixed.
  The first bullet of each section states what was held fixed. When both options detect every event at FMR
  1 % (a ceiling), compare them at 0.1 % instead (ADR 0008).
- Write "we" for the team; never "I" or "you". No rhetorical questions. No em-dashes (use commas,
  parentheses, or a colon instead).
- Hedge interpretations ("suggest", "may", "appear to", "potentially"). Write unhedged statements only for
  what the table shows directly.
- Define each abbreviation on first use, e.g. "false-match rate (FMR)" or "locality-sensitive hashing
  (LSH)".
- Coin terms in quotation marks and reuse them, e.g. "discretization gap", "common-cause false matches".
- Use connectives from a small set: Furthermore, Additionally, Specifically, However, Consequently, In
  essence, In summary, Similarly, In contrast.
- Use numbered inline lists ("1) ... 2) ...").

### 12.3 Structure, with example bullets

Placeholders in `{braces}` are filled from `facts.json`. Bullets marked `[if supported]` stay only if the
table supports them.

```markdown
# Multimodal PSI: Comparison of Datasets, Embeddings, and Matching

## 1. Datasets
{table: rows = evaluation setting (XD-Violence same source, XD-Violence movies to YouTube, XD-Violence
 to AudioSet for audio, XD-Violence to NTU RGB+D and RWF-2000 for video, SINS overall, SINS "watching TV",
 TV mix); columns = recall at FMR 1% and 0.1%, measured FMR, AUC}
- To compare datasets, we fixed the encoder ({enc_best}) and the matcher ({match_best}), and varied only
  the source of the reference and test events.
- Recall decreased from {r_same}% when the reference set and the test events came from the same source to
  {r_cross}% when the test events came from YouTube (runs {run_ids}). This suggests that matching may
  depend on recording conditions as much as on the incident itself.
- Similarly, audio recall decreased from {r_xd_audio}% within XD-Violence to {r_audioset}% on AudioSet, and
  skeleton recall decreased from {r_xd_pose}% to {r_ntu}% on NTU RGB+D.
- On the benign side, the false-match rate (FMR) on SINS "watching TV" windows reached {fmr_tv}, which is
  {tv_ratio} times the FMR on other SINS windows. We term these "common-cause false matches".

## 2. Embeddings
{table: rows = encoder; columns = modality, neural or non-neural, dimension, recall at FMR 1% and 0.1%, AUC}
- To compare embeddings, we fixed the matcher (cosine similarity to the nearest reference) and the
  threshold calibration, and varied only the encoder.
- Sound-event encoders appear to separate violent and benign windows more reliably than speech encoders
  [if supported]. Specifically, BEATs reached a recall of {r_beats}% at an FMR of 0.1%, compared with
  {r_hubert}% for HuBERT (runs {run_ids}).
- In contrast, non-neural features reached {r_mfcc}% (MFCC statistics) and {r_kin}% (pose kinematics),
  which suggests that {interpretation}.
- Fusing audio and video raised recall from {r_audio_best}% to {r_fused}% [if supported], while the
  measured FMR remained at {fmr_check}.

## 3. Matching
{table: rows = matcher and parameters (cosine, Hamming on sign bits, LSH exact with k, L, m, pHash exact,
 local classifier as a non-private reference); columns = recall at FMR 1% and 0.1%, measured FMR,
 agreement with the private protocol}
{figure: recall at FMR 0.1% for every encoder × matcher pair}
- To compare matching methods, we fixed the encoder ({enc_best}) and varied only the matcher.
- Converting embeddings into locality-sensitive hashing (LSH) codes for exact matching reduced recall from
  {r_cos}% with cosine similarity to {r_lsh}% at the same FMR, a cost we term the "discretization gap"
  (runs {run_ids}).
- However, requiring at least {m} of {L} LSH bands to match recovered {gain} points, while the measured FMR
  remained at {fmr_check}.
- Hamming distance on sign bits reached {r_ham}%, which lies between cosine similarity and LSH exact
  matching [if supported].
- Perceptual hashing (pHash), which mirrors Apple's near-duplicate setting, detected {r_phash}% of unseen
  events. This is consistent with its design for copies rather than for new events.
- A local classifier on the same embeddings reached {r_clf}%, which suggests that {gap_list} points of the
  gap stem from matching against a reference list rather than from the encoder itself.
- The Apple-style PSI protocol reproduced the plaintext LSH decisions on {agree}% of windows. The remaining
  {disagree}% were false negatives caused by windows dropped from the cuckoo table.
```

---

## 13. Files, storage, and change tracking

### 13.1 Three places, three purposes

| Place | Path | Holds | Synced or versioned by |
|---|---|---|---|
| Code repo | `~/Desktop/Research-Projects.nosync/[2229] Robotics Privacy/privacy-robotics/` | code, configs, the plan, decisions, pinned splits, small run records, reports | git (`main`; Elise keeps backups); not iCloud (`.nosync`) |
| Data root | `data/` in the repo; the environment is `.venv/` at the repo root | raw datasets (including the official XD feature archives), derived audio, caches, model weights, full logs | neither synced nor in git (`.gitignore`); can be rebuilt from the `SOURCE.md` files and the code |
| Obsidian vault | `Projects/Project_Encryption/` | notes for reading: the experiment log, and links to the plan and reports | iCloud (vault) |

**Why data and code now share one folder.** The Desktop is synced to iCloud Drive (checked on Oct 27:
`~/Library/Mobile Documents/com~apple~CloudDocs/Desktop` points to `~/Desktop`), which caused two problems:
- tens of GB of videos and caches, plus a virtual environment with thousands of small files, would all upload
  to iCloud;
- "Optimize Mac Storage" could evict files in the middle of a run.

v0.5 therefore kept data, caches, models, and the environment in `~/research-data/psi-robotics/`. On Oct 27 the
project moved to `~/Desktop/Research-Projects.nosync/`: iCloud skips folders whose name ends in `.nosync`, so both
problems disappear while everything stays in one place. The data moved into `data/`, and the environment is
`.venv/` at the repo root (ADR 0007). `paths.yaml` sets `data_root: data`, resolved against the repo root, so the
code never hard-codes the location; `mmpsi/core.py` points the model caches (`HF_HOME`, `TORCH_HOME`) to
`data/models/` unless they are already set.

### 13.2 Layout

**Repo.** Small files are committed; `data/` and `.venv/` are git-ignored.

```
privacy-robotics/
├── docs/
│   ├── plan/        PLAN.md (the living plan; version in its header)   CHANGELOG.md
│   ├── decisions/   0001-modular-strategy-pipeline.md  …  0007-data-in-repo-and-night1-encoders.md
│   └── reviews/     2026-10-27_n1.md  …   (morning reviews)
├── autoresearch/    program.md  experiment.py  results.tsv (append-only)
├── configs/         pilot_n1.yaml  …
├── splits/          <split_version>.csv  TEST_LOCK   (pinned window-to-split assignment)
├── mmpsi/  apple_psi_prototype/  tests/
├── tools/           plan_log.py  githooks/  night.sh  sync_obsidian.py
├── runs/            <night_id>/{stages.json, summary.json, roc.npz, code/}   (code/: the mmpsi/*.py and config it ran)
├── reports/         <night_id>/{feasibility.md, comparison.md, facts.json, tables/, figures/}
├── paths.yaml       data_root: data
├── .venv/           uv environment, Python 3.12 (git-ignored)
└── data/            the data root (git-ignored; below)
```

**Data root** (`data/`). Large files, never committed.

```
data/
├── raw/<dataset>/                    read-only after download; SOURCE.md (URL, date, licence, sha256)
│   ├── sins/                         DCASE 2018 Task 5: meta, doc, audio.1, audio.2 (6,378 ten-second files)
│   └── xd_violence/                  annotations.txt; hf_mirror/ (799 test videos); official/ (VGGish and I3D
│                                     feature archives, kept zipped)
├── derived/<dataset>/                audio16k/ (16 kHz mono WAV); tv_mix/; windows_<config_hash>.csv
├── cache/emb/<window_hash>_<encoder_fingerprint>.npy     cache/codes/…
├── models/                           hf/ (HF_HOME) and torch/ (TORCH_HOME): model weights
└── logs/<night_id>/                  pilot.log, night.out, heartbeat.json
```

**Obsidian.** Notes only.
- `Projects/Project_Encryption/Encryption and Robotics_experiments.md`: one dated entry per night, with the
  plan version, what ran, three to five key numbers, the go / no-go result, and `file://` links to the
  reports.
- `Research/Encryption and Robotics.md`: keeps one line linking to the plan and the experiment log.

**Naming**

| What | Pattern | Example |
|---|---|---|
| Night id | `YYYY-MM-DD_nN` | `2026-10-27_n1` |
| Run id | `<night_id>-<NNNN>` | `2026-10-27_n1-0042` |
| Git tag, set before each night | `night/<night_id>` | `night/2026-10-27_n1` |
| Commit prefixes | `plan:`, `adr:`, `code:`, `run:`, `report:`, `review:` | |

### 13.3 How plan changes are recorded

1. **One source of truth.** The plan lives in `docs/plan/PLAN.md`, in git. Its header carries `version` and
   `updated`.
   - Minor version (0.x): any content change.
   - 1.0: the configuration is frozen before the locked test.
2. **CHANGELOG.md, newest first.** `tools/plan_log.py` writes the mechanical part of each entry. It compares
   the staged plan with the last commit section by section, bumps the version, and records:
   - the time, version, and commit;
   - which sections were added, removed, or changed, and the line counts.

   Whoever makes the change adds three lines:
   - **Trigger**: who or what asked for the change;
   - **Why**;
   - **Evidence**: run ids, report paths, or URLs checked.
3. **Decision records.** A choice with real alternatives gets a short file in `docs/decisions/`: context,
   options, decision, evidence, consequences, and status. A later change marks the old record "superseded by
   NNNN" instead of deleting it.
4. **Enforcement.** A pre-commit hook (`git config core.hooksPath tools/githooks`) blocks a commit that:
   - changes the plan without a matching CHANGELOG entry;
   - edits past rows of `results.tsv`;
   - changes `mmpsi/eval/` without citing a decision record in the commit message.
5. **Runs point back to the plan.** Every run writes `plan_version` and `git_sha` to `results.tsv` and to
   `meta.json`. Every report header names the plan version it ran under.
6. **The morning loop.** After night N:
   1. the reports are committed;
   2. a review note goes into `docs/reviews/`;
   3. any plan change gets a CHANGELOG entry and, if needed, a decision record;
   4. one entry is appended to the Obsidian experiment log. `tools/sync_obsidian.py` only appends; it never
      edits existing text.

### 13.4 Version history so far

This history is reconstructed: versions 0.1 to 0.4 were never committed. Once the repo is set up, it becomes
the first entries of `CHANGELOG.md`.

| Version | Date | Change | Trigger |
|---|---|---|---|
| 0.1 | 2026-10-26 | First plan: modular pipeline built on the Strategy pattern, datasets, embeddings, matching, evaluation, autoresearch loop, weekly milestones | Elise asked for a plan for a modular autoresearch pipeline |
| 0.2 | 2026-10-27 | §12: final report in Chenxinran's bullet style | Elise asked for a report in her style |
| 0.3 | 2026-10-27 | §12 narrowed to comparisons of datasets, embeddings, and matching | Elise: the report should contain only these comparisons |
| 0.4 | 2026-10-27 | §9 rewritten as an overnight pilot followed by nightly cycles; XD-Violence from the Hugging Face mirror, since the official links need a login; label codes added to §4.2 | Elise asked for a one-night small-scale run; download links checked on Oct 27 |
| 0.5 | 2026-10-27 | §13: storage layout and change tracking; data moved out of the iCloud-synced Desktop | Elise asked how files are stored and how plan changes are recorded |
| 0.6 | 2026-10-27 | Work on `main` without a branch; duplicate root copy of `apple_psi` removed; XD files from Elise's download (Hugging Face mirror as fallback); PyAV instead of a system ffmpeg | Elise: no branch needed, she keeps backups; she is downloading XD-Violence |
| 0.7 | 2026-10-27 | Data moved from `~/research-data` into `data/` (git-ignored) and the environment to `.venv/`, in a `.nosync` folder; official XD feature archives added; §9.2 lists the encoders the code implements (PANNs, BEATs, I3D move to Night 2), the TV-mix levels of the config, calibrated m, and 150 PSI windows | Elise moved the project to `Research-Projects.nosync` and asked to align the plan with the new layout and the code |
| 0.8 | 2026-10-27 | Report rules at the ceiling: the exact Clopper-Pearson interval where every event or none is detected, and comparisons at FMR 0.1 % when both options reach 100 % at 1 % (ADR 0008) | Night 1 results: three encoders at or near 100 % event recall at FMR 1 % |

---

## References

- Bhowmick, Boneh, Myers, Talwar, Tarbe. *The Apple PSI System.* 2021. (Prototype: `apple_psi_prototype/README.md`.)
- NeuralHash as CNN descriptor plus hyperplane LSH (reverse-engineered): https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX
- Sha, Wyllie. *Private Set Intersection for Training Data Auditing.* PPS Lab student blog, 2026. https://pps-lab.com/student-blogs/private-set-intersection-for-training-data-auditing/
- Son et al. *Doubly Efficient Fuzzy PSI for High-dimensional Data with Cosine Similarity.* ePrint 2025/054. https://eprint.iacr.org/2025/054
- Uzun, Chung, Kolesnikov, Boldyreva, Lee. *Fuzzy Labeled PSI (FLPSI).* USENIX Security 2021. https://www.usenix.org/conference/usenixsecurity21/presentation/uzun
- Chakraborti, Fanti, Reiter. *Distance-Aware Private Set Intersection.* USENIX Security 2023.
- Wu et al. *Not only Look, but also Listen (XD-Violence).* ECCV 2020. https://arxiv.org/abs/2007.04687 · https://roc-ng.github.io/XD-Violence/
- "By far the only available large-scale audio-visual dataset for violence detection": https://arxiv.org/abs/2207.05500
- Dekkers et al. *DCASE 2018 Task 5: Monitoring of domestic activities.* https://arxiv.org/abs/1807.11246
- Liu et al. *NTU RGB+D 120.* https://arxiv.org/abs/1905.04757 · https://rose1.ntu.edu.sg/dataset/actionRecognition/
- Cheng et al. *RWF-2000.* https://arxiv.org/abs/1911.05913
- Rai et al. *Home Action Genome (HOMAGE).* CVPR 2021.
- RLVS has mostly no audio: https://reunir.unir.net/items/44d0b32d-6c38-403e-aa17-b5727847d395/full
- Polyak et al. *Speech Resynthesis from Discrete Disentangled Self-Supervised Representations.* Interspeech 2021 (HuBERT units → speech). https://arxiv.org/abs/2104.00355
- Charikar. *Similarity estimation techniques from rounding algorithms (SimHash).* STOC 2002.
- Gong, Lazebnik. *Iterative Quantization.* CVPR 2011.
- Datar, Immorlica, Indyk, Mirrokni. *Locality-sensitive hashing based on p-stable distributions.* SoCG 2004.
- Karpathy's *autoresearch* pattern (secondhand descriptions; check the official repo):
  https://themenonlab.blog/blog/karpathy-autoresearch-autonomous-ml-experiments ·
  https://verdent.ai/guides/what-is-autoresearch-karpathy ·
  https://hackernoon.com/i-let-karpathys-autoresearch-agent-run-overnight
