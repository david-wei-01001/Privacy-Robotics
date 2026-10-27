# Privacy-Robotics: setup, data, methods, and Night 1 results

This repository studies whether a home robot can report a violent incident to a server without sending its
recordings. The robot turns each short audio or video window into an embedding, converts the embedding into
locality-sensitive hashing (LSH) codes, and sends the codes through an Apple-style private set intersection (PSI)
protocol. The server holds a reference list of violent clips and learns which codes appear in that list, but
nothing about the other codes.

This file explains how to set up the code, which datasets, embeddings, and similarity methods the first overnight
pilot used, and what that pilot (Night 1, run `2026-10-27_n1`) found. Night 1 was a small pilot, so its numbers are
meant for planning the next runs rather than as final results.

- Plan (living document, version 0.8): [docs/plan/PLAN.md](docs/plan/PLAN.md); history:
  [docs/plan/CHANGELOG.md](docs/plan/CHANGELOG.md)
- Decision records: [docs/decisions/](docs/decisions/)
- Night 1 reports: [comparison.md](reports/2026-10-27_n1/comparison.md) and
  [feasibility.md](reports/2026-10-27_n1/feasibility.md); every number is in
  [facts.json](reports/2026-10-27_n1/facts.json)

## Contents

1. [How the pipeline works](#1-how-the-pipeline-works)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Datasets](#4-datasets)
5. [Embeddings](#5-embeddings)
6. [Similarity and matching](#6-similarity-and-matching)
7. [Evaluation](#7-evaluation)
8. [Night 1 results](#8-night-1-results)
9. [Limitations and next steps](#9-limitations-and-next-steps)
10. [Adding an encoder or a matcher](#10-adding-an-encoder-or-a-matcher)
11. [References](#11-references)

## 1. How the pipeline works

```
Robot (client)                                       Server
one 4.8 s window of audio (or video)                 violent reference clips (XD-Violence)
  encoder, e.g. CLAP         -> embedding              same encoder -> embeddings
  center, L2-normalize                                 center, L2-normalize
  LSH: L bands of k bits     -> L band codes           LSH -> one table of reference codes per band (set X)
  one PSI voucher per code   ------------------------> Apple PSI: which vouchers carry a code in X
                                                       alarm when at least m of the L bands match
```

- The server learns, for each voucher, whether its code is in X. Synthetic vouchers, switched off in Night 1, would
  hide this count below the protocol threshold t.
- Plaintext search, crypto confirmation (ADR 0002): every comparison in the pilot is computed in plaintext, which is
  fast. A separate stage then runs the PSI prototype in `apple_psi_prototype/` on a sample of windows and checks that
  the private protocol reaches the same decisions as plaintext LSH.
- Besides LSH, the pilot scores the same embeddings with cosine similarity, L2 and Hamming distances, and a
  logistic-regression classifier. These matchers cannot run inside exact-match PSI; they are reference points that
  show how much recall is lost by discretizing (Section 6).
- Every encoder and matcher is a Strategy registered by name (ADR 0001), so the configuration lists them by name
  (Section 10).

## 2. Repository layout

```
apple_psi_prototype/   Apple PSI (ftPSI-AD) prototype in pure Python, 50 tests; see its README
mmpsi/                 the multimodal pipeline
  core.py              registry, paths, logging, heartbeat, stage log
  data.py              XD-Violence, SINS, TV mix; windows; group splits
  features.py          encoders (embeddings)
  matching.py          post-processing, matchers, LSH codes, Apple PSI equivalence check
  evaluate.py          threshold calibration, metrics, bootstrap CIs, dataset settings
  report.py            tables, figures, comparison.md, feasibility.md, and the check of comparison.md
  pilot.py             the Night 1 orchestrator (resumable stages)
configs/pilot_n1.yaml  the Night 1 configuration
tools/night.sh         starts a night in a detached screen session under caffeinate
tools/plan_log.py      records plan changes (CHANGELOG entries; used by the pre-commit hook)
tools/githooks/        pre-commit and commit-msg hooks
docs/plan/             PLAN.md and CHANGELOG.md
docs/decisions/        decision records 0001 to 0008
autoresearch/          results.tsv (one row per run, append-only); program.md (placeholder until Night 2)
runs/<night_id>/       stages.json, summary.json, roc.npz, and code/ (the code and config that ran)
reports/<night_id>/    comparison.md, feasibility.md, facts.json, tables/, figures/
paths.yaml             data_root (default: data/, relative to the repository root)
requirements.txt       Python packages, pinned to the versions used for Night 1
data/                  not in git: datasets, derived audio, embedding caches, model weights, logs
```

## 3. Setup

### 3.1 Requirements

- Tested on a MacBook Pro with an Apple M5 chip and 32 GB of memory, under macOS 26. The neural encoders use Apple's
  Metal backend (MPS) when it is available and the CPU otherwise; the current code does not use CUDA.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- About 21 GB of free disk space for a Night 1 run: SINS audio archives 3.7 GB, XD-Violence test videos 10 GB,
  derived audio 2.8 GB, model weights 3.6 GB, and embedding caches 0.3 GB. The optional official XD-Violence feature
  archives add 39 GB.
- Internet access on the first run, for the datasets and the model weights.
- `screen` and `caffeinate` for overnight runs (both are part of macOS).

### 3.2 Install

Run every command from the repository root.

```bash
git clone git@github.com:david-wei-01001/Privacy-Robotics.git
cd Privacy-Robotics
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt   # also installs apple_psi_prototype (editable)
git config core.hooksPath tools/githooks                        # checks plan and results.tsv changes on commit
(cd apple_psi_prototype && ../.venv/bin/python -m pytest -q)    # 50 tests should pass
```

### 3.3 Where large files go

- `paths.yaml` sets `data_root: data`, so all large files live in `data/` inside the repository, which git ignores.
  An absolute path places them elsewhere.
- `mmpsi/core.py` caches model weights in `data/models/hf` (`HF_HOME`) and `data/models/torch` (`TORCH_HOME`),
  unless these variables are already set.
- On a Mac whose Desktop or Documents folder syncs to iCloud, keep the repository in a folder whose name ends in
  `.nosync` (iCloud skips it), or point `data_root` outside iCloud (ADR 0007).

### 3.4 Get the data

The repository contains no audio, video, or embeddings. Two small files must be downloaded by hand; the pilot
downloads the rest when it is missing.

```bash
mkdir -p data/raw/sins data/raw/xd_violence

# SINS (DCASE 2018 Task 5 development set, Zenodo record 1247090): metadata (required) and documentation
for f in meta doc; do
  curl -L -o data/raw/sins/DCASE2018-task5-dev.$f.zip \
    "https://zenodo.org/records/1247090/files/DCASE2018-task5-dev.$f.zip?download=1"
done

# XD-Violence: official test annotations (start and end frames of each violent segment)
curl -L -o data/raw/xd_violence/annotations.txt https://roc-ng.github.io/XD-Violence/images/annotations.txt
shasum -a 256 data/raw/xd_violence/annotations.txt
# expected: 27b583ba06fe2e096d7e3d281845365c3b24de7e30797ae5c1ec771667e04c2f
```

Downloaded automatically on the first run:
- **SINS audio:** the archives listed under `sins.archives` in the config (Night 1: archives 1 and 2 of 23, about
  2 GB each), from the same Zenodo record.
- **XD-Violence videos:** if `data/raw/xd_violence/` holds no video, the pilot reads the test list of the Hugging Face
  mirror [`jherng/xd-violence`](https://huggingface.co/datasets/jherng/xd-violence) and downloads only the videos it
  selects (179 for Night 1) into `data/raw/xd_violence/hf_mirror/`. If videos are present (mp4, avi, or mkv, or zip
  archives of them, anywhere under `data/raw/xd_violence/` except `official/`), the pilot selects among those only.
  Night 1 had 799 of the 800 test videos; the missing one is a normal video in the locked test, so it does not change
  the selection.
- **Model weights:** from the Hugging Face Hub and torch.hub, into `data/models/`.

Optional: the official XD-Violence VGGish and I3D feature archives (`vggish-features.zip`, 242 MB;
`i3d-features.zip`, 41 GB) go in `data/raw/xd_violence/official/`. Night 1 did not use them; the I3D features are
planned for Night 2 (ADR 0007). The pilot never extracts the archives in `official/`.

Terms: the SINS development set is licensed CC BY-NC 4.0 (see the EULA in the doc archive). For XD-Violence, check
the terms on its project page before redistributing anything derived from it.

After the first run, `data/` looks like this:

```
data/
├── raw/sins/            DCASE2018-task5-dev.{meta,doc,audio.1,audio.2}.zip
├── raw/xd_violence/     annotations.txt, hf_mirror/ (videos), official/ (optional feature archives)
├── derived/             16 kHz mono WAV (sins/, xd_violence/), tv_mix/, windows_<config hash>.csv
├── cache/emb/           <window hash>_<encoder>.npy, one embedding matrix per encoder
├── models/              hf/ and torch/ (model weights)
└── logs/<night_id>/     pilot.log, night.out, heartbeat.json
```

### 3.5 Run

Start with the smoke test, which uses 6 violent and 4 normal XD videos and 120 SINS files. With the data and models
in place it takes under a minute; the first run also downloads the SINS archives and the model weights and decodes
the SINS audio, which takes longer.

```bash
.venv/bin/python -m mmpsi.pilot --night smoke1 --smoke
```

Then the full pilot, overnight:

```bash
bash tools/night.sh <night_id>                 # e.g. 2026-10-27_n1; pattern YYYY-MM-DD_nN
cat data/logs/<night_id>/heartbeat.json         # current stage and elapsed minutes, rewritten every minute
tail -f data/logs/<night_id>/pilot.log          # full log
screen -r pilot-<night_id>                      # attach; detach again with Ctrl-a d
```

- `night.sh` runs `python -m mmpsi.pilot --config configs/pilot_n1.yaml --night <night_id>` in a detached `screen`
  session under `caffeinate`, so the Mac does not sleep; keep it on power with the lid open. When the run ends, it
  renders the reports once more.
- The stages are data, probe, extract (one encoder at a time), evaluate, datasets, apple_psi, and report. The probe
  times each encoder on a few windows; if the estimated extraction time exceeds `budget.extract_s` (4 hours), the
  pilot keeps only a share of the benign groups. Night 1 needed an estimated 1.3 hours, so it kept every window.
- Runs resume: the window table (`data/derived/windows_<config hash>.csv`) and the embeddings (`data/cache/emb/`) are
  cached, so a rerun with the same configuration skips finished work. A failing encoder or stage is logged and
  skipped.
- Use a new night id for each new run: run ids restart at 0001 within a night, and `results.tsv` skips run ids it
  already holds.
- To re-render the reports from an existing `facts.json` (for example, after editing `mmpsi/report.py`):

  ```bash
  .venv/bin/python -m mmpsi.pilot --night <night_id> --report-only
  ```

  This also saves the report code it used as `runs/<night_id>/code/report_rerender.py`.

### 3.6 Outputs

| Path | Contents |
|---|---|
| `reports/<night_id>/comparison.md` | comparison of datasets, embeddings, and matching (PLAN §12); its numbers are checked against `facts.json` |
| `reports/<night_id>/feasibility.md` | stage status and runtime, data counts, CI widths, go / no-go, and the scale a full run needs |
| `reports/<night_id>/facts.json` | every run and every derived value; the source of all numbers |
| `reports/<night_id>/tables/`, `figures/` | `runs.csv` (one row per run), recall tables, encoder × matcher heatmaps, recall by encoder, ROC curves |
| `runs/<night_id>/` | `stages.json`, `summary.json`, `roc.npz`, and `code/` (a copy of `mmpsi/*.py` and the config that ran) |
| `autoresearch/results.tsv` | one row per run across nights: plan version, git state, encoder, matcher, FMR target, measured FMR, recall, AUC |
| `data/logs/<night_id>/` | `pilot.log`, `night.out`, `heartbeat.json` (not in git) |

### 3.7 Rules for changes

- The plan changes only through `python3 tools/plan_log.py new --trigger "..." --why "..." --evidence "..."
  [--decisions NNNN]`, which bumps the version and writes the CHANGELOG entry. A choice with real alternatives also
  gets a decision record in `docs/decisions/`.
- The pre-commit hook blocks a plan change without a CHANGELOG entry and any edit to earlier rows of `results.tsv`.
- The decision rule in PLAN §7.3 changes only when Elise and David both agree.
- Commit messages start with `plan:`, `adr:`, `code:`, `run:`, `report:`, or `review:` (PLAN §13.2).

## 4. Datasets

### 4.1 Overview

| Dataset | Role | Used in Night 1 | Source and terms |
|---|---|---|---|
| XD-Violence (Wu et al., ECCV 2020) | violent reference clips (the server's list) and violent test events; normal videos as the benign set of the XD track | test set only: 79 videos with fighting (B1) or abuse (B5) and 100 normal (A) videos, with the official time annotations | [project page](https://roc-ng.github.io/XD-Violence/); videos from the official release or the Hugging Face mirror |
| SINS, as the DCASE 2018 Task 5 development set (Dekkers et al.) | benign home recordings (home track) | audio archives 1 and 2 of 23: 6,378 ten-second segments from 212 sessions, plus the metadata | [Zenodo record 1247090](https://zenodo.org/records/1247090); CC BY-NC 4.0 |
| TV mix (built by the pilot) | hard negatives: violent sound from a television inside a home | 300 windows | derived from the two datasets above |

**XD-Violence.** 4,754 untrimmed videos (217 hours) from movies and YouTube, with six violence classes. Its test set
has 800 videos (500 violent and 300 normal) with frame-level annotations of the violent segments. A file name has the
form `<group>__#<start>_<end>_label_<codes>`, where the group is the movie title or `v=<YouTube id>`, and the codes
are A normal, B1 fighting, B2 shooting, B4 riot, B5 abuse, B6 car accident, and G explosion.
- 40% of the source groups (movies and YouTube videos), violent and normal alike, are set aside as the locked test
  and are never read in Night 1.
- Violent: annotated test videos with fighting or abuse among their labels, at most 150. Outside the locked test, 79
  qualified: 64 fighting, 10 fighting and shooting, 4 abuse, and 1 riot and abuse; 54 from movies and 25 from
  YouTube.
- Normal: 100 normal test videos (48 from movies and 52 from YouTube), the first 100 in a fixed hash order.
- Audio is decoded with PyAV to 16 kHz mono; `clip_frames` decodes the video keyframes.

**SINS (DCASE 2018 Task 5).** One person living in a vacation home for one week, recorded by four microphone arrays
in the combined living room and kitchen. The recordings are cut into 10-second segments labeled with nine
activities: absence, cooking, dishwashing, eating, other, social activity, vacuum cleaning, watching TV, and working.
The four arrays record the same moments, and the segments of one continuous activity form a session. The pilot uses
the first channel of each array, at 16 kHz.

**TV mix.** Each of the 300 windows adds a violent XD window from the validation split (val_violent) to a SINS
val_check window that is not "watching TV". The violent audio is scaled to -10 dB or 0 dB (chosen at random)
relative to the room recording, and the sum is peak-normalized. These windows are benign for the robot's purpose,
since the violence comes from a screen rather than from the room, so a match counts as a false match.

### 4.2 Windows, events, and labels

- Windows are 4.8 s long with a 2.4 s hop (`window_s`, `hop_s` in the config).
- In a violent video, a window is violent when it overlaps an annotated segment by at least half its length
  (2.4 s); the other windows of violent videos are not used. Each annotated segment is one **event**.
- Every window of a normal XD video or a SINS segment is benign.

### 4.3 Splits

Splits are made by group, so that windows from one movie, YouTube video, or SINS session never fall into two
splits. A group's split comes from a SHA-1 hash of its name with a fixed salt, so the same files always give the same
splits. The locked test (40% of the XD groups) is removed first.

| Split | Violent XD groups | Benign groups (SINS sessions or normal XD groups) | Used for |
|---|---|---|---|
| reference | 50% | none | the server's reference list; also the positive class of the logistic regression and part of the PCA fit |
| val_violent | 50% | none | violent test events, for event recall |
| benign_train | none | 30% | centering mean, PCA, and the negative class of the logistic regression |
| val_calib | none | 35% | setting the threshold for each FMR target |
| val_check | none | 35% | measuring the FMR at that threshold |

### 4.4 Two tracks

- **Home track:** benign windows come from SINS; the hard negatives are the TV mix and the SINS "watching TV"
  windows. Audio encoders only, since SINS has no video.
- **XD track:** benign windows come from normal XD videos. Audio and video encoders, and the audio-video fusion.
- Both tracks use the same violent reference and test windows, and each track fits its own post-processing on its
  own benign_train windows.
- The dataset comparison (Section 8.2) fixes the encoder and the matcher and varies only where the references, the
  test events, and the benign windows come from (`DATASET_SETTINGS` in `mmpsi/evaluate.py`).

### 4.5 Counts in Night 1

| Set (dataset/split/label) | Windows | Groups | Events (clips for benign sets) |
|---|---|---|---|
| sins/benign_train/benign | 5,883 | 59 | 1,961 |
| sins/val_calib/benign | 7,851 | 82 | 2,617 |
| sins/val_check/benign | 5,400 | 71 | 1,800 |
| tv_mix/val_check/hard_negative | 300 | 47 | 300 |
| xd/benign_train/benign | 2,118 | 24 | 29 |
| xd/val_calib/benign | 3,003 | 29 | 35 |
| xd/val_check/benign | 2,068 | 24 | 36 |
| xd/reference/violent | 912 | 29 | 90 |
| xd/val_violent/violent | 692 | 24 | 67 |

In total, 28,227 windows: 19,134 from SINS, 8,793 from XD-Violence, and 300 from the TV mix.

## 5. Embeddings

Each encoder turns one window into one vector. The model weights download on first use.

| Name | Modality | Model | Dim. | One vector per window | Probe, ms per window | Extraction, minutes |
|---|---|---|---|---|---|---|
| `mfcc_stats` | audio | none (librosa); the non-neural baseline | 84 | mean and standard deviation of 40 MFCCs, of RMS energy, and of onset strength | 5.4 | 1.6 |
| `vggish` | audio | VGGish, PyTorch port [`harritaylor/torchvggish`](https://github.com/harritaylor/torchvggish) via torch.hub, default settings | 128 | mean of the 0.96 s frame embeddings | 18.4 | 7.7 |
| `hubert` | audio | [`facebook/hubert-base-ls960`](https://huggingface.co/facebook/hubert-base-ls960), a speech model | 768 | mean over time of hidden layer 9 | 34.3 | 14.8 |
| `ast` | audio | [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), Audio Spectrogram Transformer fine-tuned on AudioSet | 768 | pooled output | 86.1 | 39.9 |
| `clap` | audio | [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused), contrastive language-audio pretraining | 512 | audio embedding, after resampling to 48 kHz | 17.9 | 7.6 |
| `clip_frames` | video (XD only) | [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16) (CLIP ViT-B/16) | 512 | mean of the image embeddings of the keyframes inside the window (the nearest keyframe if none) | 11.1 | 1.6 |
| `fusion:clap+clip_frames` | audio and video (XD only) | concatenation | 1024 | the L2-normalized `clap` and `clip_frames` vectors, concatenated | n/a | n/a |

- The probe and extraction times are from Night 1 on the laptop in Section 3.1. The fusion uses the audio encoder
  with the highest event recall at FMR 1% with cosine top-1 on the home track, which was `clap`.
- **Post-processing, before every matcher:** each embedding is centered on the mean of its track's benign_train
  windows and L2-normalized. `hamming_sign` also projects onto at most 128 principal components, fitted on the
  reference and benign_train windows. `l2_raw` uses the centered embeddings before normalization.
- Planned for Night 2 and not implemented yet: PANNs (`panns_cnn14`), BEATs, and `i3d_xd` from the official I3D
  features (ADR 0007).

## 6. Similarity and matching

### 6.1 Matchers

Each matcher gives every window a score; a window raises an alarm when its score reaches a threshold.

| Matcher | Score of a window | Runs inside exact-match PSI? |
|---|---|---|
| `cosine_top1` | highest cosine similarity to any reference window | no; plaintext reference point |
| `cosine_top5` | mean of the five highest cosine similarities | no |
| `l2_raw` | negative Euclidean distance to the nearest reference, on centered, unnormalized embeddings | no |
| `hamming_sign` | negative Hamming distance from the sign bits of the PCA projection (up to 128 bits) to the nearest reference | no; it needs a distance, not an exact match |
| `lsh` (k, L) | number of the L bands whose k-bit code appears in that band's reference table | **yes**: each band code is one PSI item |
| `logreg` | decision value of a logistic regression (C = 1, balanced classes) trained on reference against benign_train windows | no; a local, non-private classifier used as a reference point |

### 6.2 Thresholds

- **Continuous scores:** the threshold is set on val_calib so that at most the target share of benign windows score
  at or above it; windows tied with the cut-off do not count as matches.
- **LSH:** the threshold is m, the smallest number of matching bands whose FMR on val_calib is at most the target.
  When even m = L does not meet the target, the setting has no valid threshold ("m > L"): it raises no alarm, and its
  recall counts as zero.

### 6.3 LSH codes and PSI items

- Random-hyperplane LSH (SimHash) on the normalized embeddings: L × k Gaussian hyperplanes from a fixed public seed
  (0). The k sign bits of band b form an integer code.
- Night 1 grid: k ∈ {8, 12, 16} bits per band and L ∈ {8, 16} bands.
- A PSI item is one byte for the band index followed by the code (k/8 bytes, rounded up), so equal codes in
  different bands are different items. The server's set X holds the items of all reference windows.

### 6.4 Apple PSI equivalence check

- The pilot takes the encoder with the highest event recall at FMR 1% with cosine top-1 on the home track (`clap`)
  and its best LSH setting at that target (k = 16, L = 16, m = 6). It sends 150 windows, 75 from val_violent and 75
  from val_check, through the prototype: one voucher per band item, with t = 5, s_max = 10, 64 bytes of associated
  data, and no synthetic vouchers.
- It compares the server's decision for each window (at least m of its L vouchers match) with the plaintext LSH
  decision. Items dropped from the server's cuckoo table are the only expected cause of disagreement, and they can
  only cause misses.

## 7. Evaluation

- **FMR targets:** a per-window false-match rate (FMR) of 1% and 0.1%. Both are placeholders until the team decides
  (PLAN §11).
- **Event recall** (the main metric): the share of val_violent events with at least one window at or above the
  threshold calibrated on val_calib.
- **Measured FMR:** the share of val_check windows at or above that threshold. It can drift from the target, so it is
  reported next to every recall.
- **Also reported:** window recall; AUC over val_violent against val_check windows; FMR on the TV mix (by level) and
  on SINS "watching TV" windows; FMR by SINS activity.
- **95% confidence intervals (CIs):** bootstrap with 1,000 resamples, by event for recall and by group (SINS session
  or XD source group) for FMR. Where every event or no event is detected, the bootstrap CI collapses to one point, so
  the reports give the exact Clopper-Pearson interval instead, marked "exact" (ADR 0008).
- **Gaps** (PLAN §7.2): the classifier gap (logistic regression minus cosine) is the cost of matching against a
  hidden list instead of classifying; the discretization gap (cosine minus the best LSH setting) is the cost of
  discretizing so that exact PSI works.
- **Rules for comparisons:** a run counts only when its measured FMR is at most 1.5 times the target (PLAN §7.3,
  item 1). When two options both detect every event at 1%, they are compared at 0.1% (ADR 0008).

## 8. Night 1 results

Night 1 ran on 2026-10-27 from 02:20 to 03:35 (74.7 minutes) on the laptop in Section 3.1, and every stage finished.
It produced 280 runs: 264 combinations of track, embedding, matcher, and FMR target, plus 16 dataset-setting runs, all
logged in `autoresearch/results.tsv`. One more run (`2026-10-27_n1-0281`) is the Apple PSI check. The run used plan
version 0.6 and code that was not yet committed (git state `68a5776+dirty`); `runs/2026-10-27_n1/code/` holds the
exact code and configuration. The reports were re-rendered at 03:55 with the reporting rules of ADR 0008, without
recomputing any run.

Run ids below omit the night prefix: `0023` stands for `2026-10-27_n1-0023`. "Recall" means event recall, and the
parentheses give 95% CIs.

### 8.1 Feasibility (go / no-go, PLAN §9.2)

| Criterion | Met? | Value |
|---|---|---|
| Recall at FMR 1% has a 95% CI no wider than ±10 points | yes | largest half-width 9.0 points (67 validation events) |
| val_check holds at least 3,000 benign windows | yes | 5,400 windows |
| At least 3 audio encoders completed | yes | all five: mfcc_stats, vggish, hubert, ast, clap |
| Every similarity family ran | yes | all six matchers |
| Apple PSI decisions match plaintext LSH, except cuckoo drops | yes | 150 of 150 windows, 0 drops |
| An encoder clearly exceeds MFCC at FMR 1% (higher, and the 95% CIs do not overlap) | yes | clap +16.4 points |
| An LSH setting has a small discretization gap (at most 5 points, best encoder) | yes, at a ceiling | clap: 0.0 points at FMR 1%, where both reach 100.0%; 7.5 points at FMR 0.1% |

- The pipeline and the metrics therefore appear feasible at this scale. However, the CIs of the measured FMR are wide:
  false matches cluster by SINS session, so for clap at FMR 1% the session-level CI is ±1.85 points around 1.31%,
  6.1 times the CI over windows. This design effect of about 37 means that the 5,400 val_check windows (71 sessions)
  behave like about 145 independent windows.

### 8.2 Datasets

We fixed the encoder (`clap`) and the matcher (cosine top-1) and varied only the source of the reference events, of
the violent test events, and of the benign windows.

| Setting | Recall at FMR 1% | Recall at FMR 0.1% | Measured FMR (1% target) | AUC | Runs |
|---|---|---|---|---|---|
| references and test events from all XD sources; benign windows from SINS | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 1.31% | 0.993 | 0265, 0266 |
| all references; test events from movies only | 100.0% (91.0 to 100.0, exact) | 100.0% (91.0 to 100.0, exact) | 1.31% | 0.992 | 0267, 0268 |
| all references; test events from YouTube only | 100.0% (87.7 to 100.0, exact) | 89.3% (78.6 to 100.0) | 1.31% | 0.996 | 0269, 0270 |
| references from movies only; test events from movies | 100.0% (91.0 to 100.0, exact) | 100.0% (91.0 to 100.0, exact) | 0.93% | 0.996 | 0271, 0272 |
| references from movies only; test events from YouTube | 100.0% (87.7 to 100.0, exact) | 82.1% (67.9 to 96.4) | 0.93% | 0.994 | 0273, 0274 |
| benign windows from normal XD videos instead of SINS | 4.5% (0.0 to 10.4) | 3.0% (0.0 to 7.5) | 0.29% | 0.829 | 0275, 0276 |
| threshold calibrated on SINS, FMR measured on normal XD videos | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 40.52% | 0.829 | 0277, 0278 |
| threshold calibrated on normal XD videos, FMR measured on SINS | 4.5% (0.0 to 10.4) | 3.0% (0.0 to 7.5) | 0.00% | 0.993 | 0279, 0280 |
| SINS "watching TV" windows only (threshold from SINS) | n/a | n/a | 6.59% | n/a | 0265 |
| TV mix at 0 dB | n/a | n/a | 65.64% | n/a | 0265 |
| TV mix at -10 dB | n/a | n/a | 33.58% | n/a | 0265 |

- **Much of the home-track separation appears to come from the recording domain.** A threshold calibrated on SINS
  for an FMR of 1% gave 1.31% on held-out SINS windows (0265) but 40.52% on normal XD windows (0277). Replacing SINS
  with normal XD videos as the benign set changed recall at FMR 1% from 100.0% to 4.5% and AUC from 0.993 to 0.829
  (0265, 0275). This suggests that movie or online-video sound differs from home recordings in ways that the
  matcher picks up, rather than only the violent content.
- **Violent sound from a television inside the home matches often.** Violent XD audio mixed into SINS windows gave
  FMRs of 65.64% at 0 dB and 33.58% at -10 dB, and SINS "watching TV" windows gave 6.59%, 5.0 times the 1.31% on all
  held-out SINS windows (0265). We term these "common-cause false matches".
- **The source of the references did not clearly change recall.** With movie-only references, recall at FMR 0.1% was
  100.0% on movie events (n = 39; 0272) and 82.1% on YouTube events (n = 28; 0274); the 95% CIs overlapped.

### 8.3 Embeddings

**Home track** (cosine top-1; benign windows from SINS):

| Encoder | Dim. | Recall at FMR 1% | Recall at FMR 0.1% | AUC | Measured FMR (1% target) | Runs |
|---|---|---|---|---|---|---|
| clap | 512 | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 0.993 | 1.31% | 0023, 0024 |
| ast | 768 | 98.5% (95.5 to 100.0) | 97.0% (92.5 to 100.0) | 0.995 | 0.70% | 0089, 0090 |
| vggish | 128 | 97.0% (92.5 to 100.0) | 95.5% (91.0 to 100.0) | 0.996 | 1.50% | 0045, 0046 |
| hubert | 768 | 86.6% (77.6 to 95.5) | 37.3% (25.4 to 49.3) | 0.912 | 0.94% | 0067, 0068 |
| mfcc_stats | 84 | 83.6% (74.6 to 92.5) | 71.6% (61.2 to 82.1) | 0.974 | 0.76% | 0001, 0002 |

**XD track** (benign windows from normal XD videos; FMR target 1%). The LSH column gives the setting with the highest
recall among those whose measured FMR stays within 1.5 times the target.

| Embedding | AUC (cosine) | Cosine top-1 | Best LSH (k, L, m) | Logistic regression (measured FMR) | Runs: cosine, LSH, logreg |
|---|---|---|---|---|---|
| ast | 0.846 | 16.4% (7.5 to 25.4) | 46.3% (34.3 to 58.2); 16, 16, 5 | 64.2% (0.68%) | 0199, 0219, 0207 |
| clap | 0.882 | 10.4% (3.0 to 17.9) | 16.4% (7.5 to 25.4); 16, 8, 7 | 53.7% (0.10%) | 0133, 0151, 0141 |
| vggish | 0.841 | 1.5% (0.0 to 4.5) | 6.0% (0.0 to 11.9); 12, 16, 16 | 59.7% (0.34%) | 0155, 0171, 0163 |
| hubert | 0.453 | 0.0% (0.0 to 5.4, exact) | 20.9% (11.9 to 31.3); 12, 16, 15 | 20.9% (0.29%) | 0177, 0193, 0185 |
| mfcc_stats | 0.618 | 10.4% (3.0 to 17.9) | 0.0%: every setting needed m > L | 22.4% (1.16%) | 0111, 0121, 0119 |
| clip_frames | 0.841 | 3.0% (0.0 to 7.5) | 41.8% (29.9 to 53.7); 16, 16, 6 | 80.6% (1.98%, above the tolerance) | 0221, 0241, 0229 |
| fusion:clap+clip_frames | 0.934 | 1.5% (0.0 to 4.5) | 44.8% (32.8 to 56.7); 16, 16, 6 | 76.1% (0.24%) | 0243, 0263, 0251 |

- **On the home track, the three sound-event encoders cannot be ranked.** clap, ast, and vggish reached 97.0% to
  100.0% at FMR 1% and 95.5% to 97.0% at 0.1%, with overlapping 95% CIs at both targets (0023, 0024, 0045, 0046,
  0089, 0090). clap exceeded the MFCC baseline by 16.4 points at FMR 1% with non-overlapping CIs (0023, 0001). The
  speech encoder hubert fell to 37.3% at FMR 0.1% (0068).
- **On the XD track, every embedding separated violent from normal videos much less well.** Cosine top-1 reached
  0.0% to 16.4% at FMR 1% (0177, 0199). The measured FMR was often far below the target (for example, 0.00% for ast;
  0199), which suggests that the threshold set on val_calib transfers poorly to the other normal videos in
  val_check. Embeddings on this track were centered on normal XD windows, so the clap value differs from the
  "benign windows from normal XD videos" setting in Section 8.2.
- **Within one domain, the matching rule rather than the encoder appears to limit recall.** On the XD track, a
  logistic regression on the same embeddings reached 20.9% to 76.1% at FMR 1% (0185, 0251; leaving out clip_frames,
  whose measured FMR exceeded the tolerance), 11.9 to 74.6 points above cosine top-1. Night 1 cannot say whether a
  private matcher can recover this gap, since the classifier is not a matcher against a hidden list.
- **Fusion improved the bulk of the score distribution more than its tail.** Concatenating clap and clip_frames
  raised AUC to 0.934, from 0.882 (audio) and 0.841 (video), but recall with cosine top-1 was 1.5% (0133, 0221,
  0243); with logistic regression, the fusion reached 76.1% against 53.7% for clap alone (0251, 0141).

### 8.4 Matching

**Home track, encoder `clap`:**

| Matcher | Recall at FMR 1% | Recall at FMR 0.1% | Measured FMR (1% target) | AUC | Threshold | Runs |
|---|---|---|---|---|---|---|
| cosine_top1 | 100.0% (94.6 to 100.0, exact) | 95.5% | 1.31% | 0.993 | calibrated | 0023, 0024 |
| cosine_top5 | 100.0% (94.6 to 100.0, exact) | 97.0% | 1.65% | 0.994 | calibrated | 0025, 0026 |
| l2_raw | 98.5% (94.0 to 100.0) | 85.1% | 1.41% | 0.979 | calibrated | 0027, 0028 |
| hamming_sign | 98.5% (95.5 to 100.0) | 61.2% | 0.93% | 0.990 | calibrated | 0029, 0030 |
| logreg | 100.0% (94.6 to 100.0, exact) | 100.0% | 2.72% | 0.999 | calibrated | 0031, 0032 |
| lsh (k = 8, L = 8) | 0.0% (0.0 to 5.4, exact) | 0.0% | 0.00% | 0.961 | none (m > L) | 0033, 0034 |
| lsh (k = 8, L = 16) | 98.5% (95.5 to 100.0) | 0.0% | 1.28% | 0.982 | m = 15 | 0035, 0036 |
| lsh (k = 12, L = 8) | 95.5% (89.6 to 100.0) | 65.7% | 0.63% | 0.983 | m = 6 | 0037, 0038 |
| lsh (k = 12, L = 16) | 98.5% (95.5 to 100.0) | 88.1% | 1.07% | 0.987 | m = 11 | 0039, 0040 |
| lsh (k = 16, L = 8) | 91.0% (83.6 to 97.0) | 85.1% | 0.43% | 0.972 | m = 4 | 0041, 0042 |
| lsh (k = 16, L = 16) | 100.0% (94.6 to 100.0, exact) | 83.6% | 0.80% | 0.987 | m = 6 | 0043, 0044 |

The threshold column refers to the 1% target. The measured FMR of cosine_top5 and logreg exceeded 1.5 times the
target, so their recall is not comparable with the other rows.

- **Discretizing for exact PSI cost little at FMR 1% and more at 0.1%.** The best LSH setting (k = 16, L = 16, m = 6;
  0043) matched cosine top-1 at 100.0% at FMR 1%, a ceiling. At FMR 0.1%, the best setting (k = 12, L = 16, m = 14)
  reached 88.1% (80.6 to 95.5; 0040) against 95.5% (89.6 to 100.0; 0024), a discretization gap of 7.5 points with
  overlapping CIs. The best setting is chosen on the same validation events, so its recall is slightly optimistic.
- **The number of matching bands is a coarse score for low FMRs.** Across both tracks and all embeddings,
  calibration needed m > L in 31 of 72 LSH runs at FMR 1% and in 40 of 72 at FMR 0.1%. Short codes were the most
  affected: with k = 8, in 21 of 24 runs at FMR 1% and in all 24 at FMR 0.1%.
- **On the XD track, LSH beat cosine top-1 for 6 of the 7 embeddings** (Section 8.3), for example 46.3% against
  16.4% for ast (0219, 0199). Because the LSH score counts the bands that match any reference, it may pool evidence
  across references, whereas cosine top-1 relies on the single nearest one.
- **The private protocol reproduced plaintext LSH exactly.** For clap with k = 16, L = 16, m = 6 (LSH run 0043, PSI
  run 0281), the Apple PSI prototype reached the plaintext decision on 150 of 150 windows, 63 of which matched. None
  of the 6,073 reference items was dropped from the cuckoo table. Each voucher was 412 bytes and took 5.58 ms to
  create on the client side and 1.8 ms to process on the server, both measured on the laptop; the server setup took
  15.95 s, and its public data was 501,130 bytes. With L = 16, each window needs 16 vouchers.

## 9. Limitations and next steps

**Limitations of Night 1**
- The violent events are few (67 validation events from 35 videos) and cover only fighting and abuse from the XD
  test set, which is why the home track sits at a ceiling.
- The home-track numbers likely overstate what a robot would achieve, because SINS and XD-Violence differ in recording
  conditions (Section 8.2). The XD track keeps violent and benign windows in one domain and is the harder test.
- SINS comes from one home. The two archives already cover 212 of its 266 sessions, so more archives add windows to
  the same sessions; narrowing the FMR CIs needs more homes or sessions.
- On the XD track, the 1% threshold rests on the top 30 of 3,003 val_calib windows from only 35 normal videos. A
  manual look after the run (not part of the generated reports) suggested that the windows above that threshold came
  from a few basketball broadcasts labeled normal, so sports on television may be a hard-negative class of its own.
- Not measured yet: false alarms per home-week, leakage (timing, scope creep, speech content), and cost on robot
  hardware (PLAN §7.1).

**Planned for Night 2** (PLAN §9.3; ADR 0007): more SINS archives; all annotated XD videos except the locked test;
PANNs, BEATs, and the official I3D features; an LSH grid around the best settings; and the start of the autoresearch
loop on the cached embeddings.

**Suggested by Night 1, not decided:** use the XD track as the main comparison; add more normal videos (rather than
more windows per video) and sports broadcasts as hard negatives; try matching rules that pool evidence across
references; and report how many distinct videos or sessions lie above each threshold.

**Open decisions for the team** (PLAN §11): the FMR target and the false-alarm budget (placeholders: 1% and 0.1% per
window, and at most one false alarm per home-week), whether XD should include shooting, the window and epoch
lengths, what the associated data `ad` contains, and whether robots in one home share an account.

## 10. Adding an encoder or a matcher

Encoders and matchers are registered by name, and the configuration lists them by name.

```python
# in mmpsi/features.py
@register("encoder", "my_encoder")
class MyEncoder(AudioEncoder):
    dim, batch = 256, 16

    def load(self):               # load the model once
        ...

    def embed(self, ys):          # ys: list of 16 kHz mono float32 arrays, one per window
        return np.zeros((len(ys), self.dim), np.float32)
```

```python
# in mmpsi/matching.py
@register("matcher", "my_matcher")
class MyMatcher(Matcher):
    discrete = False              # True for integer scores such as matching bands

    def fit(self, P, ref, benign_train):   # P: Prepared embeddings; ref, benign_train: boolean masks
        self.R = P.unit[ref]
        return self

    def score(self, P, idx):               # higher means more similar to the reference list
        return (P.unit[idx] @ self.R.T).max(1)
```

Then add `my_encoder` to `encoders:` or `{name: my_matcher}` to `matchers:` in a copy of `configs/pilot_n1.yaml`.
Embeddings are cached under the encoder's `fingerprint()`, so change the fingerprint when the encoder's output
changes.

## 11. References

- Bhowmick, Boneh, Myers, Talwar, Tarbe. *The Apple PSI System.* 2021. Prototype: `apple_psi_prototype/README.md`.
- Wu et al. *Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision.* ECCV
  2020. https://arxiv.org/abs/2007.04687 · https://roc-ng.github.io/XD-Violence/
- Dekkers et al. *DCASE 2018 Challenge Task 5: Monitoring of domestic activities based on multi-channel acoustics.*
  https://arxiv.org/abs/1807.11246 · https://zenodo.org/records/1247090
- Hershey et al. *CNN Architectures for Large-Scale Audio Classification* (VGGish). ICASSP 2017.
- Hsu et al. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.*
  IEEE/ACM TASLP 2021.
- Gong, Chung, Glass. *AST: Audio Spectrogram Transformer.* Interspeech 2021.
- Wu et al. *Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption
  Augmentation* (CLAP). ICASSP 2023.
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
- Charikar. *Similarity Estimation Techniques from Rounding Algorithms* (random-hyperplane LSH). STOC 2002.
- Further PSI references (fuzzy and distance-aware PSI): PLAN, References.
