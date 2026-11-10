# Privacy-Robotics: Night 2 and after — video encoders, matched-FMR evaluation, epoch replay, and what the server learns

This file continues [INSTRUCTIONS.md](INSTRUCTIONS.md) (Night 1) and follows its structure. Sections that did not change
point back to it. It explains what was added after Night 1, how to reproduce it on a GPU machine, and what the results
say about the project's central question:

> At a fixed false-match rate on benign clips, how many violent events can we detect, and what does the server learn
> from benign uploads?

The Night 2 pilot is run `2026-10-29_n2` (one L4 GPU, 76.1 minutes). The analyses after it (matched-FMR comparison,
epoch replay, server-view adversary) ran on its cached embeddings and needed no new extraction. All numbers below come
from that run's outputs or from the analysis tools' printed output; each result section says how it was obtained.
Written 2026-10-30.

Evidence labels used below: **[run]** a number printed by code we ran; **[code]** a statement checked by reading the
code; **[inferred]** a reading of results that we have not tested directly; **[model]** an assumption of the analysis.

## In short

1. **Reproduced and scaled.** Night 1 reproduced on a GPU machine (data and results matched). Night 2 added 3 video
   encoders, 65% more benign XD windows (7,189 to 11,389), and an LSH grid with 3 hash seeds per cell, and ran in
   76 minutes. [run]
2. **Hash-seed noise is large.** The recall of one LSH setting varies by 4 to 19 points across hash seeds
   (e.g. X-CLIP, k=12, L=32: 41.9 ± 18.9%). Single-seed rankings, including Night 1's, are one random draw. [run]
3. **A good classifier does not mean a matchable embedding.** A linear classifier on X-CLIP or VideoMAE detects 94% of
   events at 1% window FMR, but matching against the reference list recovers 25 to 58%. For clip_frames, ast and their
   fusion the best LSH seed is statistically indistinguishable from the classifier at 1% FMR. [run]
4. **A cosine threshold is useless here.** Cosine top-1, which is the statistic a fuzzy PSI with a cosine threshold
   would compute, reaches 0 to 18% at 1% FMR, against 25 to 66% for LSH. No fuzzy PSI was implemented. [run, inferred]
5. **Window-level recall overstates deployability.** 1% window FMR is about 360 false window flags per day. At 10 and
   1 false alarms per home-week, the best PSI-compatible rule detects at most 17% of events (clip_frames, k=12) and the
   unmodified protocol at most 5%. The public classifier reaches 20% (clip_frames) and 48% (fusion) at 10 per week. [run]
6. **The server learns a per-window count, and it leaks occupancy.** In the protocol as implemented the server learns,
   per voucher, only match or non-match. Grouped by window, that is the count of opened vouchers, which predicts
   whether someone is home (AUC 0.89 at k=12) and the activity (41% against 11% chance, session context). The leak
   falls with k: near chance at k=24 without dummy traffic. [code, run]
7. **No operating point gives both.** k=12 has the best PSI-compatible recall and the highest leakage; k=24 has chance-level
   leakage and 7% recall (clip_frames) or about 1% (fusion). Dummy traffic is expensive (25 times the upload for a modest
   reduction at k=12). [run; conditions in Section 9]
8. **Three of our own earlier statements were wrong or too broad** and are corrected in Section 8. Read that section.

## Contents

1. [How the pipeline works](#1-how-the-pipeline-works)
2. [What was added after Night 1](#2-what-was-added-after-night-1)
3. [Setup and reproduction](#3-setup-and-reproduction)
4. [Data in Night 2](#4-data-in-night-2)
5. [Encoders](#5-encoders)
6. [Methods added after Night 1](#6-methods-added-after-night-1)
7. [Results](#7-results)
8. [Corrections to earlier statements](#8-corrections-to-earlier-statements)
9. [Limitations and next steps](#9-limitations-and-next-steps)
10. [Adding a video encoder](#10-adding-a-video-encoder)
11. [References](#11-references)

## 1. How the pipeline works

Night 1's pipeline (INSTRUCTIONS.md, Section 1) is unchanged. What changed is how we state what the server sees and how
a window decision becomes an alarm.

```
Robot (client)                                              Server
window of audio and/or video                                violent reference clips
  encoder -> embedding -> center, L2-normalize                same encoder, same preprocessing
  LSH: L bands of k bits -> L band codes                      LSH -> one table of reference codes per band (set X)
  L PSI vouchers, sent together for the window,   ------->    Apple PSI: each voucher opens (match) or not
  in shuffled order                                           Below the threshold the server learns, per voucher,
                                                              only match / non-match. Grouped by window: COUNT of
                                                              opened vouchers. It does not learn which band or
                                                              which code matched. [code]
                                                              alarm rule, one of:
                                                                A  opened vouchers in the epoch exceed t
                                                                B  windows with >= m opened bands exceed t_w
```

Alarm rules compared in this file (all evaluated in plaintext; the PSI prototype is run only as an equivalence check):

| Rule | Meaning | Needs | Real protocol? |
|---|---|---|---|
| A | alarm when the epoch's opened vouchers exceed t | nothing beyond the unmodified protocol (m = 1, no per-window grouping) | yes |
| B | a window is flagged when at least m of its L bands open (m set at window FMR f); alarm when flagged windows in the epoch exceed t_w | the vouchers grouped by window, plus a per-window evidence key (the "banded adapter", not built) | not implemented cryptographically |
| D | a window is flagged by a local logistic-regression classifier at window FMR f; same count rule | a classifier the device holds (threshold encryption with a policy check) | plaintext proxy |
| COS | a window is flagged when its maximum cosine to any reference exceeds a threshold; same count rule | stands in for a fuzzy PSI with a cosine threshold | plaintext proxy |

An **epoch** is 250 windows, which is 10 minutes at the 2.4 s hop, so there are 1,008 epochs per week. A false-alarm
budget of 1 per home-week means a probability of about 1e-3 per epoch.

## 2. What was added after Night 1

Code added, in patch order (`video_fusion.patch` v1 to v5; apply them in order if they are not yet committed):

| Path | Added in | What it does |
|---|---|---|
| `mmpsi/features_video.py` | v1 | three video encoders on a block-based base class: `motion_stats`, `videomae_k400`, `xclip_b32` (Section 5) |
| `mmpsi/pilot.py` (edits) | v1 | registers the video encoders; `fusions:` list in the config; the Apple PSI check also runs on the best XD audio, video and fusion encoder, with the run's hash seed |
| `mmpsi/fusion.py`, `tools/fusion_eval.py` | v1, v2 | late fusion of two modalities; comparison at one measured FMR; audio/video agreement; which benign videos set a threshold |
| `tools/summarize_night.py` | v2 | one-screen summary of a night from `facts.json`, with PSI equivalence results |
| `mmpsi/frontier.py`, `tools/frontier.py` | v3 | every matcher at one measured FMR (several LSH settings over several seeds), plus opened vouchers per benign window |
| `mmpsi/replay.py` | v4, v5 | the server's view of the stream, dummy traffic, upload cost, and epoch replay of the alarm rules |
| `mmpsi/adversary.py`, `tools/replay.py` | v4, v5 | a server-view adversary that infers household activity and occupancy |
| `tests/test_video_fusion.py`, `test_frontier.py`, `test_replay.py` | v1 to v5 | 39 tests on synthetic video and synthetic worlds with a known structure |

Local changes that are not in a patch: `_device()` in `mmpsi/features.py` now returns `"cuda"` when available; one line
`cpath.parent.mkdir(parents=True, exist_ok=True)` in `pilot.py` (the embedding cache directory did not exist on a fresh
data root).

Not done from the Night 2 plan (PLAN §9.3): more SINS archives (the two archives already cover 212 of 266 sessions, and
session ids are grouped by label, Section 4.2), PANNs, BEATs and the official I3D features (the encoder set already
covers AudioSet-style sound events; video encoders were added instead), and the autoresearch loop (its keep rule of
+0.5 recall points is below one event, 1/67 = 1.5 points, so it would keep noise).

## 3. Setup and reproduction

Night 1's setup (INSTRUCTIONS.md, Section 3) applies. The differences for a Linux GPU machine follow.

### 3.1 GPU machine

- Install a CUDA build of torch matching your driver. We used torch 2.11.0+cu128 with the matching torchaudio from the
  PyTorch cu128 index; check `python -c "import torch; print(torch.cuda.is_available())"`.
- Edit `_device()` in `mmpsi/features.py`:

  ```python
  def _device():
      import torch
      if torch.cuda.is_available():
          return "cuda"
      return "mps" if torch.backends.mps.is_available() else "cpu"
  ```

- Run everything from the repository root: `data/logs/` and `.venv/bin/python` are relative paths.
- Use `tmux` instead of `tools/night.sh` (which calls `caffeinate` and `pmset`). One GPU is enough: the whole Night 2
  run took 76.1 minutes, about the same as Night 1 on a laptop, because several stages run on the CPU. Do not start two
  pilots in one checkout; they share `data/derived/` and `results.tsv`.
- Moving `data/` between machines: `data/derived/windows_*.csv` stores absolute audio paths. Delete those files on the
  new machine and let the pilot rebuild them (the embedding cache key hashes window ids, not paths, so the cache stays
  valid).

### 3.2 More XD videos

`build_xd` uses `sorted(local)` when any video is present, so after a first run only the downloaded videos are
candidates. To use more normal videos, download every test video first, then raise `max_normal`:

```bash
.venv/bin/python - <<'EOF'
import logging, mmpsi.core
from pathlib import Path
from mmpsi.data import hf_test_stems, hf_download_videos
logging.basicConfig(level=logging.INFO); log = logging.getLogger("dl")
dest = Path("data/raw/xd_violence/hf_mirror")
have = {p.stem for p in dest.rglob("*.mp4")}
todo = [s for s in hf_test_stems() if s not in have]
print(len(have), "present;", len(todo), "to download")
print(len(hf_download_videos(todo, dest, log)), "downloaded")
EOF
```

With all 799 test videos present: 82 unlocked violent and 165 unlocked normal videos qualify (the 40% locked test is
unchanged). The Night 2 run used 244 videos, so 79 violent as in Night 1 and all 165 normal. [run]

### 3.3 Night 2 configuration

```bash
.venv/bin/python - <<'EOF'
import yaml
c = yaml.safe_load(open("configs/pilot_n1.yaml"))
c["xd"]["max_normal"] = 300
c["matchers"] = [m for m in c["matchers"] if m["name"] != "lsh"]
for k in (12, 16, 20, 24):
    for L in (16, 32):
        for seed in (0, 1, 2):
            c["matchers"].append({"name": "lsh", "k": k, "L": L, "seed": seed})
c["encoders"] += ["motion_stats", "videomae_k400", "xclip_b32"]
c["fusions"] = [["clap", "clip_frames"], ["ast", "clip_frames"], ["ast", "videomae_k400"], ["ast", "xclip_b32"]]
c["budget"]["extract_s"] = 36000      # not needed: the run used 76 minutes
yaml.safe_dump(c, open("configs/pilot_n2.yaml", "w"), sort_keys=False)
EOF
```

Keep the `clap+clip_frames` pair first in `fusions:`: `report.py` narrates only the first fusion row and hard-codes the
word "CLIP" in that sentence. The other fusion results are in `facts.json` and the tools below.

### 3.4 Run

```bash
mkdir -p data/logs
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m mmpsi.pilot --config configs/pilot_n2.yaml --night 2026-10-29_n2 2>&1 | tee -a data/logs/n2.out
```

Logs are in `data/logs/<night_id>/pilot.log`, reports in `reports/<night_id>/`. The run appends rows to
`autoresearch/results.tsv`.

### 3.5 Analyses on the cached embeddings

None of these encodes anything; each takes seconds to minutes. `<night>` is `2026-10-29_n2`.

```bash
# one-screen summary of a night (cosine AUC, best LSH by seed-mean, logreg, PSI equivalence, timings)
.venv/bin/python tools/summarize_night.py <night>

# audio + video reconciliation at matched FMR (late rules max / sum / min, early fusion, agreement)
mkdir -p runs/<night>/fusion
.venv/bin/python tools/fusion_eval.py --night <night> --audio ast --video clip_frames \
  --matcher '{"name":"lsh","k":16,"L":16,"seed":0}' --matcher '{"name":"cosine_top1"}'

# every matcher at one measured FMR, LSH over several settings and seeds, with opened vouchers per benign window
mkdir -p runs/<night>/frontier
.venv/bin/python tools/frontier.py --night <night> --encoder clip_frames --encoder ast --encoder fusion:ast+clip_frames \
  --grid 12,32 --grid 10,32 --grid 8,32 --grid 12,64 --grid 16,32 --seeds 5
.venv/bin/python tools/frontier.py --night <night> --track home --encoder ast --encoder clap \
  --grid 12,32 --grid 16,32 --grid 20,32 --seeds 3 --boot 100      # household audio: read the "opens" columns only

# server-view adversary (household activity and occupancy from the opened-voucher stream)
.venv/bin/python tools/replay.py attack --night <night> --encoder ast --mode count_hist \
  --grid 12,32 --grid 16,32 --grid 20,32 --grid 24,32 --s-max 0 100 1000 --context clip session --jobs 8

# epoch replay: event recall at a fixed number of false alarms per home-week
.venv/bin/python tools/replay.py utility --night <night> --encoder clip_frames --encoder fusion:ast+clip_frames \
  --grid 16,32 --grid 20,32 --grid 24,32 --seeds 3 --fw 0.001
```

Outputs are also written as JSON to `runs/<night>/{fusion,frontier,replay}/`. The attack runs dozens of attacks; with
`--jobs 8` the default grid took several minutes per cell. `--mode open open+codes` adds the what-if adversaries of
Section 6.4.

### 3.6 Tests

```bash
.venv/bin/python -m pytest tests -q                       # 39 passed (video, fusion, frontier, replay, adversary)
(cd apple_psi_prototype && ../.venv/bin/python -m pytest -q)   # 50 passed
```

### 3.7 Gotchas we hit

- Run from the repository root; redirecting output into `runs/<night>/fusion/` needs `mkdir -p` first.
- A first run on a new data root needs `cache/emb/` to exist (the one-line fix in Section 2).
- `tests/` was missing from the first pushed commit: `git add tests`.
- Use a new night id per run. Full runs append about 260 rows to `results.tsv`; discard them with
  `git checkout -- autoresearch/results.tsv` if the run was a test.
- `fusion_eval.py` and `frontier.py` refuse a night in which the pilot dropped benign windows to fit its time budget
  (`facts.json: scale.benign_fraction_kept < 1`), because the cached embeddings then belong to a window table that is not
  on disk.

## 4. Data in Night 2

### 4.1 Counts

| | Night 1 | Night 2 |
|---|---|---|
| XD videos used | 179 | 244 (165 normal, 79 violent) |
| XD windows | 8,793 | 12,993 |
| XD benign windows (train / calib / check) | 7,189 | 11,389 (3,691 / 4,030 / 3,668) |
| XD reference / validation violent windows | 912 / 692 | 912 / 692 (90 and 67 events, unchanged) |
| SINS files / windows | 6,378 / 19,134 | 6,378 / 19,134 |
| TV-mix windows | 300 | 300 |
| Total windows | 28,227 | 32,427 |

Only the benign side of XD grew. The violent side (79 videos, 90 reference events, 67 validation events) is the whole
qualifying XD test set for fighting and abuse, so recall CIs stay wide (about ±12 points) until the themes are widened or
training-split videos are added. [run]

### 4.2 SINS has no usable chronology

The metadata gives three columns: file, label, session (for example `audio/DevNode1_ex100_1.wav  other  s100`). Sorting
the 266 sessions by number, adjacent sessions share a label in 0.970 of pairs (a shuffled order gives 0.272). That is
257 of 265 pairs, which is exactly 8 label changes: **the session numbers run in label order, not in time.**
Consequences: a continuous day cannot be reconstructed, so false alarms per home-week must be extrapolated from window
rates and any occupancy timeline is synthetic; and the session id must never be used as a feature (it encodes the label).
The attack in Section 7.6 groups by session for this reason. [run]

### 4.3 Splits

Unchanged from INSTRUCTIONS.md Section 4.3 (groups by hash; 40% of groups locked as the test set, untouched).

## 5. Encoders

Nine encoders ran, plus fusions (concatenated, L2-normalized embeddings).

| Encoder | Modality | Dim | Probe ms/window | Extraction wall time | Notes |
|---|---|---|---|---|---|
| mfcc_stats | audio | 84 | 9.2 | 3.9 min | non-neural baseline |
| vggish | audio | 128 | 6.6 | 2.4 min | AudioSet CNN |
| hubert | audio | 768 | 9.9 | 4.4 min | speech model, layer 9; kept as the speech-leakage control |
| ast | audio | 768 | 36.3 | 18.9 min | AudioSet transformer |
| clap | audio | 512 | 20.3 | 9.5 min | |
| clip_frames | video | 512 | 12.7 | 1.3 min | CLIP on keyframes only; a window without a keyframe borrows the nearest one |
| motion_stats | video | 35 | 31.5 | 7.3 min | non-neural: frame-difference energy on a 4 x 4 grid and percentiles; no appearance |
| videomae_k400 | video | 768 | 60.7 | about 13 min | VideoMAE ViT-B fine-tuned on Kinetics-400, 16 frames per block |
| xclip_b32 | video | 512 | 23.6 | 5.5 min | X-CLIP ViT-B/32, 8 frames per block |

Times are for one L4 (probe) and the pilot log (wall time), 4.8 s windows. [run]

The three new video encoders share `BlockVideoEncoder` (`mmpsi/features_video.py`):

- A window is the mean of the embeddings of the non-overlapping 2.4 s **blocks** it covers (two blocks for a 4.8 s window
  with a 2.4 s hop), so neighbouring windows share a block and every block is encoded once. The constructor's `block_s`
  must equal the hop; a misaligned window raises an error.
- The frames of a block are taken at the centres of `frames` equal sub-intervals (0.15 s apart for 16 frames), resized to
  224 x 224 without cropping, as `clip_frames` does. Limits: that spacing is wider than the 1/30 s the Kinetics models
  were trained with, audio and video are aligned only through the shared window grid, and each encoder decodes every
  video itself.
- Licenses: the VideoMAE checkpoint is CC-BY-NC-4.0 (research use); the X-CLIP checkpoint is MIT. [model cards]
- Not tested here: the neural encoders were exercised only on the lab machine's real run; the unit tests cover decoding,
  block assembly, error handling and `motion_stats` on a synthetic video, not the neural models.

## 6. Methods added after Night 1

### 6.1 Matched versus deployed false-match rate

Night 2's `summarize_night.py` and Night 1's tables calibrate a threshold on `val_calib` to a nominal FMR and measure on
`val_check`. The measured FMR then differs a lot between rows: a discrete LSH score cannot hit the target, and a
threshold set on a few benign videos need not transfer (for most Night 2 cosine rows the measured FMR is 0.000%). A
method that spends more false matches recalls more, so deployed rows are not comparable unless their measured FMRs are.

`mmpsi/fusion.py` therefore reports a second comparison, **matched FMR**: recall at exactly the stated FMR, measured on
the benign windows of `val_calib` and `val_check` together (76 movies, 7,698 windows; 67 events). Where scores tie, the
result is the time-sharing mixture of the strict and lenient rule that gives that FMR exactly. The 95% bootstrap CI
resamples benign movies and events and recomputes the threshold in every resample; paired differences use the same
resamples. This measures how well a method separates, not how well a threshold transfers.

### 6.2 Combining audio and video

Matcher scores are put on one scale by z = -log P(benign score >= s), with P estimated on benign-train windows. Rules:
`max` (alarm if either modality is unusual; an OR of two per-modality server decisions), `sum` (evidence adds up;
both modalities' vouchers would count toward one threshold), `min` (both must be unusual; an AND, which needs window
grouping or nested secret sharing and is not in the unmodified protocol), and `early` (one matcher on the concatenated
embedding, one PSI key per window). The sum-rule's protocol analogue is our reading of the protocol, not tested.

### 6.3 Matchers at one FMR, and the opened-voucher count

`tools/frontier.py` scores cosine top-1, cosine top-5, logistic regression and several LSH cells over 5 hash seeds at one
measured FMR. For each LSH cell it also reports the mean number of opened vouchers per benign window (`opens`).

### 6.4 What the server sees (verified in the code)

From `apple_psi/server.py` and `client.py` [code]:

- `Server.ingest` decrypts a voucher with the server's secret and returns `match` or `non-match`. It does not record
  which table entry matched. The band number and the k-bit code are inside the hashed item and the blinded point.
- A match record holds the decrypted payload: the Shamir share `(x, y)`, the encrypted associated data and `r`. The
  share coordinate `x` is a PRF of the voucher id under the client's secret key, and the id is random unless the caller
  passes one. Nothing identifies the item.
- The pairs inside a voucher are sent in random order.

So below the threshold the server learns one bit per voucher. If the vouchers of a window arrive together, its view of
that window is the count of opened vouchers, a graded similarity score of the window to the reference list. This is what
rule B needs (grouping by window) and what rule A does not. Two **what-if** adversaries describe weaker designs and are
not the leakage of this protocol: `open` (which bands opened: a stream sent in band order, or vouchers labelled by band)
and `open+codes` (the matched entry is visible). The "bits" figure that early versions of `tools/frontier.py` printed is
the same kind of what-if (v5 relabels it).

### 6.5 Dummy traffic and upload cost

Synthetic vouchers open like real matches and carry no usable share, so below the threshold the server cannot tell them
from real ones; they never count toward the alarm. `replay.add_synthetic` spreads `s_max` synthetic matches per epoch
uniformly over the epoch's closed vouchers. [model] The price is voucher size, measured on the prototype and equal to the
formula 260 + 8 (s_max + 1) + 64 bytes:

| s_max | 0 | 10 | 100 | 1,000 | 8,000 |
|---|---|---|---|---|---|
| bytes per voucher | 332 | 412 | 1,132 | 8,332 | 64,332 |
| upload per day, L = 32 (MB) | 382 | 475 | 1,304 | 9,598 | 74,110 |

(One voucher per band per window, 36,000 windows per day.) The Apple default of `s_max = 10` is about 10 synthetic
matches against roughly 1,000 real opens per epoch, so it protects nothing here.

### 6.6 Epoch replay

Benign epochs are built by block bootstrap: 25 blocks of 10 consecutive windows of one video, resampled with replacement
from the 75 benign movies. A positive epoch replaces one block by a violent event (at most 30 windows). Each rule reduces
an epoch to a sum over its windows; the count threshold is set so that benign epochs alarm with probability alpha
(1e-2 is about 10 false alarms per week, 1e-3 about 1 per week). The window-level thresholds of rules B, D and COS are set
at a window FMR f on benign calibration windows. The bootstrap resamples movies and events and recomputes every
threshold. Benign data are XD normal videos (movies and YouTube), which open more vouchers than household audio does, so
thresholds calibrated on them are conservative for a home. SINS has no usable chronology, so an epoch is a synthetic
arrangement of blocks. [model]

### 6.7 Server-view adversary

An honest-but-curious server predicts the SINS activity (9 classes, real labels) and occupancy (absence versus any
activity) from the stream. Features: the histogram of per-window counts over the pooled windows (`count_hist`, the
protocol as implemented), the mean count (`count`), the decision bit (`alarm`), and the two what-ifs. The server pools
over a 10-second clip or a whole session (about a minute of one device). Classifier: logistic regression, stratified
group 5-fold with sessions as groups (the four recording nodes of a session stay together; the session id is never a
feature). Reported: balanced accuracy (chance 1/9 = 11.1%) and the AUC of absence versus present from out-of-fold
probabilities. Controls: the same attack on the raw embedding with labels permuted across sessions must give chance, and
the raw embedding without permutation is the non-private upper bound. One hash seed (0) and no CIs; one classifier family
is a lower bound on what a stronger adversary extracts.

## 7. Results

### 7.1 Night 2 pilot, deployed calibration (XD track, target FMR 1%)

Source: `tools/summarize_night.py 2026-10-29_n2`. Measured FMR in parentheses. [run]

| Encoder | Cosine top-1 AUC | Cosine top-1 recall (FMR) | Best LSH (k, L), seed-mean ± sd over 3 seeds (FMR) | Logreg (FMR) |
|---|---|---|---|---|
| vggish | 0.867 | 1.5% (0.0%) | k12 L32: 8.5 ± 7.1% (0.1%) | 58.2% (0.2%) |
| mfcc_stats | 0.670 | 10.4% (0.0%) | k24 L32: 10.9 ± 3.8% (0.0%) | 23.9% (2.8%) |
| hubert | 0.510 | 0.0% (0.1%) | k16 L32: 22.4 ± 1.5% (1.1%) | 26.9% (0.3%) |
| clap | 0.899 | 3.0% (0.0%) | k12 L32: 7.0 ± 7.1% (0.0%) | 52.2% (0.1%) |
| ast | 0.857 | 11.9% (0.0%) | k12 L32: 32.3 ± 6.9% (0.1%) | 59.7% (0.3%) |
| clip_frames | 0.840 | 3.0% (0.1%) | k12 L32: 51.7 ± 6.2% (0.9%) | 79.1% (1.2%) |
| xclip_b32 | 0.809 | 0.0% (0.0%) | k12 L32: 40.3 ± 3.9% (0.4%) | 94.0% (1.1%) |
| motion_stats | 0.718 | 23.9% (0.7%) | k24 L16: 25.4 ± 3.9% (0.7%) | 32.8% (1.6%) |
| videomae_k400 | 0.881 | 0.0% (0.0%) | k12 L32: 25.4 ± 8.3% (0.8%) | 94.0% (2.1%) |
| fusion:clap+clip_frames | 0.940 | 0.0% (0.0%) | k12 L32: 35.3 ± 9.9% (0.0%) | 76.1% (0.3%) |
| fusion:ast+clip_frames | 0.913 | 0.0% (0.0%) | k12 L32: 52.7 ± 11.2% (0.0%) | 79.1% (0.2%) |
| fusion:ast+videomae_k400 | 0.911 | 1.5% (0.0%) | k12 L32: 47.8 ± 3.9% (0.1%) | 88.1% (2.1%) |
| fusion:ast+xclip_b32 | 0.885 | 0.0% (0.0%) | k12 L32: 42.3 ± 15.1% (0.1%) | 89.6% (1.3%) |

Reading: the measured FMRs range from 0.0% to 2.8%, so the rows are not comparable with each other (Section 6.1). The
threshold-free AUC shows that VideoMAE (0.881) and the fusions (up to 0.940) separate violent from benign windows about
as well as the best audio encoder, while their cosine recall at the calibrated threshold is about 0%: a calibration
effect, not missing signal. HuBERT's AUC is 0.51, chance, so it carries no violence signal and is kept only as the
speech-leakage control. Best cells sit at the grid corner (k = 12, L = 32 in 10 of 13 rows). The sd over 3 seeds is 4 to
15 points.

### 7.2 Matched-FMR comparison of matchers

Source: `tools/frontier.py --night 2026-10-29_n2 ... --seeds 5` (76 benign movies, 67 events; 95% CIs). Recall is the
share of events in which any window is flagged. [run]

| Encoder | Logreg, 1% | LSH k12 L32, 1% (5 seeds) | LSH k16 L32, 1% | Cosine top-1, 1% | Logreg minus best LSH seed, 1% | Logreg, 0.1% | LSH k12 L32, 0.1% | Logreg minus best LSH seed, 0.1% |
|---|---|---|---|---|---|---|---|---|
| videomae_k400 | 94.0% [87, 99] | 24.6 ± 7.8% | 12.5 ± 2.3% | 0.0% [0, 3] | **+44.8 [+26.7, +56.1]** | 83.6% [70, 96] | 8.3 ± 1.8% | **+66.1 [+50.5, +80.3]** |
| xclip_b32 | 94.0% [82, 99] | 41.9 ± 18.9% | 20.1 ± 14.9% | 0.0% [0, 1] | **+26.6 [+13.0, +37.0]** | 70.1% [48, 90] | 24.1 ± 11.9% | **+32.3 [+7.4, +52.6]** |
| clip_frames | 77.6% [60, 88] | 56.1 ± 5.6% | 50.8 ± 4.1% | 3.0% [0, 8] | +13.3 [-3.8, +26.7] | 50.7% [37, 66] | 34.4 ± 7.6% | +1.0 [-12.1, +21.4] |
| ast | 62.7% [51, 76] | 45.8 ± 6.7% | 41.4 ± 11.1% | 17.9% [7, 31] | +5.7 [-9.1, +26.6] | 46.3% [28, 66] | 12.6 ± 4.3% | **+25.6 [+6.1, +43.6]** |
| fusion:ast+clip_frames | 79.1% [69, 90] | 61.8 ± 11.1% | 48.1 ± 5.3% | 0.0% [0, 10] | +4.1 [-3.5, +18.6] | 71.6% [55, 82] | 35.6 ± 15.7% | **+17.3 [+1.5, +28.4]** |

Bold: the 95% CI of the difference excludes 0. The comparison against the best of 25 LSH seeds favours LSH (it selects
on the same events); against the seed-mean the gaps are larger, about 13 to 21 points for clip_frames, ast and the fusion
at 1%.

Other LSH cells at 1% FMR (seed-mean ± sd over 5 seeds):

| Encoder | k=10 L=32 | k=8 L=32 | k=12 L=64 |
|---|---|---|---|
| videomae_k400 | 31.5 ± 11.2% | 29.9 ± 15.3% | 28.4 ± 9.8% |
| xclip_b32 | 49.7 ± 6.1% | 58.2 ± 7.3% | 47.6 ± 3.7% |
| clip_frames | 57.1 ± 5.4% | 50.0 ± 2.6% | 56.6 ± 4.3% |
| ast | 45.1 ± 5.9% | 24.9 ± 2.5% | 45.5 ± 3.7% |
| fusion:ast+clip_frames | 55.6 ± 11.9% | 40.7 ± 8.7% | 65.7 ± 7.3% |

Reading:
- The classifier gap is large for the two motion-aware models and not significant at 1% for clip_frames, ast and the
  fusion. Video models that classify very well match poorly: the strongest case of Night 1's warning that a good
  representation is not automatically matchable.
- LSH beats cosine by a wide margin everywhere (cosine top-1 is 0 to 18%), because LSH pools evidence across bands and
  references while top-1 looks at one reference.
- The optimum in k is bracketed at 10 to 12 for ast, clip_frames and the fusion; L = 64 gains nothing beyond seed noise
  (clip_frames 56.1 to 56.6, ast 45.8 to 45.5), so stay at L <= 32.
- At 0.1% FMR the benign pool contains only about 8 windows per false match, so those rows are indicative.

**Opened vouchers per benign window** (XD benign windows, mean over 5 seeds; "of 32" except the last column):

| Encoder | k=16 L=32 | k=12 L=32 | k=10 L=32 | k=8 L=32 | k=12 L=64 (of 64) |
|---|---|---|---|---|---|
| videomae_k400 | 1.7 | 7.3 | 14.0 | 23.0 | 14.3 |
| xclip_b32 | 1.8 | 7.4 | 13.4 | 21.5 | 14.7 |
| clip_frames | 2.2 | 7.6 | 13.2 | 20.6 | 15.3 |
| ast | 1.2 | 7.4 | 15.2 | 25.0 | 14.6 |
| fusion:ast+clip_frames | 1.1 | 6.9 | 14.1 | 23.6 | 13.7 |

On household audio (SINS benign windows, 3 seeds), fewer vouchers open: ast 4.1 (87% of windows open at least one) at
k=12, 0.8 (40%) at k=16, 0.1 (9%) at k=20; clap 4.3 (71%), 1.3 (41%), 0.4 (22%). The recall on that track is the
SINS-versus-XD domain confound (96 to 100%) and is not a result.

### 7.3 Audio and video reconciliation

Source: `tools/fusion_eval.py`, ast + each video encoder, LSH k=16 L=16, seeds 0 and 1, matched FMR. The `matcher` header
lines were not captured in the output we have; the seed mapping was inferred from the block order and checked against ast's
cosine top-1 recall (17.9%), which agrees with Section 7.2. [run, inferred]

ast + clip_frames, event recall (CI) at matched FMR:

| | Seed | FMR | audio | video | early | late max | late sum | late min |
|---|---|---|---|---|---|---|---|---|
| | 0 | 1% | 56.8 | 51.7 | 28.2 | 62.9 | 64.6 | 68.7 |
| | 0 | 0.1% | 23.8 | 14.5 | 8.8 | 21.3 | 46.4 | 44.6 |
| | 1 | 1% | 29.8 | 45.7 | 33.5 | 38.2 | 46.3 | 47.5 |
| | 1 | 0.1% | 6.6 | 29.3 | 13.0 | 22.6 | 23.9 | 20.5 |

Paired difference, late sum minus audio: seed 0: +7.8 [-3.9, +22.9] at 1%, +22.6 [+1.0, +38.6] at 0.1%; seed 1: +16.5
[+7.0, +28.3] at 1%, +17.3 [+7.0, +36.1] at 0.1%. Late sum minus video: seed 0: +12.9 [-0.2, +28.8] at 1%, +31.9 [+14.1,
+42.3] at 0.1%; seed 1: +0.6 [-8.6, +17.1], -5.5 [-17.2, +19.4].

- `late_sum` or `late_min` is the best method in 3 of the 4 seed-by-FMR cells for ast + clip_frames, and the sum rule
  beats audio with a CI excluding 0 in 3 of 4. That is suggestive, not a verdict: the hash seed alone moves ast's recall
  from 56.8% to 29.8% at the same cell.
- With VideoMAE and X-CLIP, fusion does not help (sum minus audio -20.0 [-34.9, -3.7] and -17.5 [-29.3, +3.1] at seed 0,
  1%): adding a weak modality can hurt.
- k=16 L=16 is not the best cell (Section 7.1); early fusion at the better cell k=12 L=32 reaches 61.8 ± 11.1% against
  56.1% (clip_frames) and 45.8% (ast), within noise.
- Fusion earns its extra vouchers and leakage only if the matched-FMR paired CI excludes 0 at both FMRs and for several
  seeds. That has not been shown. The agreement table (violent events found by both, audio only, video only) was not
  captured for the matched runs.

### 7.4 Apple PSI equivalence and cost

The real PSI prototype reproduces the plaintext LSH decision on 150 windows per check, for audio, video and fusion
encoders alike. [run]

| Check | k | L | m | Equivalence | Disagreements |
|---|---|---|---|---|---|
| home: clap | 20 | 32 | 5 | 1.000 | 0 |
| xd: ast | 12 | 32 | 23 | 1.000 | 0 |
| xd: clip_frames | 12 | 32 | 19 | 1.000 | 0 |
| xd: fusion:ast+clip_frames | 12 | 32 | 19 | 1.000 | 0 |

Per voucher: 412 bytes (s_max = 10), client 8.1 ms, server 2.6 ms; server setup 38 to 68 seconds. With L = 32 that is
about 260 ms of client time per window (pure-Python prototype on the lab server's CPU, not robot hardware) and 475 MB of
upload per day. Rule B's m values (19 to 23 of 32 bands) show how many bands a window needs to count.
The harness groups vouchers by window itself; the protocol does not (Section 6.4).

### 7.5 Epoch replay: recall at a fixed number of false alarms

Source: `tools/replay.py utility` (two runs, combined). 10-minute epochs, 75 benign movies, 714 blocks, 67 events, 3 hash
seeds, window FMR f for rules B, D and COS. Cells: recall (95% CI) at about **10 false alarms/week | 1 false alarm/week**.
[run]

clip_frames:

| Rule | k=12 | k=16 | k=20 | k=24 |
|---|---|---|---|---|
| A, unmodified | 4.2 [3,8] \| 1.3 [0,3] | 5.0 [4,9] \| 1.4 [1,4] | 4.5 [3,8] \| 1.0 [1,3] | 2.7 [2,6] \| 1.0 [0,3] |
| B, f = 0.1% | 16.8 [9,28] \| 9.7 [5,24] | 12.0 [7,24] \| 9.4 [5,24] | 9.7 [5,17] \| 8.1 [4,17] | 7.0 [4,14] \| 7.0 [3,14] |
| B, f = 1% | 9.4 [5,16] \| 6.0 [2,12] | 9.5 [5,17] \| 4.1 [1,12] | | |

D (public classifier): f = 1%: 10.9 [7,28] | 6.1 [2,16]; f = 0.1%: **20.4 [10,42] | 11.0 [5,39]**. COS (cosine top-1):
f = 1%: 1.1 [0,1] | 0.1 [0,0]; f = 0.1%: 3.0 [0,9] | 3.0 [0,9].

fusion:ast+clip_frames:

| Rule | k=12 | k=16 | k=20 | k=24 |
|---|---|---|---|---|
| A, unmodified | 5.3 [3,11] \| 1.9 [1,5] | 3.9 [2,11] \| 0.9 [0,4] | 2.6 [1,7] \| 0.4 [0,2] | 1.7 [1,5] \| 0.1 [0,1] |
| B, f = 0.1% | 12.7 [6,28] \| 8.6 [4,28] | 4.2 [2,19] \| 2.2 [0,19] | 3.6 [1,8] \| 2.6 [1,8] | 1.1 [0,4] \| 0.0 [0,3] |
| B, f = 1% | 9.0 [5,27] \| 4.9 [2,21] | 6.6 [3,20] \| 2.2 [0,17] | | |

D: f = 1%: 22.4 [16,44] | 12.7 [8,29]; f = 0.1%: **48.1 [31,66] | 24.8 [18,62]**. COS: f = 1%: 0.3 [0,1] | 0.1 [0,0];
f = 0.1%: 0.1 [0,1] | 0.0 [0,0].

ast (audio, k=12 and 16 only): A 4.6 | 1.3 and 3.8 | 0.9; B (f = 0.1%) 3.0 [1,9] | 1.3 [0,8] and 0.5 [0,6] | 0.1 [0,6];
D (f = 0.1%) 20.4 [7,43] | 9.6 [2,43]; COS 1.2 | 0.0.

Reading:
- The window-level recalls of Section 7.2 (45 to 66% at 1% window FMR) correspond to about 360 false window flags per day.
  At 1 to 10 false alarms per week, the best PSI-compatible rule (B, clip_frames, k=12) detects 17% and 10%; the unmodified
  protocol (A) at most 5%, because a benign epoch opens about 1,850 vouchers and the count threshold swamps the event's
  signal. The window-level numbers must not be quoted as deployable recall.
- B is the only PSI-compatible rule with usable recall, and it needs the vouchers grouped by window.
- B falls as k grows (clip_frames 16.8, 12.0, 9.7, 7.0 at 10 per week); for the fusion it collapses (12.7 to 1.1).
  The fusion's B is worse than clip_frames' at the epoch level although its window-level LSH recall is higher.
- D reaches 20% (clip_frames) and 48% (fusion) at 10 per week, and 11% and 25% at 1 per week. For clip_frames, B and D
  overlap widely; for the fusion and ast, D is far ahead.
- COS is 0 to 3% throughout.
- Even the best row detects about a quarter of events at one false alarm per week, on a background of movies and YouTube,
  with 75 benign movies, untuned thresholds, and events of about 10 windows. These numbers are conservative for a home
  but they show that the application, as specified, is far from working on this benchmark.

### 7.6 What the server learns: the adversary

Source: `tools/replay.py attack --encoder ast` (household audio, SINS: 212 sessions, 9 activities, one hash seed). Chance
(balanced accuracy) is 11.1%. Raw embedding (non-private upper bound): 83.2% per clip, 92.7% per session, occupancy AUC
0.97 and 0.98. Control with labels permuted across sessions: 11.0% and 9.0% (chance: no leak in the pipeline). The
decision bit alone (`alarm`) gives 11.1 to 16.7% at every k. [run]

**The protocol as implemented (`count_hist`).** Balanced accuracy / occupancy AUC, clip context | session context:

| k (L = 32) | opened per window (s_max = 0) | s_max = 0 (382 MB/day) | s_max = 100 (1,304 MB/day) | s_max = 1,000 (9,598 MB/day) |
|---|---|---|---|---|
| 12 | 4.1 | 29.2% / 0.89 \| 40.8% / 0.89 | 27.4% / 0.88 \| 35.4% / 0.88 | 24.8% / 0.83 \| 35.2% / 0.87 |
| 16 | 0.8 | 24.8% / 0.84 \| 34.6% / 0.87 | 19.6% / 0.80 \| 27.5% / 0.85 | 15.1% / 0.67 \| 23.4% / 0.74 |
| 20 | 0.1 | 16.8% / 0.66 \| 24.2% / 0.76 | 14.4% / 0.61 \| 16.9% / 0.68 | 10.6% / 0.54 \| 9.7% / 0.54 |
| 24 | 0.0 | 12.3% / 0.52 \| 17.5% / 0.61 | 10.5% / 0.52 \| 14.2% / 0.54 | 11.0% / 0.49 \| 11.8% / 0.50 |

Reading:
- Without dummy traffic and with vouchers grouped by window, the count alone reveals occupancy (AUC 0.89 at k=12) and
  the activity at 3.7 times chance over a minute of context (40.8% against 11.1%).
- The leak falls steeply with k: at k = 24 it is near chance per clip and low per session, without any dummy traffic.
- The Apple default `s_max = 10` changes nothing (in the earlier `count` run, at k=12 and clip context: 24.3% against
  24.6% at s_max = 0).
- Dummy traffic works but is costly: at k = 12 it takes `s_max` = 1,000 (25 times the upload) to lower the clip-level
  accuracy from 29.2% to 24.8% and the session-level accuracy from 40.8% to 35.2%. At k = 20, `s_max` = 1,000 reaches chance
  (10.6% and 9.7%); a larger k is far cheaper per unit of protection.
- The classifier is a lower bound; there is one hash seed and no CIs, and a few points of difference are not significant.

**What-ifs, from the first attack run** (not the protocol; balanced accuracy / occupancy AUC, clip | session, s_max = 0):
`open+codes` 67.7% / 0.95 | 83.0% / 0.94 at k = 12; 48.7% / 0.86 | 66.4% / 0.90 at k = 16; 25.3% / 0.66 | 43.6% / 0.74 at
k = 20. If a design let the server learn the matched entry, activity inference would approach the raw-embedding bound, and
dummy traffic with random table entries would not hide it (47.3% / 70.0% at k = 12 with `s_max` = 8,000, against 74 GB/day).

### 7.7 Utility against leakage

The leakage can only be measured on audio (SINS has no video), while the best PSI-compatible utility comes from
clip_frames. The two columns therefore use different encoders (utility: clip_frames, rule B at window FMR 0.1%;
leakage: ast with `count_hist`, `s_max` = 0). Hash cells and epochs are the same. Indicative, not a single-encoder frontier.

| k (L = 32) | Event recall at ~10 \| ~1 false alarms/week (clip_frames, rule B) | Activity accuracy, clip \| session (ast; chance 11.1%) | Occupancy AUC, clip \| session (ast) |
|---|---|---|---|
| 12 | 16.8% \| 9.7% | 29.2% \| 40.8% | 0.89 \| 0.89 |
| 16 | 12.0% \| 9.4% | 24.8% \| 34.6% | 0.84 \| 0.87 |
| 20 | 9.7% \| 8.1% | 16.8% \| 24.2% | 0.66 \| 0.76 |
| 24 | 7.0% \| 7.0% | 12.3% \| 17.5% | 0.52 \| 0.61 |

For ast itself (the only encoder measured on both axes), rule B recalls at most 3% at 10 false alarms per week (k = 12),
so the leakage side is moot. The public classifier (D) has no per-window leakage in the idealized model (the server sees
only encrypted vouchers) and reaches 20% to 48% at 10 per week, but it exposes the criteria to the device owner, which is
the property PSI was chosen to avoid. The comparison is a threat-model question as much as a measurement.

**Answer to the project question, as of Night 2.** At a window FMR of 1% (about 360 false window flags a day), LSH + PSI
detects 46 to 62% of events and a public classifier 63 to 79% (clip_frames, ast, fusion). At 10 false alarms per home-week,
the PSI-compatible rules detect at most 17% and the classifier 20 to 48%. What the server learns from benign uploads is
a per-window count that identifies occupancy (AUC 0.89) and some activity at k = 12 and is near chance at k = 24, at which
the recall is 7% or lower. No operating point on this benchmark combines useful detection and low leakage.

## 8. Corrections to earlier statements

We list these so that nobody quotes the earlier versions.

1. **Fusion compared "at the same FMR" (v1 of `fusion_eval.py`).** It calibrated to one nominal FMR, but the measured FMRs
   differed (0.109% for audio, 0.872% for late_max), so a "+9.0% [3.0, 16.4], CI excludes 0" flag compared methods that
   spent different numbers of false matches. v2 compares at matched measured FMR (Section 6.1). The deployed rows remain
   in the output but carry no significance flags.
2. **"A public classifier beats the hidden list by tens of points."** That came from unmatched FMRs. At matched FMR
   it holds for VideoMAE (+44.8) and X-CLIP (+26.6) only; for clip_frames, ast and the fusion the best LSH seed is within
   the CI at 1% (Section 7.2).
3. **"An opened voucher reveals the window's k-bit code; dummy traffic hides counts but not codes."** Wrong for the protocol
   as implemented (Section 6.4). The server learns match or non-match only. The first attack run's `open+codes` rows and
   the "bits" column of the frontier tool are what-ifs. The leak of the protocol is the count; v5 fixes the docstrings,
   adds the count-histogram adversary, and relabels the what-if modes.
4. **Night 1's LSH ranking.** A single hash seed (seed 0) was used; the spread across seeds is 4 to 19 points.
5. **A 1% window FMR as a headline.** It is about 360 false window flags a day (Section 7.5).

## 9. Limitations and next steps

**Limitations**
- The benign pool is 75 to 76 movies (11,389 windows). The false-alarm tail below about 1 per week cannot be estimated;
  every "1 per week" row rests on a handful of videos. The violent side is 67 events from 24 groups.
- The benign background is XD normal video (movies, YouTube, sports), which is not a household. Thresholds calibrated on it
  are conservative for a home; the household background (SINS) has no violent events, and overlaying XD audio on it
  reintroduces the domain confound. A real household estimate of false alarms per week does not exist yet.
- Utility (XD, in-domain) and leakage (SINS, audio only) come from different tracks and, for video, different encoders.
  There is no household video, so the leakage of video encoders is unmeasured.
- The adversary uses one classifier family, one hash seed and no CIs. Dummy traffic is modelled as uniform noise over closed
  vouchers with a budget per 250-window epoch; the real protocol's `s_max` is per account and its design for streams
  differs. Rule B is evaluated in plaintext; the banded adapter that realizes it (and its per-window evidence key) is not
  built, and whether the window grouping can be made cryptographically without exposing the count is open.
- The epoch length (10 minutes), block length (10 windows) and event cap (30 windows) are assumptions; events average about
  10 windows (27 s).
- Client cost is measured on a lab CPU in pure Python, not on robot hardware.
- Not built: a fuzzy PSI (COS is a statistic-level proxy), the threshold-encryption adapter (D is a plaintext classifier),
  the evidence field `ad` (the harness uses 64 zero bytes), the speech-content probe with HuBERT, the locked-test run.

**Next steps, in the order we would take them** (the deadlines in PLAN §9.3 are slides on Dec 3 and the paper on Dec 10)
1. **Group k-fold evaluation.** Score every unlocked benign video and every violent event out of fold (about 165 benign
   movies and about 157 events, with a larger reference set per fold). This tightens every CI above, and is needed before
   the locked test.
2. **Household-realistic false alarms.** Define a synthetic, clearly labelled household background (SINS blocks with overlaid
   violent audio at several levels) and report false alarms per home-week and recall there, separately from XD.
3. **Protocol design experiments.** (a) Send vouchers unordered across windows (no per-window grouping) and measure how
   much leakage disappears and what utility rule A can then reach with shorter epochs; (b) a per-window dummy-traffic
   design matched to the benign count distribution; (c) a cryptographic way to count a window's matches without revealing
   the count.
4. **A learned projection before the hash.** A discriminative projection learned on public data recovers part of the
   classifier gap inside PSI, but publishes part of the criterion. Measure the recall gained and how well an owner can
   predict matches from the public projection.
5. **A stronger adversary.** More seeds with CIs, a gradient-boosted classifier, longer context, and a HuBERT speech-content
   probe for the representation-privacy side.
6. **Freeze and run the locked test once** (Night 5), with the metrics fixed in advance: matched FMR, epoch recall at a
   stated false-alarm budget, and the leakage table.
7. **Record the decisions** in PLAN and the decision records (`tools/plan_log.py`): the server-view model (Section 6.4),
   matched-FMR comparison, and the epoch alarm rules. Add the leakage and cost results to the report; ADR 0004 currently
   keeps them out of it.

**Open decisions for the team**
- The false-alarm budget and the epoch length (placeholders here: 10 and 1 per week, 10-minute epochs); the window FMR
  used by rules B, D and COS (0.1% worked better than 1%).
- Whether to adopt the per-window grouping assumption (rule B) as the design, or to treat leakage-free rule A as the
  baseline and report that it does not work.
- Which operating point to freeze: a utility-first cell (k = 12) or a leakage-first cell (k >= 20).
- Whether XD should include shooting and riot (this would add violent events); what the associated data `ad` contains;
  whether robots in one home share an account.

## 10. Adding a video encoder

Subclass `BlockVideoEncoder` in `mmpsi/features_video.py`, register it by name, and implement `embed_blocks`, which maps a
batch of uint8 clips of shape (n, frames, 224, 224, 3) to an array (n, dim). Decoding, blocks, window assembly, the time
budget and unreadable-video handling come from the base class.

```python
@register("encoder", "my_video_encoder")
class MyVideoEncoder(BlockVideoEncoder):
    dim, frames, batch = 256, 8, 16
    def load(self):
        if not hasattr(self, "model"):
            ...                                  # load weights once; self.device = _device()
    def embed_blocks(self, clips):
        ...                                      # (n, frames, 224, 224, 3) uint8 -> (n, dim) float32
```

Then add the name under `encoders:` in the configuration, and optionally pair it with an audio encoder under `fusions:`.
Keep `block_s` equal to the hop. Matchers are added as in INSTRUCTIONS.md Section 10.

## 11. References

Additional to INSTRUCTIONS.md Section 11:

- Tong, Song, Wang, Wang. *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video
  Pre-Training.* NeurIPS 2022. Checkpoint: `MCG-NJU/videomae-base-finetuned-kinetics` (CC-BY-NC-4.0).
- Ni et al. *Expanding Language-Image Pretrained Models for General Video Recognition* (X-CLIP). ECCV 2022. Checkpoint:
  `microsoft/xclip-base-patch32` (MIT).
- Kay et al. *The Kinetics Human Action Video Dataset.* 2017.
- Apple PSI prototype and the code facts in Section 6.4: `apple_psi_prototype/apple_psi/server.py` (`ingest`) and
  `client.py` (`create_real_voucher`).
