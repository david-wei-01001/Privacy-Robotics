# Night smoke_gpu1: feasibility

- Plan version 0.8; code d886953+dirty (copy in runs/smoke_gpu1/code/); started 2026-10-28 21:09, finished 2026-10-28 21:50 (41.0 minutes).
- Smoke test on tiny subsets: the rows below only show that each check can be computed.

## 1. Did each stage run?

| Stage | Status | Minutes | Notes |
|---|---|---|---|
| data | ok | 39.6 |  |
| probe | ok | 0.4 |  |
| encoder:vggish | failed | 0.1 | FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data2/u/davidwei/Privacy-Robotics/data/cache/emb/e832a73a58_vggish-v1.npy' |
| encoder:clip_frames | failed | 0.1 | FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data2/u/davidwei/Privacy-Robotics/data/cache/emb/e832a73a58_clip_frames-keyframes-v1.npy' |
| encoder:mfcc_stats | failed | 0.1 | FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data2/u/davidwei/Privacy-Robotics/data/cache/emb/e832a73a58_mfcc_stats-v1.npy' |
| encoder:hubert | failed | 0.1 | FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data2/u/davidwei/Privacy-Robotics/data/cache/emb/e832a73a58_hubert-L9-v1.npy' |
| encoder:clap | failed | 0.3 | FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data2/u/davidwei/Privacy-Robotics/data/cache/emb/e832a73a58_clap-v1.npy' |
| encoder:ast | failed | 0.5 | FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data2/u/davidwei/Privacy-Robotics/data/cache/emb/e832a73a58_ast-v1.npy' |
| evaluate | ok | 0.0 |  |
| datasets | ok | 0.0 |  |
| apple_psi | ok | 0.0 |  |
| report | ok | 0.0 |  |

- Timing probe, ms per window: mfcc_stats 10.2, vggish 7.4, hubert 10.2, ast 38.0, clap 22.4, clip_frames 9.6. Estimated extraction: 0.0 h for a budget of 0.2 h; share of benign windows kept: 100%.

## 2. Data at this scale

| Set (dataset/split/label) | Windows | Groups | Events |
|---|---|---|---|
| sins/benign_train/benign | 108 | 19 | 36 |
| sins/val_calib/benign | 171 | 20 | 57 |
| sins/val_check/benign | 81 | 13 | 27 |
| tv_mix/val_check/hard_negative | 12 | 8 | 12 |
| xd/reference/violent | 62 | 2 | 4 |
| xd/val_calib/benign | 242 | 2 | 2 |
| xd/val_check/benign | 76 | 2 | 2 |
| xd/val_violent/violent | 40 | 3 | 3 |

## 3. Are the metrics computable, and how wide are the CIs?

| Metric | Computed | At this scale |
|---|---|---|
| classifier gap on the home track (logreg minus cosine) | no | |
| discretization gap on the home track (cosine minus best LSH) | no | |
| runtime per stage | yes | section 1 |

## 4. Go / no-go (PLAN §9.2)

| Criterion | Met? | Value |
|---|---|---|

## 5. What the full run needs


## 6. Check of comparison.md (PLAN §12.1)

- Problems found: 0 numbers in 0 bullets and table rows checked against facts.json; run ids, em-dashes, banned words, and section structure checked.
- expected 3 sections, found 0
