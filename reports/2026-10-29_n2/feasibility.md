# Night 2026-10-29_n2: feasibility

- Plan version 0.8; code e6c2dec+dirty (copy in runs/2026-10-29_n2/code/); started 2026-10-29 16:59, finished 2026-10-29 18:15 (76.0 minutes).

## 1. Did each stage run?

| Stage | Status | Minutes | Notes |
|---|---|---|---|
| data | ok | 0.3 |  |
| probe | ok | 0.6 |  |
| encoder:vggish | ok | 2.4 | 32427 windows |
| encoder:mfcc_stats | ok | 3.9 | 32427 windows |
| encoder:hubert | ok | 4.4 | 32427 windows |
| encoder:clip_frames | ok | 1.3 | 12993 windows |
| encoder:clap | ok | 9.5 | 32427 windows |
| encoder:xclip_b32 | ok | 5.5 | 12993 windows |
| encoder:motion_stats | ok | 7.3 | 12993 windows |
| encoder:ast | ok | 18.8 | 32427 windows |
| encoder:videomae_k400 | ok | 13.7 | 12993 windows |
| evaluate | ok | 0.7 |  |
| datasets | ok | 0.0 |  |
| apple_psi | ok | 7.6 |  |
| report | ok | 0.0 |  |

- Timing probe, ms per window: mfcc_stats 9.2, vggish 6.6, hubert 9.9, ast 36.3, clap 20.3, clip_frames 12.7, motion_stats 31.5, videomae_k400 60.7, xclip_b32 23.6. Estimated extraction: 1.2 h for a budget of 10.0 h; share of benign windows kept: 100%.

## 2. Data at this scale

| Set (dataset/split/label) | Windows | Groups | Events |
|---|---|---|---|
| sins/benign_train/benign | 5883 | 59 | 1961 |
| sins/val_calib/benign | 7851 | 82 | 2617 |
| sins/val_check/benign | 5400 | 71 | 1800 |
| tv_mix/val_check/hard_negative | 300 | 47 | 300 |
| xd/benign_train/benign | 3691 | 35 | 52 |
| xd/reference/violent | 912 | 29 | 90 |
| xd/val_calib/benign | 4030 | 38 | 52 |
| xd/val_check/benign | 3668 | 38 | 61 |
| xd/val_violent/violent | 692 | 24 | 67 |

## 3. Are the metrics computable, and how wide are the CIs?

| Metric | Computed | At this scale |
|---|---|---|
| event recall at FMR 1% (cosine, home) | 5 encoders | 67 validation events; 95% CI half-width median 3.7, max 9.0 points; 1 encoder detected every event, where the bootstrap CI collapses to one point, so the exact binomial interval is used (94.6 to 100.0 for 67 of 67 events) |
| event recall at FMR 0.1% | 5 encoders | 95% CI half-width median 5.2 points |
| measured FMR within 1.5 times the target (PLAN §7.3) | yes | exceeded in 16 of 522 runs at the 1% target and 49 of 522 runs at the 0.1% target; at 1%: home/ast/logreg, home/clap/cosine_top5, home/clap/logreg, home/clap/lsh (k = 12, L = 32, seed = 2), home/clap/lsh (k = 20, L = 32, seed = 2), home/mfcc_stats/logreg, home/vggish/cosine_top5, home/vggish/l2_raw, home/vggish/logreg, xd/clip_frames/lsh (k = 16, L = 16, seed = 2), xd/fusion:ast+videomae_k400/logreg, xd/mfcc_stats/logreg, and 4 more (tables/runs.csv) |
| LSH threshold that meets the target (m <= L) | yes | not reachable in 43 of 432 LSH runs at FMR 1% and 75 of 432 at FMR 0.1% (calibration needed m > L) |
| measured FMR on val_check (1% target) | yes | 5400 benign windows; values 0.70% to 1.50%; 95% CI half-width (by session) up to 1.85 points |
| measured FMR at the 0.1% target | yes | 5.4 expected false matches in val_check (at least 3 are needed for a usable estimate) |
| AUC | yes | 0.912 to 0.996 |
| FMR on TV mix / SINS watching TV (1% target) | yes | TV mix 17.00% to 61.00%; watching TV 0.00% to 6.69% |
| event recall at FMR 1% on the XD track (benign windows from normal XD videos) | 13 embeddings | cosine top-1 0.0% to 23.9% (AUC 0.510 to 0.940); logistic regression 26.9% to 94.0% (runs within the FMR tolerance) |
| classifier gap on the home track (logreg minus cosine), FMR 1% | 5 encoders | 0.0 to 16.4 points |
| discretization gap on the home track (cosine minus best LSH), FMR 1% | 5 encoders | -3.0 to 17.9 points |
| Apple PSI vs plaintext LSH | yes | equivalence 100.0% on 150 windows (65 plaintext matches); cuckoo drops 0 of 18134 items; 0 disagreements, 0 explained by drops; voucher 412 bytes; 8.13 ms per voucher (robot), 2.58 ms (server); setup 68.26 s |
| runtime per stage | yes | section 1 |

## 4. Go / no-go (PLAN §9.2)

| Criterion | Met? | Value |
|---|---|---|
| metrics: recall at FMR 1% has a 95% CI no wider than +/-10 points | yes | max half-width 9.0 points |
| metrics: val_check holds at least 3,000 benign windows | yes | 5400 windows |
| pipeline: at least 3 audio encoders completed | yes | ast, clap, hubert, mfcc_stats, vggish |
| pipeline: every similarity family ran | yes | all ran |
| pipeline: Apple PSI decisions match plaintext LSH, except cuckoo drops | yes | 0 disagreements, 0 explained by drops |
| promising: an encoder clearly exceeds MFCC at FMR 1% (read as: higher, and 95% CIs do not overlap) | yes | best clap: +16.4 points, CIs do not overlap |
| promising: an LSH setting has a small discretization gap (read as: at most 5 points, best encoder) | yes | clap: 0.0 points at FMR 1% (k = 20, L = 32, seed = 1), both at 100.0%, a ceiling; -3.0 points at FMR 0.1% |

- Classifier gap for the best encoder (clap) on the home track, at FMR 0.1% (both reached 100.0% at FMR 1%): logistic regression 100.0% versus cosine 95.5% (+4.5 points). A large positive gap points to the matching rule as the bottleneck; a small gap with low recall points to the encoder.
- On the XD track, logistic regression exceeded cosine top-1 by 26.9 to 94.0 points at FMR 1% (9 embeddings within the FMR tolerance), so within one recording domain the matching rule rather than the encoder appears to be the bottleneck.

## 5. What the full run needs

- Event recall within +/-10 points needs at most about 97 validation events (at a recall of 50%, where the binomial CI is widest) and about 53 at the lowest recall observed here (83.6%, mfcc_stats); this run had 67. Ranking the encoders that sit near 100% would need many more events or a harder benign set, such as the XD track. The XD test set has 137 fighting or abuse videos, of which the locked test keeps 40% of the source groups, so more events need the XD training videos (weak labels) or a second violent source.
- FMR = 0.1% needs at least 3,000 benign val_check windows for 3 expected false matches, and about 15,367 for a CI of about +/-50% of that value; this run had 5,400. Windows within one SINS session are correlated (4 microphone nodes record the same moment), so the effective number is smaller.
- False matches cluster by session. For clap at FMR 1%, the 95% CI of the measured FMR (1.31%), resampled by session, is +/-1.85 points, 6.1 times the binomial CI over windows (+/-0.30); this design effect of about 37 means that val_check (5,400 windows from 71 sessions) behaves like about 145 independent windows. The SINS archives used here already cover 212 of the 266 sessions of the development set, so more archives add windows to the same sessions and narrow this CI less than the window count suggests; more homes or sessions would.
- One SINS archive holds about 3,189 ten-second files, which yield about 2,700 val_check windows with this split. By window count alone, about 6 of the 23 archives would give the +/-50% CI at FMR 0.1%; the session clustering above makes this estimate optimistic.
- On the XD track, val_calib holds 4,030 windows from only 52 normal videos, so the 1% threshold rests on its top 40 windows, which can come from very few videos. More normal videos, rather than more windows per video, would stabilize that threshold and its measured FMR.

## 6. Check of comparison.md (PLAN §12.1)

- Passed: 637 numbers in 86 bullets and table rows checked against facts.json; run ids, em-dashes, banned words, and section structure checked.
