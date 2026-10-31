# Night smoke_gpu1_retry: feasibility

- Plan version 0.8; code d886953+dirty (copy in runs/smoke_gpu1_retry/code/); started 2026-10-28 21:54, finished 2026-10-28 21:56 (1.4 minutes).
- Smoke test on tiny subsets: the rows below only show that each check can be computed.

## 1. Did each stage run?

| Stage | Status | Minutes | Notes |
|---|---|---|---|
| data | ok | 0.0 |  |
| probe | ok | 0.1 |  |
| encoder:vggish | ok | 0.1 | 792 windows |
| encoder:clip_frames | ok | 0.1 | 420 windows |
| encoder:hubert | ok | 0.1 | 792 windows |
| encoder:mfcc_stats | ok | 0.1 | 792 windows |
| encoder:clap | ok | 0.3 | 792 windows |
| encoder:ast | ok | 0.5 | 792 windows |
| evaluate | ok | 0.0 |  |
| datasets | ok | 0.0 |  |
| apple_psi | ok | 0.2 |  |
| report | ok | 0.0 |  |

- Timing probe, ms per window: mfcc_stats 11.5, vggish 7.7, hubert 10.5, ast 38.2, clap 22.3, clip_frames 9.6. Estimated extraction: 0.0 h for a budget of 0.2 h; share of benign windows kept: 100%.

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
| event recall at FMR 1% (cosine, home) | 5 encoders | 3 validation events; 95% CI half-width median 35.4, max 50.0 points; 4 encoders detected every event, where the bootstrap CI collapses to one point, so the exact binomial interval is used (29.2 to 100.0 for 3 of 3 events) |
| event recall at FMR 0.1% | 5 encoders | 95% CI half-width median 35.4 points |
| measured FMR within 1.5 times the target (PLAN §7.3) | yes | exceeded in 34 of 125 runs at the 1% target and 34 of 125 runs at the 0.1% target; at 1%: home/ast/lsh (k = 12, L = 8), home/ast/lsh (k = 8, L = 8), home/clap/lsh (k = 12, L = 16), home/hubert/cosine_top1, home/hubert/hamming_sign, home/hubert/l2_raw, home/hubert/lsh (k = 12, L = 16), home/hubert/lsh (k = 16, L = 16), home/mfcc_stats/logreg, home/vggish/cosine_top1, home/vggish/cosine_top5, home/vggish/l2_raw, and 22 more (tables/runs.csv) |
| LSH threshold that meets the target (m <= L) | yes | not reachable in 1 of 72 LSH runs at FMR 1% and 2 of 72 at FMR 0.1% (calibration needed m > L) |
| measured FMR on val_check (1% target) | yes | 81 benign windows; values 0.00% to 3.70%; 95% CI half-width (by session) up to 3.75 points |
| measured FMR at the 0.1% target | yes | 0.1 expected false matches in val_check (at least 3 are needed for a usable estimate) |
| AUC | yes | 0.564 to 1.000 |
| FMR on TV mix / SINS watching TV (1% target) | yes | TV mix 0.00% to 75.00%; watching TV 0.00% to 16.67% |
| event recall at FMR 1% on the XD track (benign windows from normal XD videos) | 7 embeddings | cosine top-1 0.0% to 100.0% (AUC 0.350 to 0.984) |
| classifier gap on the home track (logreg minus cosine), FMR 1% | 5 encoders | -66.7 to 33.3 points |
| discretization gap on the home track (cosine minus best LSH), FMR 1% | 5 encoders | 0.0 to 33.3 points |
| Apple PSI vs plaintext LSH | yes | equivalence 100.0% on 121 windows (11 plaintext matches); cuckoo drops 0 of 181 items; 0 disagreements, 0 explained by drops; voucher 412 bytes; 7.99 ms per voucher (robot), 2.54 ms (server); setup 0.67 s |
| runtime per stage | yes | section 1 |

## 4. Go / no-go (PLAN §9.2)

| Criterion | Met? | Value |
|---|---|---|
| metrics: recall at FMR 1% has a 95% CI no wider than +/-10 points | no | max half-width 50.0 points |
| metrics: val_check holds at least 3,000 benign windows | no | 81 windows |
| pipeline: at least 3 audio encoders completed | yes | ast, clap, hubert, mfcc_stats, vggish |
| pipeline: every similarity family ran | yes | all ran |
| pipeline: Apple PSI decisions match plaintext LSH, except cuckoo drops | yes | 0 disagreements, 0 explained by drops |
| promising: an encoder clearly exceeds MFCC at FMR 1% (read as: higher, and 95% CIs do not overlap) | no | best vggish: +0.0 points, CIs overlap |
| promising: an LSH setting has a small discretization gap (read as: at most 5 points, best encoder) | yes | vggish: 0.0 points at FMR 1% (k = 8, L = 8), both at 100.0%, a ceiling; 0.0 points at FMR 0.1% |

- Classifier gap for the best encoder (vggish) on the home track, at FMR 0.1% (both reached 100.0% at FMR 1%): logistic regression 100.0% versus cosine 100.0% (+0.0 points); the classifier's measured FMR (4.94%) exceeded the tolerance. A large positive gap points to the matching rule as the bottleneck; a small gap with low recall points to the encoder.

## 5. What the full run needs

- Event recall within +/-10 points needs at most about 97 validation events (at a recall of 50%, where the binomial CI is widest) and about 86 at the lowest recall observed here (66.7%, hubert); this run had 3. Ranking the encoders that sit near 100% would need many more events or a harder benign set, such as the XD track. The XD test set has 137 fighting or abuse videos, of which the locked test keeps 40% of the source groups, so more events need the XD training videos (weak labels) or a second violent source.
- FMR = 0.1% needs at least 3,000 benign val_check windows for 3 expected false matches, and about 15,367 for a CI of about +/-50% of that value; this run had 81. Windows within one SINS session are correlated (4 microphone nodes record the same moment), so the effective number is smaller.
- False matches cluster by session. For vggish at FMR 1%, the 95% CI of the measured FMR (3.70%), resampled by session, is +/-3.18 points, 0.8 times the binomial CI over windows (+/-4.11); this design effect of about 1 means that val_check (81 windows from 13 sessions) behaves like about 136 independent windows. The SINS archives used here already cover 52 of the 266 sessions of the development set, so more archives add windows to the same sessions and narrow this CI less than the window count suggests; more homes or sessions would.
- One SINS archive holds about 3,189 ten-second files, which yield about 2,153 val_check windows with this split. By window count alone, about 8 of the 23 archives would give the +/-50% CI at FMR 0.1%; the session clustering above makes this estimate optimistic.
- On the XD track, val_calib holds 242 windows from only 2 normal videos, so the 1% threshold rests on its top 2 windows, which can come from very few videos. More normal videos, rather than more windows per video, would stabilize that threshold and its measured FMR.

## 6. Check of comparison.md (PLAN §12.1)

- Passed: 355 numbers in 59 bullets and table rows checked against facts.json; run ids, em-dashes, banned words, and section structure checked.
