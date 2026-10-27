# Multimodal PSI: Comparison of Datasets, Embeddings, and Matching (pilot)

- Night smoke1; plan version 0.6; code 68a5776+dirty (copy in runs/smoke1/code/). Every number below comes from `facts.json`, and each bullet names the runs it comes from. Reports re-rendered 2026-10-27 03:55 from the run's facts with the report code in runs/smoke1/code/report_rerender.py.
- This is a smoke test on a few videos and SINS files. Its numbers only show that the pipeline runs and should not be read as results.

## 1. Datasets

- To compare datasets, we fixed the encoder (mfcc_stats, the encoder with the highest point estimate of event recall at an FMR of 1% in Section 2) and the matcher (cosine similarity to the nearest reference, "cosine top-1"), and varied only the source of the reference events, of the violent test events, and of the benign windows that set and measure the false-match rate (FMR). Recall is counted per validation event, and the area under the receiver operating characteristic curve (AUC) is computed over windows. Parentheses give the 95% CI, a bootstrap confidence interval (CI) over validation events; when every event or no event was detected, the bootstrap CI collapses to a single point, so we give the exact binomial (Clopper-Pearson) interval instead, marked "exact". When two settings both detected every event at the 1% target, we compare them at 0.1%.

| Setting | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) | Measured FMR (1% target) | AUC | Runs |
|---|---|---|---|---|---|
| references and test events from all XD sources; benign windows from SINS | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.00% | 1.000 | smoke1-0251, smoke1-0252 |
| all references; test events from movies only | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.00% | 1.000 | smoke1-0253, smoke1-0254 |
| references from movies only; test events from movies | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.00% | 1.000 | smoke1-0255, smoke1-0256 |
| benign windows (calibration and measurement) from normal XD videos instead of SINS | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 1.32% | 0.989 | smoke1-0257, smoke1-0258 |
| threshold calibrated on SINS, FMR measured on normal XD videos | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 15.79% | 0.989 | smoke1-0259, smoke1-0260 |
| threshold calibrated on normal XD videos, FMR measured on SINS | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.00% | 1.000 | smoke1-0261, smoke1-0262 |
| SINS "watching TV" windows only (threshold from SINS) | n/a | n/a | 0.00% | n/a | smoke1-0251 |
| TV mix: violent audio mixed into SINS at 0 dB | n/a | n/a | 66.67% | n/a | smoke1-0251 |
| TV mix: violent audio mixed into SINS at -10 dB | n/a | n/a | 33.33% | n/a | smoke1-0251 |

- A threshold calibrated on home recordings (SINS) for an FMR target of 1% produced a measured FMR of 0.00% on held-out SINS windows (run smoke1-0251), compared with 15.79% on windows from normal XD-Violence videos (run smoke1-0259). Consequently, a threshold may not transfer between recording conditions, and benign calibration data should potentially come from the deployment setting.
- Additionally, a threshold calibrated on normal XD-Violence windows produced a measured FMR of 0.00% on SINS windows and an event recall of 100.0% (run smoke1-0261), compared with 100.0% under the SINS threshold (run smoke1-0251).
- Replacing SINS with normal XD-Violence videos as the benign set, for both calibration and measurement, changed event recall at an FMR of 1% from 100.0% to 100.0% (95% CI 29.2 to 100.0, exact) and AUC from 1.000 to 0.989 (runs smoke1-0251, smoke1-0257). The 95% CIs overlapped, so the source of the benign windows did not appear to change recall clearly.
- On the benign side, the FMR on SINS "watching TV" windows (0.00%) did not exceed the FMR on all held-out SINS windows (0.00%) at the 1% target (run smoke1-0251), so television sound in this home did not appear to cause false matches.
- Similarly, violent XD-Violence audio mixed into home recordings, as if played on a television, produced FMRs of 66.67% at 0 dB and 33.33% at -10 dB (run smoke1-0251), compared with 0.00% on unmixed SINS windows. We term these "common-cause false matches": benign windows that match because they contain violent sound from another source.
- The setting "events_youtube" was skipped (too few windows in violent).
- The setting "ref_movie__events_youtube" was skipped (too few windows in violent).

## 2. Embeddings

- To compare embeddings, we fixed the matcher (cosine top-1) and the threshold calibration (benign SINS windows, FMR targets of 1% and 0.1%), and varied only the encoder.

| Encoder | Modality | Neural | Dimension | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) | AUC | Measured FMR (1% target) | Runs |
|---|---|---|---|---|---|---|---|---|
| mfcc_stats | audio | no | 84 | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 1.000 | 0.00% | smoke1-0001, smoke1-0002 |
| clap | audio | yes | 512 | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.992 | 0.00% | smoke1-0023, smoke1-0024 |
| vggish | audio | yes | 128 | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.976 | 3.70% | smoke1-0045, smoke1-0046 |
| ast | audio | yes | 768 | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.999 | 1.23% | smoke1-0089, smoke1-0090 |
| hubert | audio | yes | 768 | 66.7% (0.0 to 100.0) | 66.7% (0.0 to 100.0) | 0.564 | 3.70% | smoke1-0067, smoke1-0068 |

- mfcc_stats reached the highest event recall at an FMR of 1% (100.0%, 95% CI 29.2 to 100.0, exact; run smoke1-0001), compared with 66.7% for hubert (95% CI 0.0 to 100.0; run smoke1-0067).
- The non-neural baseline, statistics of mel-frequency cepstral coefficients (MFCCs), reached the highest recall (run smoke1-0001), which suggests that the neural encoders did not add separation at this scale.
- At the stricter FMR target of 0.1%, mfcc_stats reached the highest event recall (100.0%; run smoke1-0002), the same encoder as at 1%.
- At this scale, mfcc_stats, clap, vggish, ast, and hubert could not be ranked, because their 95% CIs overlapped at both FMR targets. Specifically, event recall at FMRs of 1% and 0.1% was 100.0% and 100.0% for mfcc_stats (runs smoke1-0001, smoke1-0002), 100.0% and 100.0% for clap (runs smoke1-0023, smoke1-0024), 100.0% and 100.0% for vggish (runs smoke1-0045, smoke1-0046), 100.0% and 100.0% for ast (runs smoke1-0089, smoke1-0090), and 66.7% and 66.7% for hubert (runs smoke1-0067, smoke1-0068).
- The best sound-event encoder and the speech encoder did not differ clearly at this scale: clap reached 100.0% at an FMR of 1% (95% CI 29.2 to 100.0, exact; run smoke1-0023), compared with 66.7% for hubert (95% CI 0.0 to 100.0; run smoke1-0067), and 100.0% compared with 66.7% at 0.1% (95% CIs 29.2 to 100.0, exact and 0.0 to 100.0; runs smoke1-0024, smoke1-0068), and the 95% CIs overlapped at 0.1%.
- A local logistic-regression classifier on the same embeddings, a non-private reference point, reached 33.3% for mfcc_stats at an FMR of 1% (run smoke1-0009), compared with 100.0% for cosine matching (run smoke1-0001). We term this difference of -66.7 points the "classifier gap". Because the classifier did not exceed cosine matching, the encoder rather than the matching rule appears to limit recall. The 95% CIs overlapped, so this gap may also reflect sampling variation. However, the measured FMR of the classifier run (3.70%) exceeded 1.5 times the 1% target, so its recall is not directly comparable.

| Encoder (XD-Violence track: benign windows from normal XD videos) | Cosine top-1: event recall at FMR 1% (95% CI) | AUC | Measured FMR | Logistic regression: event recall at FMR 1% (measured FMR) | Runs |
|---|---|---|---|---|---|
| mfcc_stats | 100.0% (29.2 to 100.0, exact) | 0.984 | 2.63% | n/a | smoke1-0111 |
| clap | 100.0% (29.2 to 100.0, exact) | 0.874 | 0.00% | n/a | smoke1-0131 |
| vggish | 100.0% (29.2 to 100.0, exact) | 0.807 | 1.32% | n/a | smoke1-0151 |
| ast | 100.0% (29.2 to 100.0, exact) | 0.899 | 0.00% | n/a | smoke1-0191 |
| clip_frames | 100.0% (29.2 to 100.0, exact) | 0.923 | 6.58% | n/a | smoke1-0211 |
| fusion:mfcc_stats+clip_frames | 100.0% (29.2 to 100.0, exact) | 0.878 | 10.53% | n/a | smoke1-0231 |
| hubert | 0.0% (0.0 to 70.8, exact) | 0.350 | 0.00% | n/a | smoke1-0171 |

- When benign windows came from normal XD videos, every audio embedding separated violent from benign windows less well than on the home track: with cosine top-1, event recall at an FMR of 1% ranged from 0.0% (hubert; run smoke1-0171) to 100.0% (fusion:mfcc_stats+clip_frames; run smoke1-0231), compared with 66.7% to 100.0% on the home track (runs smoke1-0067, smoke1-0001), and AUC ranged from 0.350 to 0.984 (runs smoke1-0171, smoke1-0111). On this track, embeddings were centered on normal XD training windows rather than on SINS windows, so its values differ from the corresponding setting in Section 1.
- Concatenating mfcc_stats with frame embeddings from Contrastive Language-Image Pre-training (CLIP) changed AUC on the XD track from 0.984 with audio only (run smoke1-0111) and 0.923 with video only (run smoke1-0211) to 0.878 (run smoke1-0231), and event recall at an FMR of 1% with cosine top-1 from 100.0% and 100.0% to 100.0%.

## 3. Matching

- To compare matching methods, we fixed the encoder (mfcc_stats) and the threshold calibration (benign SINS windows), and varied only the matcher. For locality-sensitive hashing (LSH), k is the number of bits per band, L the number of bands, and m the number of matching bands that calibration requires. The best LSH setting is the one with the highest event recall on the same validation events, so its recall is slightly optimistic.

| Matcher | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% | Measured FMR (1% target) | AUC | Threshold | Runs |
|---|---|---|---|---|---|---|
| cosine_top1 | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.00% | 1.000 | calibrated | smoke1-0001, smoke1-0002 |
| cosine_top5 | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.00% | 1.000 | calibrated | smoke1-0003, smoke1-0004 |
| l2_raw | 100.0% (29.2 to 100.0, exact) | 100.0% | 1.23% | 0.997 | calibrated | smoke1-0005, smoke1-0006 |
| hamming_sign | 33.3% (0.0 to 100.0) | 33.3% | 0.00% | 0.966 | calibrated | smoke1-0007, smoke1-0008 |
| logreg | 33.3% (0.0 to 100.0) | 0.0% | 3.70% | 0.957 | calibrated | smoke1-0009, smoke1-0010 |
| lsh (k = 8, L = 8) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.00% | 0.993 | m = 8 | smoke1-0011, smoke1-0012 |
| lsh (k = 8, L = 16) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.00% | 1.000 | m = 13 | smoke1-0013, smoke1-0014 |
| lsh (k = 12, L = 8) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.00% | 0.993 | m = 6 | smoke1-0015, smoke1-0016 |
| lsh (k = 12, L = 16) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.00% | 1.000 | m = 10 | smoke1-0017, smoke1-0018 |
| lsh (k = 16, L = 8) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.00% | 1.000 | m = 5 | smoke1-0019, smoke1-0020 |
| lsh (k = 16, L = 16) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.00% | 1.000 | m = 9 | smoke1-0021, smoke1-0022 |

- Converting embeddings into LSH codes for exact matching changed event recall from 100.0% with cosine similarity (run smoke1-0001) to 100.0% with the best LSH setting (k = 8, L = 8, m = 8; run smoke1-0011), a difference of 0.0 points that we term the "discretization gap". Because both detected every event, the 1% target cannot reveal a gap here; the 0.1% target below can.
- Across the LSH settings in the table, event recall at an FMR of 1% ranged from 100.0% (k = 8, L = 8; run smoke1-0011) to 100.0% (k = 16, L = 16; run smoke1-0021).
- At the stricter FMR target of 0.1%, the "discretization gap" was 0.0 points: event recall reached 100.0% with cosine similarity (95% CI 29.2 to 100.0, exact; run smoke1-0002), compared with 100.0% with the best LSH setting (k = 8, L = 8, m = 8; 95% CI 29.2 to 100.0, exact; run smoke1-0012). The 95% CIs overlapped, so this gap remains uncertain at this scale.
- Hamming distance on the sign bits of a principal component projection reached 33.3% (run smoke1-0007), compared with 100.0% for cosine similarity (run smoke1-0001).
- Averaging the similarity to the five nearest references (cosine top-5) reached 100.0% (run smoke1-0003), compared with 100.0% for the nearest reference alone (run smoke1-0001).
- Similarly, the L2 distance to the nearest reference on centered, unnormalized embeddings reached 100.0% (run smoke1-0005), compared with 100.0% for cosine similarity (run smoke1-0001).
- On the XD-Violence track, no LSH setting within the FMR tolerance exceeded cosine top-1 for any of the 7 embeddings; for example, clap reached 100.0% with LSH (k = 8, L = 8, m = 5; run smoke1-0139), compared with 100.0% with cosine top-1 (run smoke1-0131).
- The Apple-style private set intersection (PSI) protocol reproduced the plaintext LSH decisions on 100.0% of 121 windows, 21 of which matched in plaintext (k = 8, L = 8, m = 8; runs smoke1-0011 and smoke1-0263).
