# Multimodal PSI: Comparison of Datasets, Embeddings, and Matching (pilot)

- Night gpu1; plan version 0.8; code d886953+dirty (copy in runs/gpu1/code/). Every number below comes from `facts.json`, and each bullet names the runs it comes from.

## 1. Datasets

- To compare datasets, we fixed the encoder (vggish, the encoder with the highest point estimate of event recall at an FMR of 1% in Section 2) and the matcher (cosine similarity to the nearest reference, "cosine top-1"), and varied only the source of the reference events, of the violent test events, and of the benign windows that set and measure the false-match rate (FMR). Recall is counted per validation event, and the area under the receiver operating characteristic curve (AUC) is computed over windows. Parentheses give the 95% CI, a bootstrap confidence interval (CI) over validation events; when every event or no event was detected, the bootstrap CI collapses to a single point, so we give the exact binomial (Clopper-Pearson) interval instead, marked "exact". When two settings both detected every event at the 1% target, we compare them at 0.1%.

| Setting | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) | Measured FMR (1% target) | AUC | Runs |
|---|---|---|---|---|---|
| references and test events from all XD sources; benign windows from SINS | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 1.63% | 0.990 | gpu1-0251, gpu1-0252 |
| all references; test events from movies only | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 1.63% | 0.990 | gpu1-0253, gpu1-0254 |
| references from movies only; test events from movies | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 1.63% | 0.990 | gpu1-0255, gpu1-0256 |
| benign windows (calibration and measurement) from normal XD videos instead of SINS | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 2.63% | 0.849 | gpu1-0257, gpu1-0258 |
| threshold calibrated on SINS, FMR measured on normal XD videos | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 35.53% | 0.849 | gpu1-0259, gpu1-0260 |
| threshold calibrated on normal XD videos, FMR measured on SINS | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.06% | 0.990 | gpu1-0261, gpu1-0262 |
| SINS "watching TV" windows only (threshold from SINS) | n/a | n/a | 6.78% | n/a | gpu1-0251 |
| TV mix: violent audio mixed into SINS at 0 dB | n/a | n/a | 69.57% | n/a | gpu1-0251 |
| TV mix: violent audio mixed into SINS at -10 dB | n/a | n/a | 41.18% | n/a | gpu1-0251 |

- A threshold calibrated on home recordings (SINS) for an FMR target of 1% produced a measured FMR of 1.63% on held-out SINS windows (run gpu1-0251), compared with 35.53% on windows from normal XD-Violence videos (run gpu1-0259). Consequently, a threshold may not transfer between recording conditions, and benign calibration data should potentially come from the deployment setting.
- Additionally, a threshold calibrated on normal XD-Violence windows produced a measured FMR of 0.06% on SINS windows and an event recall of 100.0% (run gpu1-0261), compared with 100.0% under the SINS threshold (run gpu1-0251).
- Replacing SINS with normal XD-Violence videos as the benign set, for both calibration and measurement, changed event recall at an FMR of 1% from 100.0% to 100.0% (95% CI 29.2 to 100.0, exact) and AUC from 0.990 to 0.849 (runs gpu1-0251, gpu1-0257). The 95% CIs overlapped, so the source of the benign windows did not appear to change recall clearly.
- On the benign side, the FMR on SINS "watching TV" windows reached 6.78% at the 1% target, 4.2 times the 1.63% measured on all held-out SINS windows (run gpu1-0251).
- Similarly, violent XD-Violence audio mixed into home recordings, as if played on a television, produced FMRs of 69.57% at 0 dB and 41.18% at -10 dB (run gpu1-0251), compared with 1.63% on unmixed SINS windows. We term these "common-cause false matches": benign windows that match because they contain violent sound from another source.
- Among home activities, the highest FMRs at the 1% target occurred for 1) watching tv (6.78%), 2) social activity (2.86%), and 3) dishwashing (1.21%) (run gpu1-0251).
- The setting "events_youtube" was skipped (too few windows in violent).
- The setting "ref_movie__events_youtube" was skipped (too few windows in violent).

## 2. Embeddings

- To compare embeddings, we fixed the matcher (cosine top-1) and the threshold calibration (benign SINS windows, FMR targets of 1% and 0.1%), and varied only the encoder.

| Encoder | Modality | Neural | Dimension | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) | AUC | Measured FMR (1% target) | Runs |
|---|---|---|---|---|---|---|---|---|
| vggish | audio | yes | 128 | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.990 | 1.63% | gpu1-0001, gpu1-0002 |
| mfcc_stats | audio | no | 84 | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.999 | 1.09% | gpu1-0023, gpu1-0024 |
| clap | audio | yes | 512 | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.986 | 0.46% | gpu1-0067, gpu1-0068 |
| ast | audio | yes | 768 | 100.0% (29.2 to 100.0, exact) | 100.0% (29.2 to 100.0, exact) | 0.997 | 1.50% | gpu1-0089, gpu1-0090 |
| hubert | audio | yes | 768 | 66.7% (0.0 to 100.0) | 33.3% (0.0 to 100.0) | 0.695 | 0.78% | gpu1-0045, gpu1-0046 |

- vggish reached the highest event recall at an FMR of 1% (100.0%, 95% CI 29.2 to 100.0, exact; run gpu1-0001), compared with 66.7% for hubert (95% CI 0.0 to 100.0; run gpu1-0045).
- Compared with the non-neural baseline, statistics of mel-frequency cepstral coefficients (MFCCs) at 100.0% (run gpu1-0023), vggish exceeded it by 0.0 points (run gpu1-0001). However, the two 95% CIs overlapped, so the difference may reflect sampling variation at this scale.
- At the stricter FMR target of 0.1%, vggish reached the highest event recall (100.0%; run gpu1-0002), the same encoder as at 1%.
- At this scale, vggish, mfcc_stats, clap, ast, and hubert could not be ranked, because their 95% CIs overlapped at both FMR targets. Specifically, event recall at FMRs of 1% and 0.1% was 100.0% and 100.0% for vggish (runs gpu1-0001, gpu1-0002), 100.0% and 100.0% for mfcc_stats (runs gpu1-0023, gpu1-0024), 100.0% and 100.0% for clap (runs gpu1-0067, gpu1-0068), 100.0% and 100.0% for ast (runs gpu1-0089, gpu1-0090), and 66.7% and 33.3% for hubert (runs gpu1-0045, gpu1-0046).
- The best sound-event encoder and the speech encoder did not differ clearly at this scale: vggish reached 100.0% at an FMR of 1% (95% CI 29.2 to 100.0, exact; run gpu1-0001), compared with 66.7% for hubert (95% CI 0.0 to 100.0; run gpu1-0045), and 100.0% compared with 33.3% at 0.1% (95% CIs 29.2 to 100.0, exact and 0.0 to 100.0; runs gpu1-0002, gpu1-0046), and the 95% CIs overlapped at 0.1%.
- A local logistic-regression classifier on the same embeddings, a non-private reference point, reached 100.0% for vggish at an FMR of 0.1% (run gpu1-0010), compared with 100.0% for cosine matching (run gpu1-0002); at 1%, both reached 100.0% (runs gpu1-0009, gpu1-0001). We term this difference of 0.0 points the "classifier gap". Because the classifier did not exceed cosine matching, the encoder rather than the matching rule appears to limit recall. However, the measured FMR of the classifier run (0.54%) exceeded 1.5 times the 0.1% target, so its recall is not directly comparable.

| Encoder (XD-Violence track: benign windows from normal XD videos) | Cosine top-1: event recall at FMR 1% (95% CI) | AUC | Measured FMR | Logistic regression: event recall at FMR 1% (measured FMR) | Runs |
|---|---|---|---|---|---|
| vggish | 100.0% (29.2 to 100.0, exact) | 0.854 | 2.63% | n/a | gpu1-0111 |
| mfcc_stats | 100.0% (29.2 to 100.0, exact) | 0.988 | 1.32% | n/a | gpu1-0131 |
| clap | 100.0% (29.2 to 100.0, exact) | 0.862 | 1.32% | n/a | gpu1-0171 |
| clip_frames | 100.0% (29.2 to 100.0, exact) | 0.897 | 3.95% | n/a | gpu1-0211 |
| ast | 100.0% (29.2 to 100.0, exact) | 0.913 | 0.00% | n/a | gpu1-0191 |
| fusion:vggish+clip_frames | 100.0% (29.2 to 100.0, exact) | 0.889 | 0.00% | n/a | gpu1-0231 |
| hubert | 33.3% (0.0 to 100.0) | 0.350 | 0.00% | n/a | gpu1-0151 |

- When benign windows came from normal XD videos, every audio embedding separated violent from benign windows less well than on the home track: with cosine top-1, event recall at an FMR of 1% ranged from 33.3% (hubert; run gpu1-0151) to 100.0% (fusion:vggish+clip_frames; run gpu1-0231), compared with 66.7% to 100.0% on the home track (runs gpu1-0045, gpu1-0001), and AUC ranged from 0.350 to 0.988 (runs gpu1-0151, gpu1-0131). On this track, embeddings were centered on normal XD training windows rather than on SINS windows, so its values differ from the corresponding setting in Section 1.
- Concatenating vggish with frame embeddings from Contrastive Language-Image Pre-training (CLIP) changed AUC on the XD track from 0.854 with audio only (run gpu1-0111) and 0.897 with video only (run gpu1-0211) to 0.889 (run gpu1-0231), and event recall at an FMR of 1% with cosine top-1 from 100.0% and 100.0% to 100.0%.

## 3. Matching

- To compare matching methods, we fixed the encoder (vggish) and the threshold calibration (benign SINS windows), and varied only the matcher. For locality-sensitive hashing (LSH), k is the number of bits per band, L the number of bands, and m the number of matching bands that calibration requires. The best LSH setting is the one with the highest event recall on the same validation events, so its recall is slightly optimistic.

| Matcher | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% | Measured FMR (1% target) | AUC | Threshold | Runs |
|---|---|---|---|---|---|---|
| cosine_top1 | 100.0% (29.2 to 100.0, exact) | 100.0% | 1.63% | 0.990 | calibrated | gpu1-0001, gpu1-0002 |
| cosine_top5 | 100.0% (29.2 to 100.0, exact) | 100.0% | 1.81% | 0.991 | calibrated | gpu1-0003, gpu1-0004 |
| l2_raw | 100.0% (29.2 to 100.0, exact) | 100.0% | 2.43% | 0.923 | calibrated | gpu1-0005, gpu1-0006 |
| hamming_sign | 100.0% (29.2 to 100.0, exact) | 100.0% | 1.07% | 0.989 | calibrated | gpu1-0007, gpu1-0008 |
| logreg | 100.0% (29.2 to 100.0, exact) | 100.0% | 2.83% | 0.998 | calibrated | gpu1-0009, gpu1-0010 |
| lsh (k = 8, L = 8) | 100.0% (29.2 to 100.0, exact) | 66.7% | 0.28% | 0.918 | m = 5 | gpu1-0011, gpu1-0012 |
| lsh (k = 8, L = 16) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.56% | 0.971 | m = 8 | gpu1-0013, gpu1-0014 |
| lsh (k = 12, L = 8) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.39% | 0.911 | m = 3 | gpu1-0015, gpu1-0016 |
| lsh (k = 12, L = 16) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.31% | 0.976 | m = 4 | gpu1-0017, gpu1-0018 |
| lsh (k = 16, L = 8) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.02% | 0.717 | m = 2 | gpu1-0019, gpu1-0020 |
| lsh (k = 16, L = 16) | 100.0% (29.2 to 100.0, exact) | 100.0% | 0.13% | 0.791 | m = 2 | gpu1-0021, gpu1-0022 |

- Converting embeddings into LSH codes for exact matching changed event recall from 100.0% with cosine similarity (run gpu1-0001) to 100.0% with the best LSH setting (k = 16, L = 8, m = 2; run gpu1-0019), a difference of 0.0 points that we term the "discretization gap". Because both detected every event, the 1% target cannot reveal a gap here; the 0.1% target below can.
- Across the LSH settings in the table, event recall at an FMR of 1% ranged from 100.0% (k = 8, L = 8; run gpu1-0011) to 100.0% (k = 16, L = 16; run gpu1-0021).
- At the stricter FMR target of 0.1%, the "discretization gap" was 0.0 points: event recall reached 100.0% with cosine similarity (95% CI 29.2 to 100.0, exact; run gpu1-0002), compared with 100.0% with the best LSH setting (k = 12, L = 8, m = 4; 95% CI 29.2 to 100.0, exact; run gpu1-0016). The 95% CIs overlapped, so this gap remains uncertain at this scale.
- Hamming distance on the sign bits of a principal component projection reached 100.0% (run gpu1-0007), compared with 100.0% for cosine similarity (run gpu1-0001), which lies between cosine similarity and LSH exact matching.
- Averaging the similarity to the five nearest references (cosine top-5) reached 100.0% (run gpu1-0003), compared with 100.0% for the nearest reference alone (run gpu1-0001).
- Similarly, the L2 distance to the nearest reference on centered, unnormalized embeddings reached 100.0% (run gpu1-0005), compared with 100.0% for cosine similarity (run gpu1-0001).
- On the XD-Violence track, no LSH setting within the FMR tolerance exceeded cosine top-1 for any of the 7 embeddings; for example, vggish reached 100.0% with LSH (k = 12, L = 16, m = 6; run gpu1-0125), compared with 100.0% with cosine top-1 (run gpu1-0111).
- The Apple-style private set intersection (PSI) protocol reproduced the plaintext LSH decisions on 100.0% of 150 windows, 7 of which matched in plaintext (k = 16, L = 8, m = 2; runs gpu1-0019 and gpu1-0263).
