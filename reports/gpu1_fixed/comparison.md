# Multimodal PSI: Comparison of Datasets, Embeddings, and Matching (pilot)

- Night gpu1_fixed; plan version 0.8; code a4be78d+dirty (copy in runs/gpu1_fixed/code/). Every number below comes from `facts.json`, and each bullet names the runs it comes from.

## 1. Datasets

- To compare datasets, we fixed the encoder (clap, the encoder with the highest point estimate of event recall at an FMR of 1% in Section 2) and the matcher (cosine similarity to the nearest reference, "cosine top-1"), and varied only the source of the reference events, of the violent test events, and of the benign windows that set and measure the false-match rate (FMR). Recall is counted per validation event, and the area under the receiver operating characteristic curve (AUC) is computed over windows. Parentheses give the 95% CI, a bootstrap confidence interval (CI) over validation events; when every event or no event was detected, the bootstrap CI collapses to a single point, so we give the exact binomial (Clopper-Pearson) interval instead, marked "exact". When two settings both detected every event at the 1% target, we compare them at 0.1%.

| Setting | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) | Measured FMR (1% target) | AUC | Runs |
|---|---|---|---|---|---|
| references and test events from all XD sources; benign windows from SINS | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 1.31% | 0.993 | gpu1_fixed-0265, gpu1_fixed-0266 |
| all references; test events from movies only | 100.0% (91.0 to 100.0, exact) | 100.0% (91.0 to 100.0, exact) | 1.31% | 0.992 | gpu1_fixed-0267, gpu1_fixed-0268 |
| all references; test events from YouTube only | 100.0% (87.7 to 100.0, exact) | 89.3% (78.6 to 100.0) | 1.31% | 0.996 | gpu1_fixed-0269, gpu1_fixed-0270 |
| references from movies only; test events from movies | 100.0% (91.0 to 100.0, exact) | 100.0% (91.0 to 100.0, exact) | 0.93% | 0.996 | gpu1_fixed-0271, gpu1_fixed-0272 |
| references from movies only; test events from YouTube | 100.0% (87.7 to 100.0, exact) | 82.1% (67.9 to 96.4) | 0.93% | 0.994 | gpu1_fixed-0273, gpu1_fixed-0274 |
| benign windows (calibration and measurement) from normal XD videos instead of SINS | 4.5% (0.0 to 10.4) | 3.0% (0.0 to 7.5) | 0.29% | 0.829 | gpu1_fixed-0275, gpu1_fixed-0276 |
| threshold calibrated on SINS, FMR measured on normal XD videos | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 40.52% | 0.829 | gpu1_fixed-0277, gpu1_fixed-0278 |
| threshold calibrated on normal XD videos, FMR measured on SINS | 4.5% (0.0 to 10.4) | 3.0% (0.0 to 7.5) | 0.00% | 0.993 | gpu1_fixed-0279, gpu1_fixed-0280 |
| SINS "watching TV" windows only (threshold from SINS) | n/a | n/a | 6.59% | n/a | gpu1_fixed-0265 |
| TV mix: violent audio mixed into SINS at 0 dB | n/a | n/a | 65.64% | n/a | gpu1_fixed-0265 |
| TV mix: violent audio mixed into SINS at -10 dB | n/a | n/a | 33.58% | n/a | gpu1_fixed-0265 |

- When the reference set contained only movie events, event recall at an FMR of 0.1% reached 100.0% on movie events (95% CI 91.0 to 100.0, exact, n = 39; run gpu1_fixed-0272), compared with 82.1% on YouTube events (95% CI 67.9 to 96.4, n = 28; run gpu1_fixed-0274); at an FMR of 1%, both reached 100.0% (runs gpu1_fixed-0271, gpu1_fixed-0273). The 95% CIs overlapped, so at this scale the source of the reference clips did not appear to change recall clearly.
- With references from all sources, event recall at an FMR of 0.1% was 100.0% for movie events (run gpu1_fixed-0268) and 89.3% for YouTube events (run gpu1_fixed-0270), compared with 82.1% for YouTube events when the references came from movies only (run gpu1_fixed-0274); at an FMR of 1%, both reached 100.0% (runs gpu1_fixed-0267, gpu1_fixed-0269).
- A threshold calibrated on home recordings (SINS) for an FMR target of 1% produced a measured FMR of 1.31% on held-out SINS windows (run gpu1_fixed-0265), compared with 40.52% on windows from normal XD-Violence videos (run gpu1_fixed-0277). Consequently, a threshold may not transfer between recording conditions, and benign calibration data should potentially come from the deployment setting.
- Additionally, a threshold calibrated on normal XD-Violence windows produced a measured FMR of 0.00% on SINS windows and an event recall of 4.5% (run gpu1_fixed-0279), compared with 100.0% under the SINS threshold (run gpu1_fixed-0265).
- Replacing SINS with normal XD-Violence videos as the benign set, for both calibration and measurement, changed event recall at an FMR of 1% from 100.0% to 4.5% (95% CI 0.0 to 10.4) and AUC from 0.993 to 0.829 (runs gpu1_fixed-0265, gpu1_fixed-0275). Together with the threshold transfer above, this suggests that much of the separation between violent XD-Violence clips and SINS windows may come from the difference between movie or online video sound and home recordings, rather than from the violent content alone.
- On the benign side, the FMR on SINS "watching TV" windows reached 6.59% at the 1% target, 5.0 times the 1.31% measured on all held-out SINS windows (run gpu1_fixed-0265).
- Similarly, violent XD-Violence audio mixed into home recordings, as if played on a television, produced FMRs of 65.64% at 0 dB and 33.58% at -10 dB (run gpu1_fixed-0265), compared with 1.31% on unmixed SINS windows. We term these "common-cause false matches": benign windows that match because they contain violent sound from another source.
- Among home activities, the highest FMRs at the 1% target occurred for 1) watching tv (6.59%), 2) dishwashing (1.82%), and 3) social activity (0.17%) (run gpu1_fixed-0265).

## 2. Embeddings

- To compare embeddings, we fixed the matcher (cosine top-1) and the threshold calibration (benign SINS windows, FMR targets of 1% and 0.1%), and varied only the encoder.

| Encoder | Modality | Neural | Dimension | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) | AUC | Measured FMR (1% target) | Runs |
|---|---|---|---|---|---|---|---|---|
| clap | audio | yes | 512 | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 0.993 | 1.31% | gpu1_fixed-0089, gpu1_fixed-0090 |
| ast | audio | yes | 768 | 98.5% (95.5 to 100.0) | 97.0% (92.5 to 100.0) | 0.995 | 0.70% | gpu1_fixed-0067, gpu1_fixed-0068 |
| vggish | audio | yes | 128 | 97.0% (92.5 to 100.0) | 95.5% (91.0 to 100.0) | 0.996 | 1.50% | gpu1_fixed-0001, gpu1_fixed-0002 |
| hubert | audio | yes | 768 | 86.6% (77.6 to 95.5) | 37.3% (25.4 to 49.3) | 0.912 | 0.94% | gpu1_fixed-0023, gpu1_fixed-0024 |
| mfcc_stats | audio | no | 84 | 83.6% (74.6 to 92.5) | 71.6% (61.2 to 82.1) | 0.974 | 0.76% | gpu1_fixed-0045, gpu1_fixed-0046 |

- clap reached the highest event recall at an FMR of 1% (100.0%, 95% CI 94.6 to 100.0, exact; run gpu1_fixed-0089), compared with 83.6% for mfcc_stats (95% CI 74.6 to 92.5; run gpu1_fixed-0045).
- Compared with the non-neural baseline, statistics of mel-frequency cepstral coefficients (MFCCs) at 83.6% (run gpu1_fixed-0045), clap exceeded it by 16.4 points (run gpu1_fixed-0089). Because the two 95% CIs did not overlap, this difference appears robust at this scale.
- At the stricter FMR target of 0.1%, ast reached the highest event recall (97.0%; run gpu1_fixed-0068), whereas clap reached 95.5% (run gpu1_fixed-0090).
- At this scale, clap, ast, and vggish could not be ranked, because their 95% CIs overlapped at both FMR targets. Specifically, event recall at FMRs of 1% and 0.1% was 100.0% and 95.5% for clap (runs gpu1_fixed-0089, gpu1_fixed-0090), 98.5% and 97.0% for ast (runs gpu1_fixed-0067, gpu1_fixed-0068), and 97.0% and 95.5% for vggish (runs gpu1_fixed-0001, gpu1_fixed-0002).
- Sound-event encoders appear to separate violent and benign windows more reliably than the speech encoder. Specifically, clap reached 100.0% at an FMR of 1% (95% CI 94.6 to 100.0, exact; run gpu1_fixed-0089), compared with 86.6% for hubert (95% CI 77.6 to 95.5; run gpu1_fixed-0023), and 95.5% compared with 37.3% at 0.1% (95% CIs 89.6 to 100.0 and 25.4 to 49.3; runs gpu1_fixed-0090, gpu1_fixed-0024).
- A local logistic-regression classifier on the same embeddings, a non-private reference point, reached 100.0% for clap at an FMR of 0.1% (run gpu1_fixed-0098), compared with 95.5% for cosine matching (run gpu1_fixed-0090); at 1%, both reached 100.0% (runs gpu1_fixed-0097, gpu1_fixed-0089). We term this difference of 4.5 points the "classifier gap". A positive gap suggests that part of the loss may stem from matching against a reference list rather than from the encoder itself. The 95% CIs overlapped, so this gap may also reflect sampling variation.

| Encoder (XD-Violence track: benign windows from normal XD videos) | Cosine top-1: event recall at FMR 1% (95% CI) | AUC | Measured FMR | Logistic regression: event recall at FMR 1% (measured FMR) | Runs |
|---|---|---|---|---|---|
| ast | 16.4% (7.5 to 25.4) | 0.846 | 0.00% | 64.2% (0.68%) | gpu1_fixed-0177, gpu1_fixed-0185 |
| clap | 10.4% (3.0 to 17.9) | 0.882 | 0.39% | 53.7% (0.10%) | gpu1_fixed-0199, gpu1_fixed-0207 |
| mfcc_stats | 10.4% (3.0 to 17.9) | 0.618 | 0.00% | 22.4% (1.16%) | gpu1_fixed-0155, gpu1_fixed-0163 |
| clip_frames | 3.0% (0.0 to 7.5) | 0.832 | 0.48% | 80.6% (1.98%) | gpu1_fixed-0221, gpu1_fixed-0229 |
| vggish | 1.5% (0.0 to 4.5) | 0.841 | 0.05% | 59.7% (0.34%) | gpu1_fixed-0111, gpu1_fixed-0119 |
| hubert | 0.0% (0.0 to 5.4, exact) | 0.453 | 0.00% | 20.9% (0.29%) | gpu1_fixed-0133, gpu1_fixed-0141 |
| fusion:clap+clip_frames | 0.0% (0.0 to 5.4, exact) | 0.932 | 0.00% | 76.1% (0.24%) | gpu1_fixed-0243, gpu1_fixed-0251 |

- When benign windows came from normal XD videos, every audio embedding separated violent from benign windows less well than on the home track: with cosine top-1, event recall at an FMR of 1% ranged from 0.0% (hubert; run gpu1_fixed-0133) to 16.4% (ast; run gpu1_fixed-0177), compared with 83.6% to 100.0% on the home track (runs gpu1_fixed-0045, gpu1_fixed-0089), and AUC ranged from 0.453 to 0.932 (runs gpu1_fixed-0133, gpu1_fixed-0243). On this track, embeddings were centered on normal XD training windows rather than on SINS windows, so its values differ from the corresponding setting in Section 1.
- On the same track, a logistic-regression classifier trained on the same embeddings reached 20.9% (hubert; run gpu1_fixed-0141) to 76.1% (fusion:clap+clip_frames; run gpu1_fixed-0251) at an FMR of 1%, leaving out clip_frames, whose classifier exceeded 1.5 times the FMR target, compared with at most 16.4% for cosine top-1 (run gpu1_fixed-0177). This "classifier gap" suggests that the embeddings may carry information that separates violent from normal videos, and that matching each window against its nearest reference, rather than the encoder, limits recall within one recording domain.
- Concatenating clap with frame embeddings from Contrastive Language-Image Pre-training (CLIP) changed AUC on the XD track from 0.882 with audio only (run gpu1_fixed-0199) and 0.832 with video only (run gpu1_fixed-0221) to 0.932 (run gpu1_fixed-0243), and event recall at an FMR of 1% with cosine top-1 from 10.4% and 3.0% to 0.0%. With logistic regression, the fused embedding reached 76.1%, compared with 53.7% for clap alone (runs gpu1_fixed-0251, gpu1_fixed-0207). This suggests that fusion may improve the bulk of the score distribution more than its strict tail.

## 3. Matching

- To compare matching methods, we fixed the encoder (clap) and the threshold calibration (benign SINS windows), and varied only the matcher. For locality-sensitive hashing (LSH), k is the number of bits per band, L the number of bands, and m the number of matching bands that calibration requires. The best LSH setting is the one with the highest event recall on the same validation events, so its recall is slightly optimistic.

| Matcher | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% | Measured FMR (1% target) | AUC | Threshold | Runs |
|---|---|---|---|---|---|---|
| cosine_top1 | 100.0% (94.6 to 100.0, exact) | 95.5% | 1.31% | 0.993 | calibrated | gpu1_fixed-0089, gpu1_fixed-0090 |
| cosine_top5 | 100.0% (94.6 to 100.0, exact) | 97.0% | 1.65% | 0.994 | calibrated | gpu1_fixed-0091, gpu1_fixed-0092 |
| l2_raw | 98.5% (94.0 to 100.0) | 85.1% | 1.41% | 0.979 | calibrated | gpu1_fixed-0093, gpu1_fixed-0094 |
| hamming_sign | 98.5% (95.5 to 100.0) | 61.2% | 0.93% | 0.990 | calibrated | gpu1_fixed-0095, gpu1_fixed-0096 |
| logreg | 100.0% (94.6 to 100.0, exact) | 100.0% | 2.72% | 0.999 | calibrated | gpu1_fixed-0097, gpu1_fixed-0098 |
| lsh (k = 8, L = 8) | 0.0% (0.0 to 5.4, exact) | 0.0% | 0.00% | 0.961 | none (m > L) | gpu1_fixed-0099, gpu1_fixed-0100 |
| lsh (k = 8, L = 16) | 98.5% (95.5 to 100.0) | 0.0% | 1.28% | 0.982 | m = 15 | gpu1_fixed-0101, gpu1_fixed-0102 |
| lsh (k = 12, L = 8) | 95.5% (89.6 to 100.0) | 65.7% | 0.63% | 0.983 | m = 6 | gpu1_fixed-0103, gpu1_fixed-0104 |
| lsh (k = 12, L = 16) | 98.5% (95.5 to 100.0) | 88.1% | 1.07% | 0.987 | m = 11 | gpu1_fixed-0105, gpu1_fixed-0106 |
| lsh (k = 16, L = 8) | 91.0% (83.6 to 97.0) | 85.1% | 0.43% | 0.972 | m = 4 | gpu1_fixed-0107, gpu1_fixed-0108 |
| lsh (k = 16, L = 16) | 100.0% (94.6 to 100.0, exact) | 83.6% | 0.80% | 0.987 | m = 6 | gpu1_fixed-0109, gpu1_fixed-0110 |

- Converting embeddings into LSH codes for exact matching changed event recall from 100.0% with cosine similarity (run gpu1_fixed-0089) to 100.0% with the best LSH setting (k = 16, L = 16, m = 6; run gpu1_fixed-0109), a difference of 0.0 points that we term the "discretization gap". Because both detected every event, the 1% target cannot reveal a gap here; the 0.1% target below can.
- Across the LSH settings in the table, event recall at an FMR of 1% ranged from 0.0% (k = 8, L = 8; run gpu1_fixed-0099) to 100.0% (k = 16, L = 16; run gpu1_fixed-0109).
- For one setting (k = 8, L = 8, run gpu1_fixed-0099), calibration at the 1% target would have required more matching bands than the hash has (m > L), so no LSH threshold met the target and recall was counted as zero.
- At the stricter FMR target of 0.1%, the "discretization gap" was 7.5 points: event recall reached 95.5% with cosine similarity (95% CI 89.6 to 100.0; run gpu1_fixed-0090), compared with 88.1% with the best LSH setting (k = 12, L = 16, m = 14; 95% CI 80.6 to 95.5; run gpu1_fixed-0106). The 95% CIs overlapped, so this gap remains uncertain at this scale. Specifically, calibration would have required m > L for the settings in runs gpu1_fixed-0100, gpu1_fixed-0102, which suggests that the number of matching bands may be too coarse a score for low false-match rates.
- Hamming distance on the sign bits of a principal component projection reached 98.5% (run gpu1_fixed-0095), compared with 100.0% for cosine similarity (run gpu1_fixed-0089).
- Averaging the similarity to the five nearest references (cosine top-5) reached 100.0% (run gpu1_fixed-0091), compared with 100.0% for the nearest reference alone (run gpu1_fixed-0089).
- Similarly, the L2 distance to the nearest reference on centered, unnormalized embeddings reached 98.5% (run gpu1_fixed-0093), compared with 100.0% for cosine similarity (run gpu1_fixed-0089).
- On the XD-Violence track, the best LSH setting within the FMR tolerance reached a higher event recall at an FMR of 1% than cosine top-1 for 6 of the 7 embeddings. For example, fusion:clap+clip_frames reached 49.3% with LSH (k = 12, L = 16, m = 12; run gpu1_fixed-0259), compared with 0.0% with cosine top-1 (run gpu1_fixed-0243). Because the LSH score counts the bands that match any reference, it may pool evidence across references, whereas cosine top-1 relies on the single nearest one.
- The Apple-style private set intersection (PSI) protocol reproduced the plaintext LSH decisions on 100.0% of 150 windows, 63 of which matched in plaintext (k = 16, L = 16, m = 6; runs gpu1_fixed-0109 and gpu1_fixed-0281).
