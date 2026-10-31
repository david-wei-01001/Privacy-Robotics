# Multimodal PSI: Comparison of Datasets, Embeddings, and Matching (pilot)

- Night 2026-10-29_n2; plan version 0.8; code e6c2dec+dirty (copy in runs/2026-10-29_n2/code/). Every number below comes from `facts.json`, and each bullet names the runs it comes from.

## 1. Datasets

- To compare datasets, we fixed the encoder (clap, the encoder with the highest point estimate of event recall at an FMR of 1% in Section 2) and the matcher (cosine similarity to the nearest reference, "cosine top-1"), and varied only the source of the reference events, of the violent test events, and of the benign windows that set and measure the false-match rate (FMR). Recall is counted per validation event, and the area under the receiver operating characteristic curve (AUC) is computed over windows. Parentheses give the 95% CI, a bootstrap confidence interval (CI) over validation events; when every event or no event was detected, the bootstrap CI collapses to a single point, so we give the exact binomial (Clopper-Pearson) interval instead, marked "exact". When two settings both detected every event at the 1% target, we compare them at 0.1%.

| Setting | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) | Measured FMR (1% target) | AUC | Runs |
|---|---|---|---|---|---|
| references and test events from all XD sources; benign windows from SINS | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 1.31% | 0.993 | 2026-10-29_n2-1045, 2026-10-29_n2-1046 |
| all references; test events from movies only | 100.0% (91.0 to 100.0, exact) | 100.0% (91.0 to 100.0, exact) | 1.31% | 0.992 | 2026-10-29_n2-1047, 2026-10-29_n2-1048 |
| all references; test events from YouTube only | 100.0% (87.7 to 100.0, exact) | 89.3% (78.6 to 100.0) | 1.31% | 0.996 | 2026-10-29_n2-1049, 2026-10-29_n2-1050 |
| references from movies only; test events from movies | 100.0% (91.0 to 100.0, exact) | 100.0% (91.0 to 100.0, exact) | 0.93% | 0.996 | 2026-10-29_n2-1051, 2026-10-29_n2-1052 |
| references from movies only; test events from YouTube | 100.0% (87.7 to 100.0, exact) | 82.1% (67.9 to 96.4) | 0.93% | 0.994 | 2026-10-29_n2-1053, 2026-10-29_n2-1054 |
| benign windows (calibration and measurement) from normal XD videos instead of SINS | 3.0% (0.0 to 7.5) | 0.0% (0.0 to 5.4, exact) | 0.05% | 0.860 | 2026-10-29_n2-1055, 2026-10-29_n2-1056 |
| threshold calibrated on SINS, FMR measured on normal XD videos | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 34.43% | 0.860 | 2026-10-29_n2-1057, 2026-10-29_n2-1058 |
| threshold calibrated on normal XD videos, FMR measured on SINS | 3.0% (0.0 to 7.5) | 0.0% (0.0 to 5.4, exact) | 0.00% | 0.993 | 2026-10-29_n2-1059, 2026-10-29_n2-1060 |
| SINS "watching TV" windows only (threshold from SINS) | n/a | n/a | 6.59% | n/a | 2026-10-29_n2-1045 |
| TV mix: violent audio mixed into SINS at 0 dB | n/a | n/a | 65.64% | n/a | 2026-10-29_n2-1045 |
| TV mix: violent audio mixed into SINS at -10 dB | n/a | n/a | 33.58% | n/a | 2026-10-29_n2-1045 |

- When the reference set contained only movie events, event recall at an FMR of 0.1% reached 100.0% on movie events (95% CI 91.0 to 100.0, exact, n = 39; run 2026-10-29_n2-1052), compared with 82.1% on YouTube events (95% CI 67.9 to 96.4, n = 28; run 2026-10-29_n2-1054); at an FMR of 1%, both reached 100.0% (runs 2026-10-29_n2-1051, 2026-10-29_n2-1053). The 95% CIs overlapped, so at this scale the source of the reference clips did not appear to change recall clearly.
- With references from all sources, event recall at an FMR of 0.1% was 100.0% for movie events (run 2026-10-29_n2-1048) and 89.3% for YouTube events (run 2026-10-29_n2-1050), compared with 82.1% for YouTube events when the references came from movies only (run 2026-10-29_n2-1054); at an FMR of 1%, both reached 100.0% (runs 2026-10-29_n2-1047, 2026-10-29_n2-1049).
- A threshold calibrated on home recordings (SINS) for an FMR target of 1% produced a measured FMR of 1.31% on held-out SINS windows (run 2026-10-29_n2-1045), compared with 34.43% on windows from normal XD-Violence videos (run 2026-10-29_n2-1057). Consequently, a threshold may not transfer between recording conditions, and benign calibration data should potentially come from the deployment setting.
- Additionally, a threshold calibrated on normal XD-Violence windows produced a measured FMR of 0.00% on SINS windows and an event recall of 3.0% (run 2026-10-29_n2-1059), compared with 100.0% under the SINS threshold (run 2026-10-29_n2-1045).
- Replacing SINS with normal XD-Violence videos as the benign set, for both calibration and measurement, changed event recall at an FMR of 1% from 100.0% to 3.0% (95% CI 0.0 to 7.5) and AUC from 0.993 to 0.860 (runs 2026-10-29_n2-1045, 2026-10-29_n2-1055). Together with the threshold transfer above, this suggests that much of the separation between violent XD-Violence clips and SINS windows may come from the difference between movie or online video sound and home recordings, rather than from the violent content alone.
- On the benign side, the FMR on SINS "watching TV" windows reached 6.59% at the 1% target, 5.0 times the 1.31% measured on all held-out SINS windows (run 2026-10-29_n2-1045).
- Similarly, violent XD-Violence audio mixed into home recordings, as if played on a television, produced FMRs of 65.64% at 0 dB and 33.58% at -10 dB (run 2026-10-29_n2-1045), compared with 1.31% on unmixed SINS windows. We term these "common-cause false matches": benign windows that match because they contain violent sound from another source.
- Among home activities, the highest FMRs at the 1% target occurred for 1) watching tv (6.59%), 2) dishwashing (1.82%), and 3) social activity (0.17%) (run 2026-10-29_n2-1045).

## 2. Embeddings

- To compare embeddings, we fixed the matcher (cosine top-1) and the threshold calibration (benign SINS windows, FMR targets of 1% and 0.1%), and varied only the encoder.

| Encoder | Modality | Neural | Dimension | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% (95% CI) | AUC | Measured FMR (1% target) | Runs |
|---|---|---|---|---|---|---|---|---|
| clap | audio | yes | 512 | 100.0% (94.6 to 100.0, exact) | 95.5% (89.6 to 100.0) | 0.993 | 1.31% | 2026-10-29_n2-0175, 2026-10-29_n2-0176 |
| ast | audio | yes | 768 | 98.5% (95.5 to 100.0) | 97.0% (92.5 to 100.0) | 0.995 | 0.70% | 2026-10-29_n2-0233, 2026-10-29_n2-0234 |
| vggish | audio | yes | 128 | 97.0% (92.5 to 100.0) | 95.5% (91.0 to 100.0) | 0.996 | 1.50% | 2026-10-29_n2-0001, 2026-10-29_n2-0002 |
| hubert | audio | yes | 768 | 86.6% (77.6 to 95.5) | 37.3% (25.4 to 49.3) | 0.912 | 0.94% | 2026-10-29_n2-0117, 2026-10-29_n2-0118 |
| mfcc_stats | audio | no | 84 | 83.6% (74.6 to 92.5) | 71.6% (61.2 to 82.1) | 0.974 | 0.76% | 2026-10-29_n2-0059, 2026-10-29_n2-0060 |

- clap reached the highest event recall at an FMR of 1% (100.0%, 95% CI 94.6 to 100.0, exact; run 2026-10-29_n2-0175), compared with 83.6% for mfcc_stats (95% CI 74.6 to 92.5; run 2026-10-29_n2-0059).
- Compared with the non-neural baseline, statistics of mel-frequency cepstral coefficients (MFCCs) at 83.6% (run 2026-10-29_n2-0059), clap exceeded it by 16.4 points (run 2026-10-29_n2-0175). Because the two 95% CIs did not overlap, this difference appears robust at this scale.
- At the stricter FMR target of 0.1%, ast reached the highest event recall (97.0%; run 2026-10-29_n2-0234), whereas clap reached 95.5% (run 2026-10-29_n2-0176).
- At this scale, clap, ast, and vggish could not be ranked, because their 95% CIs overlapped at both FMR targets. Specifically, event recall at FMRs of 1% and 0.1% was 100.0% and 95.5% for clap (runs 2026-10-29_n2-0175, 2026-10-29_n2-0176), 98.5% and 97.0% for ast (runs 2026-10-29_n2-0233, 2026-10-29_n2-0234), and 97.0% and 95.5% for vggish (runs 2026-10-29_n2-0001, 2026-10-29_n2-0002).
- Sound-event encoders appear to separate violent and benign windows more reliably than the speech encoder. Specifically, clap reached 100.0% at an FMR of 1% (95% CI 94.6 to 100.0, exact; run 2026-10-29_n2-0175), compared with 86.6% for hubert (95% CI 77.6 to 95.5; run 2026-10-29_n2-0117), and 95.5% compared with 37.3% at 0.1% (95% CIs 89.6 to 100.0 and 25.4 to 49.3; runs 2026-10-29_n2-0176, 2026-10-29_n2-0118).
- A local logistic-regression classifier on the same embeddings, a non-private reference point, reached 100.0% for clap at an FMR of 0.1% (run 2026-10-29_n2-0184), compared with 95.5% for cosine matching (run 2026-10-29_n2-0176); at 1%, both reached 100.0% (runs 2026-10-29_n2-0183, 2026-10-29_n2-0175). We term this difference of 4.5 points the "classifier gap". A positive gap suggests that part of the loss may stem from matching against a reference list rather than from the encoder itself. The 95% CIs overlapped, so this gap may also reflect sampling variation.

| Encoder (XD-Violence track: benign windows from normal XD videos) | Cosine top-1: event recall at FMR 1% (95% CI) | AUC | Measured FMR | Logistic regression: event recall at FMR 1% (measured FMR) | Runs |
|---|---|---|---|---|---|
| motion_stats | 23.9% (14.9 to 34.3) | 0.718 | 0.74% | 32.8% (1.55%) | 2026-10-29_n2-0697, 2026-10-29_n2-0705 |
| ast | 11.9% (4.5 to 19.4) | 0.857 | 0.00% | 59.7% (0.27%) | 2026-10-29_n2-0523, 2026-10-29_n2-0531 |
| mfcc_stats | 10.4% (3.0 to 17.9) | 0.670 | 0.00% | 23.9% (2.81%) | 2026-10-29_n2-0349, 2026-10-29_n2-0357 |
| clip_frames | 3.0% (0.0 to 7.5) | 0.840 | 0.08% | 79.1% (1.23%) | 2026-10-29_n2-0581, 2026-10-29_n2-0589 |
| clap | 3.0% (0.0 to 7.5) | 0.899 | 0.03% | 52.2% (0.05%) | 2026-10-29_n2-0465, 2026-10-29_n2-0473 |
| fusion:ast+videomae_k400 | 1.5% (0.0 to 4.5) | 0.911 | 0.00% | 88.1% (2.10%) | 2026-10-29_n2-0929, 2026-10-29_n2-0937 |
| vggish | 1.5% (0.0 to 4.5) | 0.867 | 0.03% | 58.2% (0.19%) | 2026-10-29_n2-0291, 2026-10-29_n2-0299 |
| hubert | 0.0% (0.0 to 5.4, exact) | 0.510 | 0.08% | 26.9% (0.27%) | 2026-10-29_n2-0407, 2026-10-29_n2-0415 |
| xclip_b32 | 0.0% (0.0 to 5.4, exact) | 0.809 | 0.00% | 94.0% (1.06%) | 2026-10-29_n2-0639, 2026-10-29_n2-0647 |
| videomae_k400 | 0.0% (0.0 to 5.4, exact) | 0.881 | 0.00% | 94.0% (2.13%) | 2026-10-29_n2-0755, 2026-10-29_n2-0763 |
| fusion:clap+clip_frames | 0.0% (0.0 to 5.4, exact) | 0.940 | 0.00% | 76.1% (0.30%) | 2026-10-29_n2-0813, 2026-10-29_n2-0821 |
| fusion:ast+clip_frames | 0.0% (0.0 to 5.4, exact) | 0.913 | 0.00% | 79.1% (0.25%) | 2026-10-29_n2-0871, 2026-10-29_n2-0879 |
| fusion:ast+xclip_b32 | 0.0% (0.0 to 5.4, exact) | 0.885 | 0.00% | 89.6% (1.25%) | 2026-10-29_n2-0987, 2026-10-29_n2-0995 |

- When benign windows came from normal XD videos, every audio embedding separated violent from benign windows less well than on the home track: with cosine top-1, event recall at an FMR of 1% ranged from 0.0% (hubert; run 2026-10-29_n2-0407) to 23.9% (motion_stats; run 2026-10-29_n2-0697), compared with 83.6% to 100.0% on the home track (runs 2026-10-29_n2-0059, 2026-10-29_n2-0175), and AUC ranged from 0.510 to 0.940 (runs 2026-10-29_n2-0407, 2026-10-29_n2-0813). On this track, embeddings were centered on normal XD training windows rather than on SINS windows, so its values differ from the corresponding setting in Section 1.
- On the same track, a logistic-regression classifier trained on the same embeddings reached 26.9% (hubert; run 2026-10-29_n2-0415) to 94.0% (xclip_b32; run 2026-10-29_n2-0647) at an FMR of 1%, leaving out mfcc_stats, motion_stats, videomae_k400, and fusion:ast+videomae_k400, whose classifier exceeded 1.5 times the FMR target, compared with at most 23.9% for cosine top-1 (run 2026-10-29_n2-0697). This "classifier gap" suggests that the embeddings may carry information that separates violent from normal videos, and that matching each window against its nearest reference, rather than the encoder, limits recall within one recording domain.
- Concatenating clap with frame embeddings from Contrastive Language-Image Pre-training (CLIP) changed AUC on the XD track from 0.899 with audio only (run 2026-10-29_n2-0465) and 0.840 with video only (run 2026-10-29_n2-0581) to 0.940 (run 2026-10-29_n2-0813), and event recall at an FMR of 1% with cosine top-1 from 3.0% and 3.0% to 0.0%. With logistic regression, the fused embedding reached 76.1%, compared with 52.2% for clap alone (runs 2026-10-29_n2-0821, 2026-10-29_n2-0473). This suggests that fusion may improve the bulk of the score distribution more than its strict tail.

## 3. Matching

- To compare matching methods, we fixed the encoder (clap) and the threshold calibration (benign SINS windows), and varied only the matcher. For locality-sensitive hashing (LSH), k is the number of bits per band, L the number of bands, and m the number of matching bands that calibration requires. The best LSH setting is the one with the highest event recall on the same validation events, so its recall is slightly optimistic.

| Matcher | Event recall at FMR 1% (95% CI) | Event recall at FMR 0.1% | Measured FMR (1% target) | AUC | Threshold | Runs |
|---|---|---|---|---|---|---|
| cosine_top1 | 100.0% (94.6 to 100.0, exact) | 95.5% | 1.31% | 0.993 | calibrated | 2026-10-29_n2-0175, 2026-10-29_n2-0176 |
| cosine_top5 | 100.0% (94.6 to 100.0, exact) | 97.0% | 1.65% | 0.994 | calibrated | 2026-10-29_n2-0177, 2026-10-29_n2-0178 |
| l2_raw | 98.5% (94.0 to 100.0) | 85.1% | 1.41% | 0.979 | calibrated | 2026-10-29_n2-0179, 2026-10-29_n2-0180 |
| hamming_sign | 98.5% (95.5 to 100.0) | 61.2% | 0.93% | 0.990 | calibrated | 2026-10-29_n2-0181, 2026-10-29_n2-0182 |
| logreg | 100.0% (94.6 to 100.0, exact) | 100.0% | 2.72% | 0.999 | calibrated | 2026-10-29_n2-0183, 2026-10-29_n2-0184 |
| lsh (k = 12, L = 16, seed = 0) | 98.5% (95.5 to 100.0) | 88.1% | 1.07% | 0.987 | m = 11 | 2026-10-29_n2-0185, 2026-10-29_n2-0186 |
| lsh (k = 12, L = 16, seed = 1) | 92.5% (85.1 to 98.5) | 88.1% | 0.30% | 0.986 | m = 11 | 2026-10-29_n2-0187, 2026-10-29_n2-0188 |
| lsh (k = 12, L = 16, seed = 2) | 100.0% (94.6 to 100.0, exact) | 98.5% | 0.94% | 0.989 | m = 10 | 2026-10-29_n2-0189, 2026-10-29_n2-0190 |
| lsh (k = 12, L = 32, seed = 0) | 100.0% (94.6 to 100.0, exact) | 88.1% | 1.22% | 0.991 | m = 19 | 2026-10-29_n2-0191, 2026-10-29_n2-0192 |
| lsh (k = 12, L = 32, seed = 1) | 100.0% (94.6 to 100.0, exact) | 92.5% | 0.85% | 0.991 | m = 19 | 2026-10-29_n2-0193, 2026-10-29_n2-0194 |
| lsh (k = 12, L = 32, seed = 2) | 100.0% (94.6 to 100.0, exact) | 95.5% | 1.74% | 0.992 | m = 17 | 2026-10-29_n2-0195, 2026-10-29_n2-0196 |
| lsh (k = 16, L = 16, seed = 0) | 100.0% (94.6 to 100.0, exact) | 83.6% | 0.80% | 0.987 | m = 6 | 2026-10-29_n2-0197, 2026-10-29_n2-0198 |
| lsh (k = 16, L = 16, seed = 1) | 98.5% (95.5 to 100.0) | 94.0% | 0.65% | 0.985 | m = 6 | 2026-10-29_n2-0199, 2026-10-29_n2-0200 |
| lsh (k = 16, L = 16, seed = 2) | 97.0% (92.5 to 100.0) | 95.5% | 0.57% | 0.986 | m = 6 | 2026-10-29_n2-0201, 2026-10-29_n2-0202 |
| lsh (k = 16, L = 32, seed = 0) | 100.0% (94.6 to 100.0, exact) | 86.6% | 1.13% | 0.989 | m = 10 | 2026-10-29_n2-0203, 2026-10-29_n2-0204 |
| lsh (k = 16, L = 32, seed = 1) | 100.0% (94.6 to 100.0, exact) | 94.0% | 0.83% | 0.990 | m = 10 | 2026-10-29_n2-0205, 2026-10-29_n2-0206 |
| lsh (k = 16, L = 32, seed = 2) | 98.5% (95.5 to 100.0) | 98.5% | 1.11% | 0.993 | m = 9 | 2026-10-29_n2-0207, 2026-10-29_n2-0208 |
| lsh (k = 20, L = 16, seed = 0) | 98.5% (95.5 to 100.0) | 88.1% | 1.17% | 0.973 | m = 3 | 2026-10-29_n2-0209, 2026-10-29_n2-0210 |
| lsh (k = 20, L = 16, seed = 1) | 100.0% (94.6 to 100.0, exact) | 94.0% | 1.07% | 0.973 | m = 3 | 2026-10-29_n2-0211, 2026-10-29_n2-0212 |
| lsh (k = 20, L = 16, seed = 2) | 98.5% (95.5 to 100.0) | 98.5% | 0.48% | 0.963 | m = 3 | 2026-10-29_n2-0213, 2026-10-29_n2-0214 |
| lsh (k = 20, L = 32, seed = 0) | 98.5% (95.5 to 100.0) | 89.6% | 0.96% | 0.985 | m = 5 | 2026-10-29_n2-0215, 2026-10-29_n2-0216 |
| lsh (k = 20, L = 32, seed = 1) | 100.0% (94.6 to 100.0, exact) | 98.5% | 0.72% | 0.983 | m = 5 | 2026-10-29_n2-0217, 2026-10-29_n2-0218 |
| lsh (k = 20, L = 32, seed = 2) | 98.5% (95.5 to 100.0) | 98.5% | 1.57% | 0.984 | m = 4 | 2026-10-29_n2-0219, 2026-10-29_n2-0220 |
| lsh (k = 24, L = 16, seed = 0) | 91.0% (83.6 to 97.0) | 83.6% | 0.48% | 0.923 | m = 2 | 2026-10-29_n2-0221, 2026-10-29_n2-0222 |
| lsh (k = 24, L = 16, seed = 1) | 97.0% (92.5 to 100.0) | 94.0% | 0.57% | 0.914 | m = 2 | 2026-10-29_n2-0223, 2026-10-29_n2-0224 |
| lsh (k = 24, L = 16, seed = 2) | 95.5% (89.6 to 100.0) | 85.1% | 0.28% | 0.906 | m = 2 | 2026-10-29_n2-0225, 2026-10-29_n2-0226 |
| lsh (k = 24, L = 32, seed = 0) | 95.5% (89.6 to 100.0) | 91.0% | 0.39% | 0.968 | m = 3 | 2026-10-29_n2-0227, 2026-10-29_n2-0228 |
| lsh (k = 24, L = 32, seed = 1) | 95.5% (89.6 to 100.0) | 91.0% | 0.76% | 0.954 | m = 3 | 2026-10-29_n2-0229, 2026-10-29_n2-0230 |
| lsh (k = 24, L = 32, seed = 2) | 97.0% (92.5 to 100.0) | 94.0% | 0.48% | 0.966 | m = 3 | 2026-10-29_n2-0231, 2026-10-29_n2-0232 |

- Converting embeddings into LSH codes for exact matching changed event recall from 100.0% with cosine similarity (run 2026-10-29_n2-0175) to 100.0% with the best LSH setting (k = 20, L = 32, m = 5; run 2026-10-29_n2-0217), a difference of 0.0 points that we term the "discretization gap". Because both detected every event, the 1% target cannot reveal a gap here; the 0.1% target below can.
- Across the LSH settings in the table, event recall at an FMR of 1% ranged from 91.0% (k = 24, L = 16, seed = 0; run 2026-10-29_n2-0221) to 100.0% (k = 20, L = 32, seed = 1; run 2026-10-29_n2-0217).
- At the stricter FMR target of 0.1%, the "discretization gap" was -3.0 points: event recall reached 95.5% with cosine similarity (95% CI 89.6 to 100.0; run 2026-10-29_n2-0176), compared with 98.5% with the best LSH setting (k = 20, L = 16, seed = 2, m = 4; 95% CI 95.5 to 100.0; run 2026-10-29_n2-0214). The 95% CIs overlapped, so this gap remains uncertain at this scale.
- Hamming distance on the sign bits of a principal component projection reached 98.5% (run 2026-10-29_n2-0181), compared with 100.0% for cosine similarity (run 2026-10-29_n2-0175).
- Averaging the similarity to the five nearest references (cosine top-5) reached 100.0% (run 2026-10-29_n2-0177), compared with 100.0% for the nearest reference alone (run 2026-10-29_n2-0175).
- Similarly, the L2 distance to the nearest reference on centered, unnormalized embeddings reached 98.5% (run 2026-10-29_n2-0179), compared with 100.0% for cosine similarity (run 2026-10-29_n2-0175).
- On the XD-Violence track, the best LSH setting within the FMR tolerance reached a higher event recall at an FMR of 1% than cosine top-1 for 13 of the 13 embeddings. For example, fusion:ast+clip_frames reached 64.2% with LSH (k = 12, L = 32, seed = 2, m = 19; run 2026-10-29_n2-0891), compared with 0.0% with cosine top-1 (run 2026-10-29_n2-0871). Because the LSH score counts the bands that match any reference, it may pool evidence across references, whereas cosine top-1 relies on the single nearest one.
- The Apple-style private set intersection (PSI) protocol reproduced the plaintext LSH decisions on 100.0% of 150 windows, 65 of which matched in plaintext (k = 20, L = 32, m = 5; runs 2026-10-29_n2-0217 and 2026-10-29_n2-1061).
