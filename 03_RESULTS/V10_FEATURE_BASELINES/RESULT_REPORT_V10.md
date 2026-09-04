# TrustDive v10 matched-feature baseline results

## Decision

**METRIC TRADE-OFF.** TrustDive achieved the lowest overall MAE and the lowest MAE on the 94 high-disagreement videos. TSA-style achieved the highest Spearman correlation, lowest RMSE, and lowest Relative-L2. The paired clustered-bootstrap intervals did not establish a clear difference between TrustDive and either matched-feature baseline.

CoRe-style and TSA-style are literature-grounded matched-feature implementations inspired by CoRe (ICCV 2021) and FineDiving/TSA (CVPR 2022). They are not official reproductions of those methods and are not presented as end-to-end SOTA comparisons.

## Official-test performance

All methods were evaluated on the same 749 official-test videos. Bold entries are the best observed value for each metric.

| Method | Spearman rho (higher) | MAE (lower) | RMSE (lower) | Relative-L2 (lower) | High-disagreement MAE, n=94 (lower) |
|---|---:|---:|---:|---:|---:|
| Deterministic RICA2 | 0.8278 | 6.1673 | 8.7599 | 0.7040 | 8.4375 |
| RICA2 + CoRe-style reference adapter | 0.8339 | 5.7401 | 8.3891 | 0.6457 | 7.5603 |
| RICA2 + TSA-style phase adapter | **0.8380** | 5.7602 | **8.3264** | **0.6361** | 7.8142 |
| TrustDive | 0.8342 | **5.7246** | 8.3665 | 0.6422 | **7.4950** |

## Paired clustered-bootstrap comparisons

Differences are TrustDive minus the named baseline. Negative MAE and RMSE differences favor TrustDive; positive Spearman differences favor TrustDive. Intervals use 10,000 event-family-clustered bootstrap samples.

| Baseline | Subset | Metric | Difference | 95% CI |
|---|---|---:|---:|---:|
| CoRe-style | All 749 | MAE | -0.0155 | [-0.0781, 0.0386] |
| CoRe-style | All 749 | Spearman rho | +0.0003 | [-0.0022, 0.0030] |
| CoRe-style | All 749 | RMSE | -0.0226 | [-0.1084, 0.0514] |
| CoRe-style | High disagreement, n=94 | MAE | -0.0652 | [-0.2473, 0.0908] |
| TSA-style | All 749 | MAE | -0.0356 | [-0.1838, 0.1179] |
| TSA-style | All 749 | Spearman rho | -0.0038 | [-0.0122, 0.0048] |
| TSA-style | All 749 | RMSE | +0.0402 | [-0.1809, 0.2463] |
| TSA-style | High disagreement, n=94 | MAE | -0.3192 | [-0.8409, 0.1614] |

## Model selection

- CoRe-style: hidden dimension 64, learning rate 1e-3, seven final epochs.
- TSA-style: hidden dimension 128, learning rate 3e-4, two final epochs.
- Both configurations were selected only from fit/validation results. Calibration was unused, and official-test labels were not accessed during model selection.
- Three fixed seeds (20260904, 20260905, and 20260906) were trained after contract freezing; their mean prediction was used as the reported ensemble.

## Interpretation for the manuscript

The v10 experiment supports the statement that TrustDive is **competitive with representative literature-grounded matched-feature baselines while uniquely providing exact counterfactual phase evidence**. It does not support a claim of comprehensive or statistically clear scoring superiority over CoRe-style and TSA-style.

If these results are added to the paper, the table should bold the actual best value in each column. The wording should distinguish the common frozen-feature protocol from official end-to-end reproductions. Internal component ablations should remain separate because they answer a different question: which TrustDive inputs contribute to its prediction.

## Verification and anomalies

- Frozen inputs, data roles, reference maps, and manuscript hashes passed audit.
- Final predictions contain 4,494 rows (749 videos x 2 models x 3 seeds), with no missing or non-finite values.
- Legal references are restricted to the permitted training pool, use the same action type, and exclude the query event family.
- Open-set samples fall back to deterministic RICA2 as specified.
- Repeated inference from the same checkpoint is exactly reproducible.
- The complete project test suite passed (90 tests).
- Recorded GPU time was 142.95 seconds (0.0397 h), below the 2 h cap.
- The first smoke report exposed a double-counted runtime extrapolation; a second attempt exposed the zero-reference open-set edge case. Both implementation defects were corrected before pilot model selection. The failed attempts remain in the GPU ledger.

## Material Passport

`experiment-agent / result-report / 2026-09-04 / VERIFIED / feature_baselines_v10`


