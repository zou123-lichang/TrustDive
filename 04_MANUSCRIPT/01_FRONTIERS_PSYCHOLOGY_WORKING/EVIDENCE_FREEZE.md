# TrustDive manuscript evidence freeze

Freeze date: 2026-08-31  
Evidence basis: protocol-locked v7 artifacts; v8/v9 negative gates are used only to constrain claims.  
Primary contract SHA-256: `0a6035f4065e78f785ceb01d1e16ad65daab39c40bf1e16371c22adf0a7ce1a5`

Manuscript-level claim audit refreshed: 2026-09-01. This refresh did not alter any frozen experimental artifact or statistic.

## Frozen data and analysis sources

| Artifact | Role in manuscript | SHA-256 |
|---|---|---|
| `03_RESULTS/00_AUDIT/manifest.parquet` | 3,000-video manifest and official split | `f416e4569da03df23e0c6062fa110a31421b142e91ef5c86b6c8dafffbdb3092` |
| `03_RESULTS/V2_DISAGREEMENT/00_AUDIT/panel_targets_v2.parquet` | validated judge arrays, panel dispersion, analysis roles | `cfc5538e93af6cd661479beb7af8badd15e90301c56849516c7f55c36607a2a2` |
| `03_RESULTS/V7_RISK_TASK/01_RISK_TASK/risk_task_manifest_v7.parquet` | training-defined high-disagreement and error-risk labels | `b6a5afd65ad0f9efc1ec58a2b66788022e94db6356feea58fd9bd9235a11d8c3` |
| `03_RESULTS/V7_RISK_TASK/01_RISK_TASK/risk_thresholds_v7.json` | frozen training thresholds | `47614b81a09bab40459dfe9ee3cc106d4ebc4d7a7b58d2fa829af5a07891a742` |
| `03_RESULTS/V7_RISK_TASK/02_BASELINES/baseline_trials_v7.csv` | validation-only adapter/model selection | `5bd825342e53bc28679ecf8fd981f32dc91e2822cbcb7143d09a96e73047ce27` |
| `03_RESULTS/V7_RISK_TASK/03_SCORE/predictions_v7.parquet` | official-test scoring predictions | `8bfd401e4edf6a8d36f73e1d687c1fb702517cc33e5d51be99cb68839cdae043` |
| `03_RESULTS/V7_RISK_TASK/03_SCORE/score_summary_v7.json` | score-model selection status and open-set count | `34b6503b907002569da69d8c4d4694e4845532089b3795ad25a6ef161055255f` |
| `03_RESULTS/V7_RISK_TASK/04_PHASE_EVIDENCE/phase_evidence_v7.parquet` | per-video coalitions, Shapley values, interventions and stability | `9bb7d3dab55ff5c5b9092f424b3127bc566d716fdea2ee66a5d38c2a92e81068` |
| `03_RESULTS/V7_RISK_TASK/04_PHASE_EVIDENCE/phase_evidence_summary_v7.json` | phase-evidence summary | `182ea2637b68543c605b568718ea936f04d9aa9524fff16f105a9643962974f6` |
| `03_RESULTS/V7_RISK_TASK/05_RISK_REVIEW/review_priority_v7.parquet` | exploratory review ranking | `392bd612ee17cf802c637c97e0da2f0b1bed78410eca4b24d00f6de9ba032cc1` |
| `03_RESULTS/V7_RISK_TASK/05_RISK_REVIEW/analysis_summary_v7.json` | final v7 statistical summary and claim boundary | `3bf06a9a01ccecab485dd8dc06c960ac995362bea3b4cf9fe24bbc48f5f8f2b5` |
| `runs/run_manifest_v7.json` | environment, command and output-hash manifest | `ed959265fb791d3f6c6fcffdd670f6f49658c9b78686751aa875fc0727104eb2` |

## Claim-to-number map

| Claim ID | Evaluated set | Frozen result | Direct source | Manuscript boundary |
|---|---:|---|---|---|
| C1 overall scoring | 749 official-test videos | RICA²: rho 0.827787, MAE 6.167332, RMSE 8.759853; TrustDive: rho 0.834248, MAE 5.724603, RMSE 8.366534 | `analysis_summary_v7.json` -> `score` | Improvement is relative to the reproduced deterministic RICA² implementation, not a claim of SOTA. |
| C2 high-disagreement scoring | 325 valid seven-judge test videos; 94 above the fit-set threshold | MAE 8.437511 to 7.495029; reduction 11.1701%; paired clustered CI for TrustDive minus RICA² `[-1.489157, -0.360943]` | `analysis_summary_v7.json` -> `high_judge_risk_scoring` | A robustness subgroup defined by observed score dispersion; not abnormal-judge detection. |
| C2b differential subgroup benefit | Same 325-video panel subset | Difference-in-improvement estimate `0.270986`; clustered CI `[-0.318066, 0.816563]` | `analysis_summary_v7.json` -> `high_judge_risk_scoring` -> `risk_directed_gain` | The CI crosses zero. Do not state that TrustDive improves high-disagreement videos significantly more than ordinary-disagreement videos. |
| C3 exact reconstruction | 735 closed-set official-test videos | maximum Shapley reconstruction error `2.38419e-7`; maximum scorer-alignment error `4.76837e-7` | `phase_evidence_summary_v7.json` | Exactness applies to the final model output only. |
| C4 intervention fidelity | 735 closed-set official-test videos | top-phase match 0.906122; clustered CI `[0.883047, 0.928251]`; targeted-minus-random median 0.067163, CI `[0.053731, 0.083388]` | `analysis_summary_v7.json` -> `phase_evidence` | Model-faithfulness evidence; not a reconstruction of human deductions. |
| C5 stability boundary | 735 closed-set official-test videos | boundary cosine 0.788393; boundary top-phase agreement 0.642177; alternate-reference cosine 0.862909 | `phase_evidence_summary_v7.json` | Report as moderate stability, not invariance. |
| C6 phase distribution | 735 closed-set official-test videos | entry 69.5238%, flight 17.2789%, takeoff 13.1973% | `phase_evidence_summary_v7.json` | Potentially task- and model-dependent. |
| C7 risk-weighting ablation | 94 high-disagreement test videos | risk-weighted minus plain Ridge MAE 0.009495; clustered CI `[-0.069797, 0.076458]` | `analysis_summary_v7.json` | Attribute the scoring gain to latent reference adaptation, not risk weighting. |
| C8 exploratory review | 749 official-test videos | combined review accepted-MAE reduction 2.6682%; clustered CI `[-0.001509, 0.053403]` | `analysis_summary_v7.json` | Exploratory and non-decisive; no human-interaction claim. |

## Frozen sample definitions

- Total FineDiving videos: 3,000; official train/test: 2,251/749.
- Development roles within the official training set: fit 1,559; validation 369; calibration 323. These roles are disjoint by the derived event-family identifier.
- Action types: 52; action groups: 6; difficulty values: 23; event families: 39.
- Valid seven-judge panels: 1,368 overall and 325 in the official test set. Three malformed score arrays were excluded only from panel analyses.
- High disagreement: sample SD of the seven judge scores at or above the fit-set 75th percentile, `0.3933978962`; 94 of the 325 eligible test videos met this definition.
- Closed-set phase evidence: at least three legal same-action, different-event-family references; 735/749 test videos. The remaining 14 fell back to RICA² and received no reference-based decomposition.

## Prohibited extrapolations

- No abnormal, biased, corrupt or inattentive real-judge detection.
- No claim that model-attributed phases reveal a judge's psychological reasoning.
- No demonstrated improvement in human understanding, trust, behavior or decision quality.
- No injury-risk or athlete-safety interpretation of “risk.”
- No first/SOTA language without same-protocol reproduction.
