# Post-review evidence addendum (2026-09-03)

This addendum records the bounded analyses performed in response to the pre-submission review. It supplements rather than overwrites the frozen v7 evidence. All new outputs are under `03_RESULTS/MANUSCRIPT_REVISION_2026_09_03/`, and the result manifest confirms that no historical v7 artifact was overwritten.

## Revised claim boundary

The score improvement should be attributed primarily to calibration of the frozen RICA2 latent representation. Same-action reference statistics provide a small incremental ranking and RMSE benefit, but their incremental MAE effect relative to a latent-only Ridge model is uncertain. The reference neighborhood remains essential to the paper's main interpretability contribution: it defines the comparison context for exact counterfactual phase attribution.

Accordingly, the manuscript now uses the following contribution chain:

1. bounded latent calibration of a reproduced deterministic RICA2 scorer;
2. exact reference-conditioned decomposition of the deployed scorer into takeoff, flight, and entry contributions;
3. intervention, annotated-boundary, reference-count, runtime, and failure-boundary audits of that evidence.

## Component ablation (official test, n=749)

| Model | Spearman rho | MAE | RMSE |
|---|---:|---:|---:|
| Frozen RICA2 | 0.8278 | 6.1673 | 8.7599 |
| Score-only linear calibration | 0.8310 | 5.8212 | 8.4675 |
| Latent-only Ridge | 0.8344 | 5.6963 | 8.3516 |
| Reference-only Ridge | 0.8325 | 5.7853 | 8.4133 |
| Full latent-reference Ridge | 0.8353 | 5.6903 | 8.3364 |
| Prespecified risk-weighted TrustDive | 0.8342 | 5.7246 | 8.3665 |

Paired event-family-clustered bootstrap intervals (10,000 draws) showed that full latent-reference Ridge versus latent-only Ridge changed rho by +0.00092 (95% CI 0.00003 to 0.00186), MAE by -0.0060 (95% CI -0.0246 to 0.0143), and RMSE by -0.0152 (95% CI -0.0270 to -0.0003). The reference statistics therefore did not establish an additional MAE reduction beyond latent calibration. The prespecified TrustDive configuration versus frozen RICA2 changed MAE by -0.4427 (95% CI -0.6437 to -0.2570), whereas the rho difference of +0.00646 had a 95% CI of -0.00010 to 0.01276.

## Phase parser audit

The parser predicts eight temporal tokens per video. On the official test set (n=749), token accuracy was 0.9676 and macro-F1 was 0.9669. Projecting token boundaries back to the original video timeline produced mean absolute boundary errors of 7.68 frames for takeoff-to-flight and 7.94 frames for flight-to-entry. Because this projection is limited by the coarse eight-token representation, the manuscript reports token-level accuracy and labels the frame errors as projected values.

## Exact attribution and added value over leave-one-out

Across 735 reference-supported test videos, the highest-contribution phase exceeded the mean of the two nonselected phase interventions by a median 0.1467 execution-quality points (95% clustered CI 0.1308 to 0.1633), and exceeded the strongest nonselected phase by 0.1159 (95% CI 0.0981 to 0.1293). The sign of the Shapley contribution agreed with the leave-one-out effect in 94.20% of phase-video pairs.

Leave-one-out effects did not provide an exact additive decomposition in the presence of phase interactions. Their median reconstruction error was 0.0395 execution-quality points, and 41.77% of videos exceeded an absolute error of 0.05. Exact three-phase enumeration is therefore justified as an accounting method rather than merely as an alternative ranking statistic.

## Annotated-boundary oracle

Repeating attribution with FineDiving annotated boundaries retained a 91.84% match between the highest attributed phase and the largest intervention phase. The targeted phase exceeded the strongest nonselected phase by a median 0.1023 points (95% clustered CI 0.0882 to 0.1129). Predicted- and annotated-boundary contribution vectors had a median cosine similarity of 0.9206, while highest-phase identity agreed in 68.03% of videos. The core intervention-fidelity conclusion therefore survives the annotated-boundary oracle, but case-level phase identity is not invariant to segmentation.

Projected parser boundary error had only a weak positive association with predicted-versus-annotated attribution divergence (Spearman rho 0.1003; 95% event-family-clustered CI 0.0272 to 0.1733). Boundary error therefore contributes to, but does not explain most of, case-level attribution instability.

## Reference-count and runtime sensitivity

K=3, 5, and 10 produced similar score performance; K=5 had the lowest MAE (5.6903) and highest rho (0.8353) by small margins. On a deterministic 100-video subset, K=3 versus K=5 attribution vectors had median cosine similarity 0.9549 and 68% highest-phase agreement; K=10 versus K=5 yielded 0.9714 and 81%. Warm batched attribution required 13.71, 16.14, and 22.34 ms per video for K=3, 5, and 10, respectively, on an NVIDIA GeForce RTX 5070. This supports K=5 as a balanced default, not as a universally optimal value.

## Official-split source coverage

All 749 official-test videos belonged to one of the 39 event families already represented in the official training split. A test-family-seen versus test-family-unseen subgroup comparison is therefore not identifiable under the official FineDiving split. The manuscript now states this directly and retains source-isolated retraining on nonoverlapping competitions as future work.

## Immutable sources

- Contract: `01_PROTOCOL/manuscript_revision_contract_2026-09-03.yaml`
- Result manifest: `03_RESULTS/MANUSCRIPT_REVISION_2026_09_03/run_manifest.json`
- Contract SHA-256: `cff389f36447a629936b4564cf52185f310395e4758a0a678a210af0aa0c7551`
- Historical v7 outputs overwritten: `false`
