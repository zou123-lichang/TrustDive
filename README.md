# TrustDive

TrustDive is the code and result archive for *Exact Reference-Conditioned Phase
Attribution for Transparent Diving Assessment*. The method starts from a
deterministic RICA² score, applies a bounded calibration in its frozen latent
space, and uses same-action examples from other competition events to define a
fixed comparison neighborhood. Within that neighborhood, it separates the
deployed prediction into takeoff, flight, and entry contributions.

The phase contributions describe the model, not a judge's reasoning. FineDiving
videos are not included in this repository.

## Main results

On the 749-video FineDiving test split, the prespecified TrustDive model reduced
MAE from 6.17 to 5.72 points and changed Spearman's rho from 0.8278 to 0.8342.
An additional post-freeze component audit showed that most of the score improvement came
from latent calibration; reference statistics added only a small ranking and
RMSE benefit beyond the latent-only model. For the 735 videos with enough legal
references, the three phase contributions reconstructed the deployed
prediction to numerical precision. The phase with the largest attributed
effect agreed with the largest direct phase-replacement effect in 90.61% of
videos.

## Repository contents

- `01_PROTOCOL/`: frozen analysis contracts.
- `02_CODE/src/trustdive/`: data, scoring, attribution, analysis, and figure code.
- `02_CODE/tests/`: unit and reproducibility tests.
- `03_RESULTS/V7_RISK_TASK/`: frozen predictions and statistical summaries used
  in the paper.
- `03_RESULTS/MANUSCRIPT_REVISION_2026_09_03/`: component, parser, oracle,
  interaction, reference-count, and runtime audits added during manuscript
  review.
- `04_MANUSCRIPT/.../EVIDENCE_FREEZE.md`: claim-to-artifact hashes.
- `04_MANUSCRIPT/.../figures_final/source_data/`: source data for the paper figures.
- `runs/`: the v7 run manifest and small fitted adapter models.

The versioned modules preserve the sequence of analyses that led to the paper.
The manuscript's primary results come from the frozen v7 artifacts; the
post-freeze diagnostic audit supplements them without replacing any v7 prediction.

## Installation

Python 3.10 was used for the reported experiments.

```bash
conda env create -f 02_CODE/environment/environment.yml
conda activate trustdive
python -m pip install -e "02_CODE[test]"
```

The exact package versions from the final Windows environment are recorded in
`02_CODE/environment/requirements-lock.txt`.

## Data and external model

Download FineDiving from the [official project](https://github.com/xujinglin/FineDiving)
and set its local path before running data-dependent commands.

PowerShell:

```powershell
$env:TRUSTDIVE_FINE_DIVING = "C:\path\to\Released_FineDiving_Dataset"
```

Bash:

```bash
export TRUSTDIVE_FINE_DIVING=/path/to/Released_FineDiving_Dataset
```

The scoring foundation follows the deterministic FineDiving setup of
[RICA²](https://github.com/abrarmajeedi/rica2_aqa). Its source code, dataset,
and large checkpoint files remain under their original distribution terms and
are not redistributed here. The compatibility patch used for the Windows run
is provided in `02_CODE/external_patches/`.

## Verify the published evidence

The following command checks all 12 frozen artifacts against the hashes used in
the manuscript:

```bash
python -m trustdive.manuscript_audit
```

Run the paper-specific tests with:

```bash
python -m pytest \
  02_CODE/tests/test_v7_risk_task.py \
  02_CODE/tests/test_manuscript_revision.py \
  02_CODE/tests/test_manuscript_figures.py -q
```

The bounded post-freeze diagnostic analyses can be regenerated with the command below once
the frozen RICA² latent and temporal-sequence caches have been produced by the
earlier pipeline stages:

```bash
python -m trustdive.manuscript_revision
```

The release includes the resulting per-video audit tables and their hashes, so
the reported statistics can be checked without redistributing the large
third-party-derived feature caches.

The result tables are already included, so the manuscript statistics can be
audited without downloading the videos. Rebuilding image-based case figures or
training the complete scorer requires FineDiving and the external RICA² setup.

## License

The TrustDive source code is released under the MIT License. FineDiving,
RICA², and other third-party material are governed by their own licenses and
terms of use.

## Citation

The manuscript is under submission. Until a final bibliographic record is
available, please use the metadata in `CITATION.cff` and cite the FineDiving and
RICA² papers when using their data or model components.
