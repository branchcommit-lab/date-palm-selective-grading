# Selective Grading Reliability through Referral Composition, Targeted Refinement and Generalization

Reproducible analysis and visualization resources for a study of selective date-fruit size grading. The experimental programme examines referral-policy behaviour (C1), Dandhi-targeted refinement (C2), and grading under changes in visual representation, acquisition conditions and excluded categories (C3).

## Source data and experimental scope

The source collection is [*Size-Based Date Fruit Dataset for Classification*](https://doi.org/10.17632/d986hyr2kb.1), Mendeley Data, Version 1. The analysis uses 5,828 eligible photographs from four cultivars, partitioned by source group into training (3,603), development (835), calibration (556) and locked-test (834) sets. The C1 evaluation contains 12,510 observations derived from 834 test originals: one original, two benign variants and twelve held-out synthetically altered variants per source. Synthetic transformations are experimental probes and do not represent naturally occurring defects or their prevalence.

The repository distributes **frozen prediction records and statistical analysis code**. The original photographs and model weights are not distributed. Analysis-level reproduction is supported; new model training and fresh image-level inference require additional external materials.

## Reproduce results, tables and figures

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python run.py results
python publication/rebuild_manuscript_assets.py
python publication/test_manuscript_assets.py
python tests/test_release.py
python src/methods/verify_package.py --root .
```

The analysis workflow produces C1/C2/C3 numerical summaries from recorded predictions. The publication script creates **Figures 1–7**, **Tables 1–7** and **Supplementary Tables S1–S3** in `publication/generated/`. Charts are regenerated from the same numerical records; presentation details can differ from typeset article displays.

## Repository contents

| Path | Contents |
|---|---|
| `data/metadata/` | Source-group split registry, calibration thresholds and development-selection information |
| `data/raw_records/` | Observation-level C1/C2 records and individual C3 evaluation runs |
| `data/tables/` | Consolidated model and experiment records |
| `configs/` | Frozen configuration parameters |
| `src/results_workflow.py` | Recorded-prediction analysis and statistical comparisons |
| `src/methods/` | Experimental model implementations, notebooks and inference utilities |
| `notebooks/` | End-to-end analysis notebook |
| `publication/` | Main and supplementary figure/table reproduction |
| `results/` | Reference outputs for the experimental analyses |
| `checkpoints/` | Model-identity metadata (no weight binaries) |
| `tests/` | Statistical, integrity and execution-scope tests |

## Reproducibility scope

The records support the three C1 referral policies, original-image accepted-grade error, source-paired bootstrap comparisons, paired error-capture tests, C2 corrected/regressed cases, seven photographic conditions, twelve representation runs, twelve cultivar holdouts and nine size-class holdouts. These are separate experimental protocols with different training or evaluation populations where specified in the manuscript.

`python run.py inference` audits existing prediction-level evidence. The `preflight` and `execute` options document the external inputs required for new inference; model weights and original image/mask benchmarks are not bundled. `python run.py retrain` reports the optional training-method registry but does not launch training. See [Inference](docs/INFERENCE.md) and [Training](docs/RETRAINING.md).

## Citation and license

The repository's source dataset is Shaikh, R. A., Mujtaba, G., Maitlo, A. K., Ali, D. & Shaikh, H. (2025), *Size-Based Date Fruit Dataset for Classification*, Mendeley Data, Version 1, https://doi.org/10.17632/d986hyr2kb.1.

The original software in this repository is released under the MIT license (see `LICENSE`). The original image dataset and third-party dependencies are governed by their respective terms. Model weights are not part of this distribution.
