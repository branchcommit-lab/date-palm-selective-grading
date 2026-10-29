# Publication tables and figures

This directory contains the numerical tables and figure-generation scripts for the accompanying selective-grading study. All figures and tables are generated from the frozen observation records. The figure code does not load pretrained checkpoints or execute image inference.

## Commands

```bash
python run.py results
python publication/rebuild_manuscript_assets.py
python publication/test_manuscript_assets.py
```

Outputs are written to `publication/generated/`. The publication workflow creates seven figures and ten tables (seven main tables and Supplementary Tables S1–S3). `run.py results` also produces three general-purpose plots in `outputs/results/`.

## Figures and source records

| Figure | Content | Main source |
|---|---|---|
| 1 | Dataset, referral policies and experimental protocols | Split and protocol metadata |
| 2 | C1 referral and accepted-set comparisons | `data/raw_records/final_candidate_predictions.csv` |
| 3 | C2 corrected and regressed originals | `data/raw_records/final_candidate_predictions.csv` |
| 4 | C3 four-view classification accuracy | `data/tables/C3_FOUR_VIEW_ALL_12_RUNS.csv` |
| 5 | C3 excluded-size-class discrimination | `data/raw_records/c3/size_holdout/` |
| 6 | C3 photographic-condition differences | `data/raw_records/stress_predictions.csv` |
| 7 | C3 cultivar holdout and baseline context | `data/tables/C3_CULTIVAR_HOLDOUT_ALL_12_RUNS.csv` and the C2 baseline |

Figure 7 uses the 68.81% Dandhi *baseline* accuracy as a separate contextual comparator. The model and population for the cultivar-holdout experiment differ from those for the baseline. Figure 1 identifies four held-out alteration families at three severity levels. Supplementary Table S1 reports the family/severity referral observations; S2 provides paired error-capture counts and S3 provides cultivar-holdout seed variation.

## Interpretation of reproduction

The regeneration script checks source counts, referral totals, bootstrap contrasts and C3 class-holdout and stress results. Figure formatting may differ from article typesetting, while the underlying reported values follow the frozen records. Original photographs and checkpoint weights are external to this package.
