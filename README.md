# Detecting Antecedents of Violence and Stanchions of Peace Across Seasons and Biomes

Code for the manuscript. We model weekly conflict and social unrest on a global 0.5° grid (ACLED, 2018-2024) with XGBoost,
explain the models with TreeSHAP, and quantify how stable the sustainability and biome-specific patterns are with a group bootstrap
over 5°×5° spatial blocks.

## Pipeline

| Step | What it does | Script | Output |
|---|---|---|---|
| 1 | Download weekly climate, then build the grid-cell-week dataset (events, climate, socioeconomic, sustainability, biome, V-Dem) | `scripts/preprocessing.py` | `data/partitions/data_final_world_XX.csv` |
| 2 | Main XGBoost model, validation metrics, SHAP values on the explanation sample | `scripts/model_xgb.py` | `results/plot_data_*.csv`, `results/shap_XGB_*.csv`, PR curves |
| 3 | Alternative models (random forest, LightGBM, CatBoost, extra-trees, logistic regression) | `scripts/model_alternates.py` | metrics in the job logs |
| 4 | Group bootstrap: 40 replicates per outcome (`--mode explain_only` is a cheaper variant that skips retraining) | `scripts/bootstrap_shap.py` | `results/boot/shap_<Outcome>_<rep>.csv` |
| 5 | Figures: fitted-model importance (Fig. 1) and the bootstrap figures (sustainability, biomes) | `scripts/plot.py` | `figures/F1_importance.png`, `figures/F*_uq_iqr.png` |

`scripts/functions.py` holds the shared helpers.

## Quick start

Data are not included (see [`data/README.md`](data/README.md)). Install the dependencies, place the data under `data/`, and run from the repository root.

```bash
pip install -r requirements.txt

# 1. preprocessing. Download once (CAMS needs a ~/.cdsapirc key), then build the 50 date partitions
python scripts/preprocessing.py --stage download --var all
for p in $(seq 1 50); do python scripts/preprocessing.py --part $p; done      # dataset + climate + vdem

# 2. main model: task 1 = Conflict, task 2 = Unrest (note the printed best_iteration for each)
for t in 1 2; do JOB_NAME=train_xgb_world_1 SGE_TASK_ID=$t python scripts/model_xgb.py; done

# 3. alternative models (optional; the comparison tables)
for m in rf lgbm catboost et logreg; do for t in 1 2; do \
  JOB_NAME=train_ens_world_1 SGE_TASK_ID=$t python scripts/model_alternates.py --model $m; done; done

# 4. bootstrap: odd tasks = Conflict, even = Unrest; tasks 1..8 with --n_rep 10 give 40 replicates per outcome
for t in $(seq 1 8); do JOB_NAME=boot_shap_world_1 SGE_TASK_ID=$t \
  python scripts/bootstrap_shap.py --n_rep 10 --n_expl 300000 --rounds_conflict 1120 --rounds_unrest 505; done

# 5. figures (--figs fitted or --figs boot draws only one group)
python scripts/plot.py --boot results/boot --interval iqr
```

On a cluster running SGE, the same steps are submitted with `qsub jobs/preprocessing.sh`, `jobs/train_xgb.sh`, `jobs/model_alternates.sh`, and
`jobs/bootstrap_shap.sh` (from the repository root). The scripts read `JOB_NAME` (`<name>_<continent>_<config line>`) and `SGE_TASK_ID`, which is why the
local commands above set them. XGBoost runs on a GPU (`device="cuda"`); change it to `"cpu"` in `model_xgb.py` and `bootstrap_shap.py` if none is available.
The `--rounds_*` values are `best_iteration + 1` from step 2.

## Layout

```
scripts/         functions, preprocessing, model_xgb, model_alternates, bootstrap_shap, plot
jobs/            SGE job scripts
model_configs.txt   feature-engineering settings (4-week rolling sums for event counts, 4-week rolling means for climate)
data/            inputs (not tracked)
results/         model outputs and bootstrap replicates (not tracked)
figures/         generated figures (not tracked)
```

## Outcomes and design in brief

- **Outcomes**: conflict (battles, explosions/remote violence, violence against civilians) and unrest (protests, riots); binary indicator of at least one event in a grid-cell-week.
- **Split**: train 2018-2022, evaluate 2023-2024.
- **Bootstrap**: blocks of 5°×5° are drawn with replacement; each training row is weighted by how often its block was drawn. Medians and 25th-75th percentile intervals are taken across the 40 replicates.
- **SHAP**: native TreeSHAP (`pred_contribs`) on a fixed explanation sample, log-odds scale.

## License and citation

To be added.
