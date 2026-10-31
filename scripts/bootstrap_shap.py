import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Group bootstrap of the XGBoost + SHAP analysis, two modes:
#
#   --mode refit          (default) retrain the model on block-resampled data, LOOPING over replicates inside one job so the
#                         data are loaded and prepared only once per job (GPU).
#       Task layout:  odd task IDs -> Conflict, even -> Unrest
#                     chunk = (SGE_TASK_ID - 1) // 2 ; this job runs replicates chunk * n_rep ... (chunk + 1) * n_rep - 1
#       Examples:     qsub -t 1-8 with --n_rep 10 -> 8 jobs, 40 replicates per outcome
#       Replicates already on disk are skipped, so a killed job can simply be resubmitted.
#
#   --mode explain_only   no retraining, no GPU: resample the saved explanation sample and recompute the summaries.
# ----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['refit', 'explain_only'], default='refit')
parser.add_argument('--rounds_conflict', type=int, default=None,
                    help='refit: boosting rounds for Conflict (main run best_iteration + 1)')
parser.add_argument('--rounds_unrest', type=int, default=None,
                    help='refit: boosting rounds for Unrest (main run best_iteration + 1)')
parser.add_argument('--n_rep', type=int, default=10, help='refit: replicates per job')
parser.add_argument('--n_train', type=int, default=5000000)
parser.add_argument('--n_expl', type=int, default=200000)
parser.add_argument('--expl_ppt_is_raw', type=int, default=1,
                    help='refit: 1 if the saved plot_data CSV still has raw (non-anomaly) precipitation')
parser.add_argument('--line', type=int, default=1, help='explain_only: model configuration number')
parser.add_argument('--n_boot', type=int, default=500, help='explain_only: number of resamples')
parser.add_argument('--res', default='./results', help='explain_only: folder with the main-run outputs')
parser.add_argument('--out', default='./results/boot_explain', help='explain_only: output folder')
args = parser.parse_args()

# summaries to save (names follow the feature names; anything not in feature_names is skipped)
bio_vars = ['tmin_rollmean_4', 'tmax_rollmean_4', 'ppt']
dep_vars = ['EPI_index', 'MF_pc_t', 'ECO_pc_gha', 'GHG_pc_t']
n_bins = 25
group_candidates = ['country', 'iso3', 'ISO_A3', 'iso_a3', 'NAME_0', 'admin', 'COUNTRY']


def run_explain_only(args):
    """Explanation-only group bootstrap: no retraining and no GPU.
    Resamples the groups (countries, or 5x5 degree blocks) of the saved explanation sample and recomputes the same
    summaries from the SHAP values saved by the main run (results/shap_XGB_<line>_<outcome>.csv). It measures how much the
    attributions change with which places are in the sample, holding the fitted model fixed. Writes the same long format
    as the retraining mode, so plot.py reads either folder."""
    non_feature = ['lon', 'lat', 'week_start', 'season_name']
    os.makedirs(args.out, exist_ok=True)

    for output_var in ['Unrest', 'Conflict']:
        shap_df = pd.read_csv(f'{args.res}/shap_XGB_{args.line:02d}_{output_var}.csv')
        df_plot = pd.read_csv(f'{args.res}/plot_data_{args.line:02d}_{output_var}.csv')
        assert len(shap_df) == len(df_plot), 'SHAP and plot data must be row-aligned'

        feats = [c for c in shap_df.columns if c not in non_feature]
        A = shap_df[feats].to_numpy(dtype=np.float32)
        absA = np.abs(A)
        fidx = {f: i for i, f in enumerate(feats)}

        group_col = next((c for c in group_candidates if c in df_plot.columns), None)
        if group_col is not None:
            g = df_plot[group_col].astype(str).to_numpy()
        else:
            g = (np.floor(df_plot['lat'] / 5).astype(int).astype(str) + '_' +
                 np.floor(df_plot['lon'] / 5).astype(int).astype(str)).to_numpy()
        print(output_var, 'resampling unit:', group_col if group_col is not None else '5x5 degree blocks')

        uniq, inv = np.unique(g, return_inverse=True)
        order = np.argsort(inv, kind='stable')
        splits = np.split(order, np.cumsum(np.bincount(inv))[:-1])

        biome_codes, biome_names = pd.factorize(df_plot['biome_str'])
        dep_info = {}
        for var in [v for v in dep_vars if v in fidx]:
            b = pd.qcut(df_plot[var], q=n_bins, duplicates='drop').cat.codes.to_numpy()
            x = pd.DataFrame({'x': df_plot[var].to_numpy(), 'b': b}).groupby('b')['x'].median()
            dep_info[var] = (b, x)

        rows = []
        for rep in range(args.n_boot):
            rng = np.random.default_rng(20000 + rep)
            draw = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([splits[k] for k in draw])

            imp = absA[idx].mean(axis=0)
            for f, i in fidx.items():
                rows.append(('importance', f, '', 'mean_abs', float(imp[i]), rep))

            bc = biome_codes[idx]
            for var in [v for v in bio_vars if v in fidx]:
                s = A[idx, fidx[var]]
                for k, name in enumerate(biome_names):
                    m = bc == k
                    if m.any():
                        rows.append(('biome', var, name, 'median', float(np.median(s[m])), rep))

            for var, (b, x) in dep_info.items():
                bb = b[idx]
                s = A[idx, fidx[var]]
                nb = len(x)
                sums = np.bincount(bb, weights=s, minlength=nb)
                cnts = np.bincount(bb, minlength=nb)
                for k in range(nb):
                    if cnts[k] > 0:
                        rows.append(('dependence', var, str(k), 'x_median', float(x.iloc[k]), rep))
                        rows.append(('dependence', var, str(k), 'mean_shap', float(sums[k] / cnts[k]), rep))

        out = pd.DataFrame(rows, columns=['kind', 'feature', 'group', 'stat', 'value', 'rep'])
        out['output'] = output_var
        out.to_csv(f'{args.out}/shap_{output_var}_all.csv', index=False)
        print('saved', f'{args.out}/shap_{output_var}_all.csv', len(out), 'rows')


if args.mode == 'explain_only':
    run_explain_only(args)
    sys.exit(0)

if args.rounds_conflict is None or args.rounds_unrest is None:
    parser.error('--rounds_conflict and --rounds_unrest are required in refit mode')

import xgboost as xgb
from functions import *

line = int(os.getenv('JOB_NAME').split('_')[-1])
continent = os.getenv('JOB_NAME').split('_')[-2]
task = int(os.getenv('SGE_TASK_ID')) - 1
output_var = ['Conflict', 'Unrest'][task % 2]
chunk = task // 2
reps = list(range(chunk * args.n_rep, (chunk + 1) * args.n_rep))
rounds = args.rounds_conflict if output_var == 'Conflict' else args.rounds_unrest
print(f'task={task + 1} output={output_var} replicates={reps[0]}..{reps[-1]} rounds={rounds}')

configs = read_model_configs()
count_lags = configs[line-1]['count_lags']
count_rolling = configs[line-1]['count_rolling']
clim_lags = configs[line-1]['clim_lags']
clim_rolling = configs[line-1]['clim_rolling']
climate_vars = ['hi', 'tmin', 'tmax', 'wbgt', 'drought', 'ppt', 'pm25']

data, feature_names = prepare_data(continent, climate_vars, count_lags, count_rolling, clim_lags, clim_rolling)

# ---- same feature engineering as the main script ----
south_shift = {"Winter": "Summer", "Spring": "Fall", "Summer": "Winter", "Fall": "Spring"}
data['month_str'] = data.week_start.dt.month_name()
data['month'] = data.week_start.dt.month.astype('int8')
data['season_str'] = data.month.apply(get_season)
data.loc[data["lat"] < 0, "season_str"] = data.loc[data["lat"] < 0, "season_str"].map(south_shift)
data['equatorial'] = (np.abs(data.lat) < 30).astype('int8')
data['season_cat'] = data['season_str'].astype('category')
data['season'] = data['season_cat'].cat.codes.astype('int8')

data.rename(columns={'BIOME_NAME': 'biome_str'}, inplace=True)
data = data[data.biome_str != 'N/A']
data['biome_str'] = data['biome_str'].str.replace(' ', '_')
data['biome_cat'] = data['biome_str'].astype('category')
data['biome'] = data['biome_cat'].cat.codes.astype('int8')

feature_names += ['biome', 'equatorial'] + ['season', 'month']

data.dropna(subset=['hi', 'tmin', 'tmax', 'wbgt', 'ppt'], inplace=True)
phase_cols = [c for c in data.columns if 'phase_' in c]
data.loc[:, phase_cols] = data.loc[:, phase_cols].fillna(0.)

# ---- resampling unit, stored as integer codes (per-row strings use far too much memory) ----
# the data has no country column, so groups are 5x5 degree blocks
group_col = next((c for c in group_candidates if c in data.columns), None)
if group_col is not None:
    group_codes = pd.factorize(data[group_col])[0].astype(np.int32)
else:
    group_codes = ((np.floor(data['lat'].to_numpy() / 5).astype(np.int32) + 18) * 100 +
                   (np.floor(data['lon'].to_numpy() / 5).astype(np.int32) + 36))
print('resampling unit:', group_col if group_col is not None else '5x5 degree blocks', flush=True)

# ---- training rows WITHOUT copying the whole frame (the validation years are not needed) ----
train_mask = (data.week_start.dt.year <= 2022).to_numpy()
ppt_vars = [v for v in data.columns if 'ppt' in v]
ppt_month_mean = data.loc[train_mask, ['month'] + ppt_vars].groupby('month')[ppt_vars].mean()


def ppt_anomaly(df):
    # numpy arithmetic, so there is no index alignment
    out = df.copy()
    out[ppt_vars] = df[ppt_vars].to_numpy() - ppt_month_mean.loc[df['month'].to_numpy()].to_numpy()
    return out


# fixed training subsample, identical in every replicate
rs = np.random.RandomState(1)
train_rows = np.flatnonzero(train_mask)
sel_rows = train_rows[rs.choice(len(train_rows), size=min(args.n_train, len(train_rows)))]
cols = list(dict.fromkeys(feature_names + ['month', output_var] + ppt_vars))
tr = ppt_anomaly(data.iloc[sel_rows][cols])
uniq, inv = np.unique(group_codes[sel_rows], return_inverse=True)
del data, group_codes, train_mask, train_rows, sel_rows
import gc
gc.collect()

X_train = tr[feature_names]
y_train = tr[output_var]
spw = np.sqrt((y_train.shape[0] - y_train.sum()) / y_train.sum())
print(f'training rows: {len(tr)}, groups: {len(uniq)}', flush=True)

# ---- fixed explanation sample (the one behind your figures), prepared once ----
df_expl = pd.read_csv(f'./results/plot_data_{line:02d}_{output_var}.csv')
sel = np.random.RandomState(7).choice(len(df_expl), size=min(args.n_expl, len(df_expl)), replace=False)
df_expl = df_expl.iloc[sel].reset_index(drop=True)
if args.expl_ppt_is_raw:
    df_expl = ppt_anomaly(df_expl)
dexpl = xgb.DMatrix(df_expl[feature_names], feature_names=feature_names)
biome_arr = df_expl['biome_str'].to_numpy()

dep_bins = {}
for var in [v for v in dep_vars if v in feature_names]:
    dep_bins[var] = pd.qcut(df_expl[var], q=n_bins, duplicates='drop').cat.codes.to_numpy()

os.makedirs('./results/boot', exist_ok=True)

for rep in reps:
    out_path = f'./results/boot/shap_{output_var}_{rep:03d}.csv'
    if os.path.exists(out_path):
        print('skip existing', out_path)
        continue
    t0 = time.time()

    # group bootstrap weights
    rng = np.random.default_rng(10000 + rep)
    draw = rng.integers(0, len(uniq), size=len(uniq))
    counts = np.bincount(draw, minlength=len(uniq))
    w = counts[inv].astype(np.float32)
    keep = np.flatnonzero(w > 0)

    params = {
        "objective": "binary:logistic",
        "eval_metric": 'aucpr',
        "seed": 42 + rep,
        "device": "cuda",
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 30,
        "reg_lambda": 5.0,
        "scale_pos_weight": spw,
    }
    dtrain = xgb.DMatrix(X_train.iloc[keep], label=y_train.iloc[keep], weight=w[keep],
                         feature_names=feature_names)
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=rounds, verbose_eval=False)
    del dtrain

    # native TreeSHAP on the GPU; last column is the bias term
    contrib = model.predict(dexpl, pred_contribs=True)[:, :-1]
    shap_df = pd.DataFrame(contrib, columns=feature_names)

    rows = []
    for f, v in shap_df.abs().mean().items():
        rows.append(('importance', f, '', 'mean_abs', float(v)))

    for var in [v for v in bio_vars if v in feature_names]:
        for b, s in shap_df[var].groupby(biome_arr):
            rows.append(('biome', var, b, 'median', float(s.median())))
            rows.append(('biome', var, b, 'q25', float(s.quantile(0.25))))
            rows.append(('biome', var, b, 'q75', float(s.quantile(0.75))))

    for var, bins in dep_bins.items():
        tmp = pd.DataFrame({'x': df_expl[var].to_numpy(), 's': shap_df[var].to_numpy(), 'b': bins})
        agg = tmp.groupby('b').agg(x=('x', 'median'), s=('s', 'mean'))
        for b, r in agg.iterrows():
            rows.append(('dependence', var, str(int(b)), 'x_median', float(r['x'])))
            rows.append(('dependence', var, str(int(b)), 'mean_shap', float(r['s'])))

    out = pd.DataFrame(rows, columns=['kind', 'feature', 'group', 'stat', 'value'])
    out['output'] = output_var
    out['rep'] = rep
    out.to_csv(out_path, index=False)
    print(f'replicate {rep} done in {(time.time() - t0) / 60:.1f} min -> {out_path}', flush=True)