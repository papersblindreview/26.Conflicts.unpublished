import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score
from functions import *

# ----------------------------------------------------------------------------------
# Usage:  python scripts/model_alternates.py --model {rf,lgbm,catboost,et,logreg}
#   logreg   Logistic regression (non-ensemble baseline; cuML on GPU, scikit-learn CPU fallback)
#   rf       Random forest      (cuML on GPU; falls back to scikit-learn on CPU)
#   lgbm     LightGBM           (tries CUDA, then OpenCL GPU, then CPU)
#   catboost CatBoost           (GPU)
#   et       Extra-trees        (scikit-learn, CPU only: no GPU implementation available)
# Job environment variables (JOB_NAME, SGE_TASK_ID) work exactly as in the XGBoost script.
# ----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['rf', 'lgbm', 'catboost', 'et', 'logreg'])
parser.add_argument('--n_train', type=int, default=5000000)
args = parser.parse_args()
model_name = args.model

np.random.seed(1)

line = int(os.getenv('JOB_NAME').split('_')[-1])
task_id = int(os.getenv('SGE_TASK_ID')) - 1
output = task_id % 2   # tasks alternate Conflict (even) / Unrest (odd); the model comes from --model
continent = os.getenv('JOB_NAME').split('_')[-2]
configs = read_model_configs()

y_groups = ['Conflict', 'Unrest']
output_var = y_groups[output]

count_lags = configs[line-1]['count_lags']
count_rolling = configs[line-1]['count_rolling']
clim_lags = configs[line-1]['clim_lags']
clim_rolling = configs[line-1]['clim_rolling']

climate_vars = ['hi', 'tmin', 'tmax', 'wbgt', 'drought', 'ppt', 'pm25']

data, feature_names = prepare_data(continent, climate_vars, count_lags, count_rolling, clim_lags, clim_rolling)

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

features_s = ['biome', 'equatorial']
features_t = ['season', 'month']
feature_names += features_s + features_t

data.dropna(subset=['hi', 'tmin', 'tmax', 'wbgt', 'ppt'], inplace=True)
phase_cols = [c for c in data.columns if 'phase_' in c]
data.loc[:, phase_cols] = data.loc[:, phase_cols].fillna(0.)

# split data (identical to the XGBoost script)
data_train = data[data.week_start.dt.year <= 2022].copy()
ppt_vars = [v for v in data_train.columns if 'ppt' in v]
# precipitation enters the models as an anomaly: the value minus the training-period mean for the same calendar month.
# The means are looked up by month (not by row index), so training, validation and explanation data are treated identically.
ppt_month_mean = data_train.groupby('month')[ppt_vars].mean()


def ppt_anomaly(df):
    cols = [c for c in ppt_vars if c in df.columns]
    out = df.copy()
    out[cols] = df[cols].to_numpy() - ppt_month_mean.loc[df['month'].to_numpy(), cols].to_numpy()
    return out


data_train = ppt_anomaly(data_train)

data_val = ppt_anomaly(data[data.week_start.dt.year > 2022])

N = args.n_train
train_ids = np.random.choice(data_train.shape[0], size=min(N, data_train.shape[0]))

X_train = data_train[feature_names]
X_val = data_val[feature_names]
y_train = data_train[output_var]
y_val = data_val[output_var]
X_train, y_train = X_train.iloc[train_ids], y_train.iloc[train_ids]

for c in feature_names: print(c.upper())

# same class weighting as the XGBoost script
spw = float(np.sqrt((y_train.shape[0] - y_train.sum()) / y_train.sum()))
print(f'MODEL: {model_name}   scale_pos_weight: {spw:.3f}')


def predict_chunks(fn, X, chunk=1000000):
    """Predict in chunks to keep GPU / host memory bounded on the ~15M-row validation set."""
    return np.concatenate([fn(X[i:i + chunk]) for i in range(0, X.shape[0], chunk)])


#####################################################################################
# Models
#####################################################################################

if model_name == 'rf':
    # NaNs are not supported by cuML, so use a sentinel value (trees split on it harmlessly)
    Xtr = X_train.fillna(-999).to_numpy(dtype=np.float32)
    Xva = X_val.fillna(-999).to_numpy(dtype=np.float32)
    ytr = y_train.to_numpy(dtype=np.int32)
    try:
        from cuml.ensemble import RandomForestClassifier as cuRF
        # cuML has no class_weight, so approximate scale_pos_weight by replicating positives
        rep = max(int(round(spw)), 1)
        pos = np.where(ytr == 1)[0]
        Xtr = np.concatenate([Xtr] + [Xtr[pos]] * (rep - 1))
        ytr = np.concatenate([ytr] + [ytr[pos]] * (rep - 1))
        model = cuRF(n_estimators=300, max_depth=16, max_features=0.8, n_bins=128,
                     min_samples_leaf=30, bootstrap=True, max_samples=0.8,
                     random_state=42, output_type='numpy')
        model.fit(Xtr, ytr)
        print('random forest: cuML (GPU)')
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300, max_depth=16, max_features=0.8,
                                       min_samples_leaf=30, class_weight={0: 1.0, 1: spw},
                                       n_jobs=-1, random_state=42)
        model.fit(Xtr, ytr)
        print('random forest: scikit-learn (CPU fallback; cuML not installed)')
    y_prob = predict_chunks(lambda x: np.asarray(model.predict_proba(x))[:, 1], Xva)

elif model_name == 'et':
    from sklearn.ensemble import ExtraTreesClassifier
    Xtr = X_train.fillna(-999).to_numpy(dtype=np.float32)
    Xva = X_val.fillna(-999).to_numpy(dtype=np.float32)
    model = ExtraTreesClassifier(n_estimators=200, max_depth=18, max_features=0.8,
                                 min_samples_leaf=30, class_weight={0: 1.0, 1: spw},
                                 n_jobs=-1, random_state=42)
    model.fit(Xtr, y_train.to_numpy())
    print('extra trees: scikit-learn (CPU only)')
    y_prob = predict_chunks(lambda x: model.predict_proba(x)[:, 1], Xva)

elif model_name == 'lgbm':
    import lightgbm as lgb
    base = dict(objective='binary', learning_rate=0.05, num_leaves=31, max_depth=5,
                min_child_samples=30, reg_lambda=5.0, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, scale_pos_weight=spw, n_estimators=1000,
                random_state=42, verbose=-1)
    fitted = None
    for device in ['cuda', 'gpu', 'cpu']:   # use the GPU build if one is available
        try:
            model = lgb.LGBMClassifier(device=device, **base)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                      eval_names=['train', 'val'], eval_metric='average_precision',
                      callbacks=[lgb.early_stopping(100, first_metric_only=False),
                                 lgb.log_evaluation(10)])
            fitted = device
            break
        except lgb.basic.LightGBMError as e:
            print(f'LightGBM device={device} failed: {e}')
    print(f'lightgbm: device={fitted}')
    y_prob = predict_chunks(lambda x: model.predict_proba(x)[:, 1], X_val)

elif model_name == 'logreg':
    # Linear baseline: needs imputation, scaling, and one-hot coding of the integer-coded
    # categoricals (biome, season, month), which would be meaningless as numeric values.
    from sklearn.preprocessing import StandardScaler
    cat_cols = [c for c in ['biome', 'season', 'month'] if c in feature_names]
    num_cols = [c for c in feature_names if c not in cat_cols]

    medians = X_train[num_cols].median()
    scaler = StandardScaler().fit(X_train[num_cols].fillna(medians))
    dummy_cols = pd.get_dummies(X_train[cat_cols].astype(str), dtype=np.float32).columns

    def prep(X):
        num = scaler.transform(X[num_cols].fillna(medians)).astype(np.float32)
        cat = (pd.get_dummies(X[cat_cols].astype(str), dtype=np.float32)
                 .reindex(columns=dummy_cols, fill_value=0.).to_numpy(dtype=np.float32))
        return np.hstack([num, cat])

    Xtr = prep(X_train)
    ytr = y_train.to_numpy(dtype=np.int32)
    try:
        from cuml.linear_model import LogisticRegression as cuLR
        model = cuLR(penalty='l2', C=1.0, class_weight={0: 1.0, 1: spw},
                     max_iter=1000, output_type='numpy')
        model.fit(Xtr, ytr)
        print('logistic regression: cuML (GPU)')
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2', C=1.0, class_weight={0: 1.0, 1: spw},
                                   max_iter=300)
        model.fit(Xtr, ytr)
        print('logistic regression: scikit-learn (CPU fallback; cuML not installed)')
    chunk = 1000000
    y_prob = np.concatenate([np.asarray(model.predict_proba(prep(X_val.iloc[i:i + chunk])))[:, 1]
                             for i in range(0, X_val.shape[0], chunk)])

elif model_name == 'catboost':
    from catboost import CatBoostClassifier, Pool
    model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=5, l2_leaf_reg=5.0,
                               loss_function='Logloss', eval_metric='PRAUC',
                               scale_pos_weight=spw, bootstrap_type='Bernoulli', subsample=0.8,
                               grow_policy='Depthwise', min_data_in_leaf=30,
                               task_type='GPU', random_seed=42, early_stopping_rounds=100,
                               use_best_model=True, verbose=10)
    model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val))
    print('catboost: GPU')
    y_prob = predict_chunks(lambda x: model.predict_proba(x)[:, 1], X_val)


#####################################################################################
# evaluate model performance on validation data and plot precision-recall curve
#####################################################################################
y_val_np = y_val.to_numpy()
y_pred = (y_prob >= 0.5).astype(int)
precision, recall, thresholds = precision_recall_curve(y_val_np, y_prob)

pr_dir = f"./figures/shap/{model_name.upper()}/{continent}"
os.makedirs(pr_dir, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(recall, precision)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall Curve ({model_name})")
fig.savefig(f"{pr_dir}/PR_curve_{output_var}.png")

print('RESULTS')
print(classification_report(y_val_np, y_pred))
print(f'AUC-PR (average precision): {average_precision_score(y_val_np, y_prob):.5f}')
