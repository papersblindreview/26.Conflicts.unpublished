import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, log_loss
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score
from sklearn.inspection import permutation_importance
from functions import *
import geopandas as gpd
import pickle
import os
np.random.seed(1)


fig_dir = './figures'
os.makedirs('./results', exist_ok=True)
 
line = int(os.getenv('JOB_NAME').split('_')[-1])
output = int(os.getenv('SGE_TASK_ID')) - 1
continent = os.getenv('JOB_NAME').split('_')[-2]
configs = read_model_configs()

y_groups = ['Conflict','Unrest']
output_var = y_groups[output]

count_lags = configs[line-1]['count_lags']
count_rolling = configs[line-1]['count_rolling']
clim_lags = configs[line-1]['clim_lags']
clim_rolling = configs[line-1]['clim_rolling']

climate_vars = ['hi', 'tmin', 'tmax', 'wbgt','drought','ppt','pm25']
  
data, feature_names = prepare_data(continent, climate_vars, count_lags, count_rolling, clim_lags, clim_rolling)

south_shift = {"Winter": "Summer","Spring": "Fall","Summer": "Winter","Fall": "Spring"}
data['month_str'] = data.week_start.dt.month_name()
data['month'] = data.week_start.dt.month.astype('int8')
data['season_str'] = data.month.apply(get_season)
data.loc[data["lat"] < 0, "season_str"] = data.loc[data["lat"] < 0, "season_str"].map(south_shift)
data['equatorial'] = (np.abs(data.lat) < 30).astype('int8') 
data['season_cat'] = data['season_str'].astype('category')
data['season'] = data['season_cat'].cat.codes.astype('int8')
code_to_str = dict(enumerate(data['season_cat'].cat.categories))

data.rename(columns={'BIOME_NAME':'biome_str'}, inplace=True)
data = data[data.biome_str != 'N/A']
data['biome_str'] = data['biome_str'].str.replace(' ', '_')
data['biome_cat'] = data['biome_str'].astype('category')
data['biome'] = data['biome_cat'].cat.codes.astype('int8')

features_s = ['biome','equatorial']
features_t = ['season','month'] 
feature_names += features_s + features_t

data.dropna(subset=['hi', 'tmin', 'tmax', 'wbgt','ppt'], inplace=True)
data.loc[:,[c for c in data.columns if 'phase_' in c]] = data.loc[:,[c for c in data.columns if 'phase_' in c]].fillna(0.)
numeric_cols = data.drop(columns=['Conflict','Unrest']).select_dtypes(include=['number']).columns.tolist()


# split data
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

# prepare data for training and evaluation
N = 5000000
train_ids = np.random.choice(data_train.shape[0], size=min(N,data_train.shape[0]))

X = ppt_anomaly(data[feature_names])   # all rows, precipitation as anomaly (used for the predicted-risk maps)
X_train = data_train[feature_names] 
X_val = data_val[feature_names]

y_train = data_train[output_var]
y_val = data_val[output_var]
X_train, y_train = X_train.iloc[train_ids], y_train.iloc[train_ids]

for c in feature_names: print(c.upper())


# prepare data for SHAP analysis
df_plot = data.dropna(subset=feature_names).copy()
low, high = np.percentile(df_plot[numeric_cols], [2.5, 97.5], axis=0)
mask_out = np.all((df_plot[numeric_cols] >= low) & (df_plot[numeric_cols] <= high), axis=1)
df_plot = df_plot.loc[mask_out]

total_n = 300000 
groups = df_plot[["biome_str", "month",'Conflict','Unrest']].drop_duplicates()
n_groups = len(groups)
n_per_group = total_n // n_groups
rand_ids = []
for _, row in groups.iterrows():
  b = df_plot["biome_str"] == row["biome_str"]
  m = df_plot["month"] == row["month"]
  c = df_plot["Conflict"] == row["Conflict"]
  u = df_plot["Unrest"] == row["Unrest"]
  
  group_idx = df_plot[(b) & (m) & (c) & (u)].index
  
  sampled = np.random.choice(group_idx, size=min(n_per_group, len(group_idx)), replace=False)
  rand_ids.extend(sampled)
  
df_plot = df_plot.loc[rand_ids]
df_plot.reset_index(drop=True, inplace=True)

df_plot.to_csv(f'./results/plot_data_{line:02d}_{output_var}.csv', index=False)

with open(f'./results/features_{line:02d}_{output_var}.pkl', 'wb') as f:
  pickle.dump(feature_names, f)

ids_expl = np.random.choice(len(rand_ids), size=15000, replace=False)
X_explanation = df_plot.iloc[ids_expl,:][feature_names]

#####################################################################################
 
params_reg = {
    "objective": "binary:logistic",
    "eval_metric": 'aucpr', 
    "seed": 42,
    "device": "cuda",
    "tree_method": "hist",
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 30,
    "reg_lambda": 5.0,
    "scale_pos_weight": np.sqrt((y_train.shape[0] - y_train.sum()) / y_train.sum()),
}

dtrain_reg = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval_reg = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
dtest_reg = xgb.DMatrix(ppt_anomaly(df_plot[feature_names]), feature_names=feature_names)   # df_plot (and the saved CSV) keep raw precipitation
dall_reg = xgb.DMatrix(X[feature_names], feature_names=feature_names)

reg_model = xgb.train(params=params_reg, dtrain=dtrain_reg, num_boost_round=2000,
                      evals=[(dtrain_reg, "train"), (dval_reg, "val")],
                      early_stopping_rounds=100, verbose_eval=10)


unique_weeks = np.sort(data['week_start'].unique())
week_to_plot = unique_weeks[330]
data[f"p_mean_{output_var}"] = reg_model.predict(dall_reg)
#plot_preds(data, week_to_plot, mod_type='XGB', continent=continent, output_var=output_var, line=line)


#####################################################################################
# evaluate model performance on validation data and plot precision-recall curve

reg_model.set_param({"device": "cuda"})
y_prob = reg_model.predict(dval_reg)
y_pred = (y_prob >= 0.5).astype(int)
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(recall, precision)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
os.makedirs(f"./figures/shap/XGB/{continent}", exist_ok=True)
fig.savefig(f"./figures/shap/XGB/{continent}/PR_curve_{output_var}.png")

print('RESULTS')
print(classification_report(y_val, y_pred))
print(f'AUC-PR (average precision): {average_precision_score(y_val, y_prob):.5f}')

#####################################################################################
# SHAP values of the fitted model on the explanation sample (native TreeSHAP, log-odds scale; last column is the bias term)

shap_values_reg = reg_model.predict(dtest_reg, pred_contribs=True)[:, :-1]
shap_df_reg = pd.DataFrame(shap_values_reg, columns=feature_names)

shap_df_reg["lon"] = df_plot["lon"].values
shap_df_reg["lat"] = df_plot["lat"].values
shap_df_reg['week_start'] = df_plot['week_start'].values
shap_df_reg['season_name'] = df_plot['season_str'].values

shap_df_reg.to_csv(f'./results/shap_XGB_{line:02d}_{output_var}.csv', index=False)
