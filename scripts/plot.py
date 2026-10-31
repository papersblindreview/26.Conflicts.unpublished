import argparse
import glob
import os
import matplotlib
matplotlib.use('Agg')          # no display needed on a cluster node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Self-contained: no other project files needed.
# Usage:  python scripts/plot.py                          25th-75th percentile across bootstrap replicates (default)
#         python scripts/plot.py --interval ci95          95% interval across bootstrap replicates
#         python scripts/plot.py --interval both          thin = 95%, thick = IQR across replicates
#         python scripts/plot.py --interval within        biome figure only: within-biome IQR of SHAP values
#                                                    (median over replicates)
#         python scripts/plot.py --boot ./results/boot_explain --tag explain
#         python scripts/plot.py --sharey all             sustainability figure: one y-axis for both outcomes (default: row)
#         python scripts/plot.py --figs fitted            only the fitted-model importance figure (F1_importance.png)
# Biome figure points are not color-coded; whether the 95% interval excludes zero is in the CSV ('robust').
parser = argparse.ArgumentParser()
parser.add_argument('--boot', default='./results/boot')
parser.add_argument('--res', default='./results')
parser.add_argument('--out', default='./figures')
parser.add_argument('--line', type=int, default=1)
parser.add_argument('--tag', default='uq', help='suffix for output file names')
parser.add_argument('--interval', choices=['ci95', 'iqr', 'both', 'within'], default='iqr')
parser.add_argument('--figs', choices=['all', 'fitted', 'boot'], default='all',
                    help="fitted = Fig. 1 from the main-run SHAP files only; boot = bootstrap figures only; all = both")
parser.add_argument('--sharey', choices=['none', 'row', 'all'], default='row',
                    help='y-axis sharing in the sustainability figure: row = within each outcome (default), all = one axis for both outcomes')
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
suffix = args.tag if args.interval == 'ci95' else f'{args.tag}_{args.interval}'
band_mode = 'ci95' if args.interval == 'within' else args.interval   # 'within' only applies to the biome figure


# ----------------------------- helpers -----------------------------
def load_boot(path):
    files = sorted(glob.glob(f'{path}/shap_*.csv'))
    print(f'{len(files)} replicate files in {path}')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _pct(q):
    return lambda x: np.percentile(x, q)


def biome_summary(boot):
    """Per outcome x variable x biome: median of the biome medians, intervals across replicates, sign agreement.
    lo/hi = 95% interval, lo50/hi50 = IQR across replicates, box_lo/box_hi = within-biome IQR of SHAP values
    (median over replicates; only available from the bootstrap run, not the explain-only run)."""
    keys = ['output', 'feature', 'group']
    d = boot[(boot.kind == 'biome') & (boot.stat == 'median')]
    s = d.groupby(keys)['value'].agg(
        est='median', lo=_pct(2.5), hi=_pct(97.5), lo50=_pct(25), hi50=_pct(75),
        pos=lambda x: (x > 0).mean(), n='count').reset_index()
    s['sign_agree'] = np.maximum(s['pos'], 1 - s['pos'])
    s['robust'] = (s['lo'] > 0) | (s['hi'] < 0)   # 95% interval excludes zero
    q = boot[(boot.kind == 'biome') & (boot.stat.isin(['q25', 'q75']))]
    if len(q):
        w = (q.pivot_table(index=keys, columns='stat', values='value', aggfunc='median')
             .rename(columns={'q25': 'box_lo', 'q75': 'box_hi'}).reset_index())
        s = s.merge(w, on=keys, how='left')
    return s


def importance_summary(boot, top_k=15):
    """Mean |SHAP| per predictor with intervals and how often it ranks in the top_k."""
    d = boot[boot.kind == 'importance'].copy()
    d['rank'] = d.groupby(['output', 'rep'])['value'].rank(ascending=False)
    s = d.groupby(['output', 'feature']).agg(
        est=('value', 'median'), lo=('value', _pct(2.5)), hi=('value', _pct(97.5)),
        lo50=('value', _pct(25)), hi50=('value', _pct(75)),
        top_freq=('rank', lambda r: (r <= top_k).mean()),
        median_rank=('rank', 'median')).reset_index()
    return s.sort_values(['output', 'est'], ascending=[True, False])


def dependence_summary(boot):
    """Binned dependence curves: x position plus median and bands of the mean SHAP per bin."""
    d = boot[boot.kind == 'dependence']
    x = (d[d.stat == 'x_median'].groupby(['output', 'feature', 'group'])['value']
         .median().rename('x').reset_index())
    y = (d[d.stat == 'mean_shap'].groupby(['output', 'feature', 'group'])['value']
         .agg(est='median', lo=_pct(2.5), hi=_pct(97.5), lo50=_pct(25), hi50=_pct(75)).reset_index())
    s = x.merge(y, on=['output', 'feature', 'group'])
    s['bin'] = s['group'].astype(int)
    return s.sort_values(['output', 'feature', 'bin'])


def plot_dependence_band(dep, output, feature, ax, xlabel=None, mode='ci95'):
    d = dep[(dep.output == output) & (dep.feature == feature)]
    if mode in ('ci95', 'both'):
        ax.fill_between(d['x'], d['lo'], d['hi'], alpha=0.2 if mode == 'both' else 0.3, color='C0')
    if mode in ('iqr', 'both'):
        ax.fill_between(d['x'], d['lo50'], d['hi50'], alpha=0.4, color='C0')
    ax.plot(d['x'], d['est'], lw=2, color='C0')
    ax.axhline(0, color='red', ls='dashed')
    ax.set_xlabel(xlabel or feature)
    ax.set_ylabel('SHAP value')
    ax.grid(True)


def plot_biome_intervals(summary, df_plot, vars_list, vars_titles, biome_name_map,
                         outputs, interval='ci95', xlim=(-0.29, 0.29), save_path=None):
    """Point = median SHAP across replicates. Bar depends on `interval`. All points are drawn the same way."""
    order = (df_plot.assign(equator_dist=df_plot['lat'].abs())
             .groupby('biome_str')['equator_dist'].mean().sort_values().index.tolist())
    fig, axes = plt.subplots(len(outputs), len(vars_list), figsize=(12, 12), sharex=True)
    for i, out in enumerate(outputs):
        for j, v in enumerate(vars_list):
            ax = axes[i, j]
            d = summary[(summary.output == out) & (summary.feature == v)].set_index('group')
            for k, b in enumerate(order, start=1):
                if b not in d.index:
                    continue
                r = d.loc[b]
                if interval == 'ci95':
                    ax.hlines(k, r['lo'], r['hi'], color='k', lw=2)
                elif interval == 'iqr':
                    ax.hlines(k, r['lo50'], r['hi50'], color='k', lw=3)
                elif interval == 'both':
                    ax.hlines(k, r['lo'], r['hi'], color='k', lw=1)
                    ax.hlines(k, r['lo50'], r['hi50'], color='k', lw=4)
                elif interval == 'within':
                    bl, bh = r.get('box_lo', np.nan), r.get('box_hi', np.nan)
                    if not (np.isnan(bl) or np.isnan(bh)):
                        ax.hlines(k, bl, bh, color='k', lw=3)
                ax.plot(r['est'], k, 'o', ms=9, mfc='k', mec='k')
            ax.axvline(0, color='red', linestyle='dashed')
            ax.set_yticks(range(1, len(order) + 1))
            if j == 0:
                ax.set_ylabel(out, fontsize=20, fontweight='bold')
                ax.set_yticklabels([biome_name_map.get(b, b) for b in order], fontsize=18)
            else:
                ax.set_yticklabels([])
            if i == len(outputs) - 1:
                ax.set_xlabel('SHAP value', fontsize=18)
            if i == 0:
                ax.set_title(vars_titles[j], fontsize=13)
            ax.set_xlim(*xlim)
            ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_fitted_importance(res, out, line, outputs, top_k=15):
    """Fig. 1: mean |SHAP| of the fitted (main-run) models, from results/shap_XGB_<line>_<outcome>.csv.
    Shows the union of each outcome's top_k predictors; blue = unrest, orange = conflict."""
    non_feature = ['lon', 'lat', 'week_start', 'season_name']
    imp = {}
    for o in outputs:
        s = pd.read_csv(f'{res}/shap_XGB_{line:02d}_{o}.csv')
        imp[o] = s.drop(columns=[c for c in non_feature if c in s]).abs().mean()
    df = pd.DataFrame(imp)
    keep = sorted(set().union(*[set(df[o].nlargest(top_k).index) for o in outputs]), key=lambda f: -df.loc[f].max())
    df = df.loc[keep]
    print(f'Fig. 1: {len(df)} unique predictors in the top {top_k} of at least one outcome')

    colors = {'Unrest': 'C0', 'Conflict': 'C1'}
    y = np.arange(len(df))
    h = 0.4
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.5))
    for k, o in enumerate(outputs):
        ax.barh(y + (k - 0.5) * h, df[o].to_numpy(), height=h, color=colors[o], label=o)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_ylim(len(df) - 0.5, -0.5)   # largest predictor at the top, no empty rows
    ax.set_xlabel('mean |SHAP value| (log-odds)')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(f'{out}/F1_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    df.to_csv(f'{out}/F1_importance_values.csv')


# ----------------------------- analysis -----------------------------
biome_name_map = {
    "Tropical_&_Subtropical_Moist_Broadleaf_Forests": "Tropical Moist Forests",
    "Tropical_&_Subtropical_Grasslands,_Savannas_&_Shrublands": "Tropical Grasslands",
    "Tropical_&_Subtropical_Dry_Broadleaf_Forests": "Tropical Dry Forests",
    "Tropical_&_Subtropical_Coniferous_Forests": "Tropical Coniferous Forests",
    "Temperate_Grasslands,_Savannas_&_Shrublands": "Temperate Grasslands",
    "Temperate_Conifer_Forests": "Temperate Conifer Forests",
    "Temperate_Broadleaf_&_Mixed_Forests": "Temperate Broadleaf",
    "Montane_Grasslands_&_Shrublands": "Montane Grasslands",
    "Mediterranean_Forests,_Woodlands_&_Scrub": "Mediterranean Forests",
    "Mangroves": "Mangroves",
    "Flooded_Grasslands_&_Savannas": "Flooded Grasslands",
    "Deserts_&_Xeric_Shrublands": "Deserts",
    "Boreal_Forests/Taiga": "Boreal Forests",
}
outputs = ['Unrest', 'Conflict']

if args.figs in ('all', 'fitted'):
    plot_fitted_importance(args.res, args.out, args.line, outputs)
if args.figs == 'fitted':
    raise SystemExit('figures written to ' + args.out)

boot = load_boot(args.boot)
print(boot.groupby('output')['rep'].nunique().rename('replicates'))

# ---- sanity check: main-run mean |SHAP| inside the 95% bootstrap interval? ----
imp = importance_summary(boot)
for out in outputs:
    f = f'{args.res}/shap_XGB_{args.line:02d}_{out}.csv'
    if not os.path.exists(f):
        continue
    s = pd.read_csv(f)
    main = s.drop(columns=[c for c in ['lon', 'lat', 'week_start', 'season_name'] if c in s]).abs().mean()
    d = imp[imp.output == out].set_index('feature').join(main.rename('main'))
    inside = ((d.main >= d.lo) & (d.main <= d.hi)).mean()
    print(f'{out}: main-run value inside the 95% interval for {inside:.0%} of predictors')

# ---- Fig. 1: importance with intervals ----
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
for ax, out in zip(axes, outputs):
    d = imp[imp.output == out].head(15).iloc[::-1]
    lo, hi = (d['lo50'], d['hi50']) if band_mode == 'iqr' else (d['lo'], d['hi'])
    ax.barh(d['feature'], d['est'], xerr=[d['est'] - lo, hi - d['est']], capsize=3)
    if band_mode == 'both':
        ax.errorbar(d['est'], d['feature'], xerr=[d['est'] - d['lo50'], d['hi50'] - d['est']],
                    fmt='none', ecolor='k', elinewidth=4)
    for y, (_, r) in enumerate(d.iterrows()):
        ax.text(ax.get_xlim()[1], y, f" top-15 in {r['top_freq']:.0%}", va='center', fontsize=8)
    ax.set_title(out)
    ax.set_xlabel('mean |SHAP|')
plt.tight_layout()
plt.savefig(f'{args.out}/F1_importance_{suffix}.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---- Fig. 3: biome contributions with intervals ----
bio = biome_summary(boot)
bio.to_csv(f'{args.out}/biome_intervals_{suffix}.csv', index=False)
print(bio.groupby(['output', 'feature'])['robust'].agg(['sum', 'count']).rename(columns={'sum': 'excl_zero'}))
if args.interval == 'within' and 'box_lo' not in bio:
    print('no within-biome quartiles in this folder (explain-only run); use --interval iqr instead')
df_plot = pd.read_csv(f'{args.res}/plot_data_{args.line:02d}_Conflict.csv', usecols=['biome_str', 'lat'])
bio_vars = ['tmin_rollmean_4', 'tmax_rollmean_4', 'ppt']
bio_titles = ['4-week Avg. Min Temp', '4-week Avg. Max Temp', 'Precipitation']
plot_biome_intervals(bio, df_plot, bio_vars, bio_titles, biome_name_map, outputs, interval=args.interval,
                     save_path=f'{args.out}/F3_seasonal_{suffix}.png')

# ---- Fig. 2: sustainability dependence bands ----
dep = dependence_summary(boot)
dep_vars = ['EPI_index', 'MF_pc_t', 'ECO_pc_gha', 'GHG_pc_t']
dep_labels = ['EPI score', 'Material footprint (t/person)', 'Ecological footprint (ha/person)',
              'GHG emissions (t CO2e/person)']
sharey = {'none': False, 'row': 'row', 'all': True}[args.sharey]
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=sharey)
for i, out in enumerate(outputs):
    for j, v in enumerate(dep_vars):
        plot_dependence_band(dep, out, v, axes[i, j], xlabel=dep_labels[j], mode=band_mode)
        if j == 0:
            axes[i, j].set_ylabel(f'{out}\nSHAP value')
        elif args.sharey != 'none':
            axes[i, j].set_ylabel('')
plt.tight_layout()
plt.savefig(f'{args.out}/F2_sustainability_{suffix}.png', dpi=300)
plt.close(fig)
print('figures written to', args.out)