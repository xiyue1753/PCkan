"""SHAP analysis on the full joint config->val-MSE pool (site 212).

Source: the shared trial registry llm_trial_registry_212_fixed85_sub0.25.json,
which contains EVERY config evaluated by the comparison experiment (LLM/TPE/DEHB/
random/HEBO, all at fixed-85ep + subset-0.25 + batch-1024) plus the temperature
scan (0.3/0.5/0.9). Trains a GBDT surrogate for median validation MSE, validates
with 5-fold CV, then computes SHAP feature importance and key interactions.

Output: s_data/cleaned/figs/fig_shap_importance.png + printed diagnostics.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
import shap

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJ)

CLEANED = 's_data/cleaned'
REG = os.path.join(CLEANED, 'llm_trial_registry_212_fixed85_sub0.25.json')
OUT = os.path.join(CLEANED, 'figs', 'fig_shap_importance.png')

HP = ['learning_rate', 'hidden_dim', 'num_layers', 'out_channels', 'dropout']
AUG = ['use_mixup', 'use_mixup_phys', 'use_mixup_temporal', 'use_jitter',
       'sigma_mixup', 'sigma_phys', 'sigma_temporal', 'sigma_jitter',
       'augmentation_factor']
FEATS = HP + AUG


def main():
    with open(REG) as f:
        reg = json.load(f)
    data = reg['data']
    print(f'=== SHAP pool (full registry) ===')
    print(f'registry configs: {len(data)}')

    rows = []
    for k, v in data.items():
        c = v['config']
        rows.append({**{ft: c[ft] for ft in FEATS},
                     'objective': v['median_val_mse'],
                     'test': v['test_mse_median']})
    df = pd.DataFrame(rows)
    # remove rows with missing/NaN features
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    print(f'valid configs (features complete): {len(df)}')

    X = df[FEATS].copy()
    y = df['objective'].values
    print(f'\nval MSE: min={y.min():.4f} med={np.median(y):.4f} '
          f'max={y.max():.4f} std={y.std():.4f} '
          f'(low <0.08 = good; >0.1 = poor/high-val region)')

    # ---- surrogate with 5-fold CV ----
    model = GradientBoostingRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = cross_val_score(model, X, y, cv=cv, scoring='r2')
    maes = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    print(f'\n[surrogate CV] R2 = {r2s.mean():.3f} ± {r2s.std():.3f} '
          f'(per-fold: {np.round(r2s,3)})')
    print(f'[surrogate CV] MAE = {-maes.mean():.4f} ± {maes.std():.4f}')

    # ---- full model + SHAP ----
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1]
    print('\n[feature importance (mean |SHAP|)]')
    for i in order:
        print(f'  {FEATS[i]:<22} {imp[i]:.4f}')

    print('\n[key pairwise interaction (mean |phi_i * phi_j|)]')
    top = order[:6]
    pairs = {}
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            pairs[(FEATS[a], FEATS[b])] = np.mean(np.abs(sv[:, a] * sv[:, b]))
    for (n1, n2), v in sorted(pairs.items(), key=lambda kv: -kv[1])[:6]:
        print(f'  {n1} × {n2}: {v:.5f}')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })
    # beeswarm of the top-10 factors (SHAP value per sample, colored by the
    # feature value), which keeps the learning-rate dominance readable without
    # a single bar overwhelming the plot.
    top_idx = np.argsort(imp)[::-1][:10]
    names = [FEATS[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    shap.summary_plot(sv[:, top_idx], X[names].values,
                      feature_names=names, show=False, max_display=len(top_idx))
    ax = plt.gca()
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved -> {OUT}')

    print('\n[credibility]')
    print(f'  n={len(df)}, p={len(FEATS)}, ratio={len(df)/len(FEATS):.1f}')
    print(f'  R2 mean={r2s.mean():.3f}')


if __name__ == '__main__':
    main()
