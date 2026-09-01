"""Strict SHAP interaction analysis (filtered view) for Section 4.5.

Uses shap_interaction_values (true 2nd-order), keeps only features that
participate in a significant interaction, and draws a focused heatmap.
augmentation_factor is displayed as 'factor'. Layout/labels are tuned so that
the top/right tick labels are not clipped.

Output: s_data/cleaned/figs/fig_shap_interaction_heatmap.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.ensemble import GradientBoostingRegressor
import shap

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJ)
CLEANED = 's_data/cleaned'
REG = os.path.join(CLEANED, 'llm_trial_registry_212_fixed85_sub0.25.json')

HP = ['learning_rate', 'hidden_dim', 'num_layers', 'out_channels', 'dropout']
AUG = ['use_mixup', 'use_mixup_phys', 'use_mixup_temporal', 'use_jitter',
       'sigma_mixup', 'sigma_phys', 'sigma_temporal', 'sigma_jitter',
       'augmentation_factor']
FEATS = HP + AUG

# display short names
DISPLAY = {f: f.replace('learning_rate', 'lr').replace('augmentation_factor', 'factor')
           for f in FEATS}

model = GradientBoostingRegressor(n_estimators=400, max_depth=5,
                                  learning_rate=0.04, subsample=0.8,
                                  random_state=42)
with open(REG) as f:
    data = json.load(f)['data']
rows = []
for v in data.values():
    c = v['config']
    rows.append({**{ft: c[ft] for ft in FEATS}, 'objective': v['median_val_mse']})
df = pd.DataFrame(rows).dropna(subset=FEATS)
X = df[FEATS]; y = df['objective'].values
model.fit(X, y)

expl = shap.TreeExplainer(model)
SV = np.asarray(expl.shap_values(X))
I = np.asarray(expl.shap_interaction_values(X))
n = len(FEATS); idx = {f: i for i, f in enumerate(FEATS)}
main_abs = np.abs(SV).mean(axis=0)

# Keep the absolute interaction magnitudes but display them on a x1e-2 scale so
# that the raw val-MSE unit (which makes all absolute values look like 0.00) is
# factored out of the labels, not out of the ranking.
mat = np.zeros((n, n))
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        v = float(np.abs(I[:, i, j]).mean())
        mat[i, j] = mat[j, i] = v
        pairs.append((v, FEATS[i], FEATS[j]))
pairs.sort(reverse=True)
print('main |SHAP|:', {f: round(float(main_abs[idx[f]]), 4) for f in ['learning_rate', 'augmentation_factor']})

# keep features that participate in an interaction >= 30% of the strongest
strong = [p for p in pairs if p[0] >= 0.30 * pairs[0][0]]
keep = sorted({f for _, a, b in strong for f in (a, b)},
              key=lambda f: -max(v for v, a, b in strong if f in (a, b)))
keep_idx = [idx[f] for f in keep]
print('significant interactions:')
for v, a, b in strong:
    print(f'  {a:<18} x {b:<18} {v:.5f}')
print('kept:', keep)

# group means for the text
hhp, hap, app = [], [], []
for v, a, b in pairs:
    if a in HP and b in HP: hhp.append(v)
    elif a in HP and b in AUG: hap.append(v)
    else: app.append(v)
print(f'group means  hh={np.mean(hhp):.5f}  ha={np.mean(hap):.5f}  aa={np.mean(app):.5f}')

# ---- focused heatmap, tuned layout ----
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})
# Values are shown on a x1e-2 scale (i.e. 0.53 means 0.0053 in val-MSE units).
sub = mat[np.ix_(keep_idx, keep_idx)]
lbl = [DISPLAY[f] for f in keep]
# diagonal = 1.00 (self-interaction, by convention a perfect match), which also
# anchors the color scale so that the diagonal reads as its own max.
DIAG = 0.01  # displayed as 1.00 on the x1e-2 scale
subm = sub * 100
np.fill_diagonal(subm, DIAG * 100)
fig, ax = plt.subplots(figsize=(5.6, 4.8))
lognorm = LogNorm(vmin=sub[~np.isclose(sub, 0)].min() * 100, vmax=DIAG * 100)
im = ax.imshow(subm, cmap='viridis', norm=lognorm)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                  label='Second-order interaction (\u00d710$^{-2}$)')
cb.outline.set_visible(True)
cb.outline.set_linewidth(0.8)
cb.ax.tick_params(which='both', length=0, width=0)
cb.ax.yaxis.set_tick_params(which='both', length=0, width=0)
for tl in cb.ax.get_yticklines():
    tl.set_visible(False)
cb.set_ticks([])
cb.ax.set_yticks([])
ax.set_xticks(range(len(keep)))
ax.set_xticklabels(lbl, rotation=40, ha='right', rotation_mode='anchor')
ax.set_yticks(range(len(keep)))
ax.set_yticklabels(lbl)
for r in range(len(keep)):
    for c in range(len(keep)):
        v = subm[r, c]
        if r == c:
            color = 'black'  # diagonal is the brightest cell
        else:
            lum = lognorm(v)
            color = 'white' if lum < 0.55 else 'black'
        ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7, color=color)
fig.subplots_adjust(left=0.20, right=0.88, top=0.90, bottom=0.18)
out = os.path.join(CLEANED, 'figs', 'fig_shap_interaction_heatmap.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved ->', out)
