"""Fig. 11: effect of the augmentation factor under each learning-rate group.

Grouped bar chart on the mixup-enabled cold-start subset (n=193): the x-axis is
the learning-rate group (low/mid/high) and within each group three bars show the
mean validation MSE for a low, mid, or high augmentation factor. At low learning
rates a larger factor lowers the MSE, whereas at high learning rates it raises
it, showing the factor's value depends on the learning rate.

Output: s_data/cleaned/figs/fig_lr_factor_interaction.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJ)
CLEANED = 's_data/cleaned'
REG = os.path.join(CLEANED, 'llm_trial_registry_212_fixed85_sub0.25.json')
OUT = os.path.join(CLEANED, 'figs', 'fig_lr_factor_interaction.png')

HP = ['learning_rate', 'hidden_dim', 'num_layers', 'out_channels', 'dropout']
AUG = ['use_mixup', 'use_mixup_phys', 'use_mixup_temporal', 'use_jitter',
       'sigma_mixup', 'sigma_phys', 'sigma_temporal', 'sigma_jitter',
       'augmentation_factor']
FEATS = HP + AUG

with open(REG) as f:
    data = json.load(f)['data']
rows = []
for v in data.values():
    c = v['config']
    rows.append({**{ft: c[ft] for ft in FEATS}, 'objective': v['median_val_mse']})
df = pd.DataFrame(rows).dropna(subset=FEATS)
m = df[df['use_mixup'] == True].copy()

m['lr_g'] = pd.qcut(m['learning_rate'], 3, labels=['Low lr', 'Mid lr', 'High lr'])
m['fac_g'] = pd.qcut(m['augmentation_factor'], 3, labels=['Low factor', 'Mid factor', 'High factor'])
tab = m.pivot_table(index='lr_g', columns='fac_g', values='objective', aggfunc='mean')
# order: lr Low -> High, factor Low -> High
lr_order = ['Low lr', 'Mid lr', 'High lr']
fac_order = ['Low factor', 'Mid factor', 'High factor']
vals = tab.loc[lr_order, fac_order].values  # rows=lr, cols=factor

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})
colors = ['#0072B2', '#E69F00', '#D55E00']  # low/mid/high factor
fig, ax = plt.subplots(figsize=(6.0, 3.8))
x = np.arange(len(lr_order))
w = 0.26
for j, fac in enumerate(fac_order):
    ax.bar(x + (j - 1) * w, vals[:, j], width=w, color=colors[j],
           label=fac.replace(' factor', ''))
    for i in range(len(lr_order)):
        ax.text(x[i] + (j - 1) * w, vals[i, j] + 0.01, f'{vals[i, j]:.3f}',
                ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(['Low lr', 'Mid lr', 'High lr'])
ax.set_xlabel('Learning rate')
ax.set_ylabel('Mean validation MSE')
ax.legend(title='Augmentation factor', frameon=False, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='out', length=3, labelsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print('saved ->', OUT)
print(tab.round(4))
