"""Figure 2. Hyperparameter-tuning vs data-quality-enhancement gain under
cold-start data scales, as two curves on a single shared axis.

HPO curve: site 212, cand_big vs default.
Augmentation curve: site 213, mixup_s05_f3 vs none.
Gain = pooled-mean relative change in validation MSE over 10 seeds.
x = cold-start training-data fraction: 15 / 25 / 50 / 100 %.

Output: s_data/cleaned/figs/fig_hp_vs_aug_scale_212.png
"""
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_LLM_DIR)
os.chdir(_PROJ_DIR)

CLEANED = 's_data/cleaned'
OUT = os.path.join(CLEANED, 'figs', 'fig_hp_vs_aug_scale_212.png')
SCALES = [0.15, 0.25, 0.5, 1.0]
XT = ['15%', '25%', '50%', '100%']
X = np.arange(len(SCALES))

BLUE, ORANGE = '#0072B2', '#E69F00'   # Okabe-Ito

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.9,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.fontsize': 9,
})


def pooled_gain(df, base_cfg, cand_cfg):
    base = df[df.config == base_cfg].val_mse.mean()
    cand = df[df.config == cand_cfg].val_mse.mean()
    return (cand / base - 1) * 100


def main():
    hp = pd.read_csv(f'{CLEANED}/llm_hp_scale_212.csv')
    aug = pd.read_csv(f'{CLEANED}/llm_scale_effect_213.csv')

    hp_curve = [pooled_gain(hp[hp.scale == s], 'default', 'cand_big') for s in SCALES]
    aug_curve = [pooled_gain(aug[aug.scale == s], 'none', 'mixup_s05_f3') for s in SCALES]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.axhline(0, color='#999999', lw=0.8, ls='--')
    ax.plot(X, hp_curve, color=BLUE, marker='o', lw=1.8, ms=6.5,
            markerfacecolor=BLUE, markeredgecolor=BLUE, label='Hyperparameter tuning')
    ax.plot(X, aug_curve, color=ORANGE, marker='D', lw=1.8, ms=6.5,
            markerfacecolor=ORANGE, markeredgecolor=ORANGE,
            label='Data-quality enhancement')

    ax.set_xticks(X)
    ax.set_xticklabels(XT)
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-32, 6)
    ax.set_xlabel('Cold-start data fraction')
    ax.set_ylabel('Validation MSE change (%)')
    ax.legend(frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3.5, labelsize=8.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print('saved ->', OUT)

    print('\nHPO gain % (cand_big vs default, pooled, 10 seeds):',
          [round(v, 2) for v in hp_curve])
    print('Data-quality gain % (mixup_s05_f3 vs none, pooled, 10 seeds):',
          [round(v, 2) for v in aug_curve])


if __name__ == '__main__':
    main()
