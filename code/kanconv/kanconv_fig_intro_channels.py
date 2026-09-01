# -*- coding: utf-8 -*-
"""Introduction figure 1: per-channel scatter of each meteorological feature
vs PV power for the HKUST Zone_J1 site (normalized to [-1,1]).

Layout: 2 rows x 3 cols, all six x-labels shown, 5 ticks per axis, one shared
left y-label, distinct Okabe-Ito colors per channel, Times New Roman.

Outputs:
  - s_data/cleaned/figs/fig_intro_channels.png   (for the LaTeX paper)
  - eps_export/fig_intro_channels.{eps,pdf,svg}  (vector formats for submission)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))

CLEANED = 's_data/cleaned'
PNG_OUT = os.path.join(CLEANED, 'figs', 'fig_intro_channels.png')
EPS_OUT = os.path.join(PROJECT_ROOT.parent, 'eps_export')  # project-root sibling
CSV = os.path.join(CLEANED, 'hkust', 'hkust_Zone_J1.csv')

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})


def norm(s):
    s = s.astype(float).values
    s = s - s.min()
    s = 2 * (s / max(s.max(), 1e-9)) - 1
    return s


def main():
    df = pd.read_csv(CSV).dropna(subset=['Active_Power', 'GHI', 'Temperature',
                                         'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos'])
    power = norm(df['Active_Power'])
    channels = [
        ('GHI',           norm(df['GHI']),         '#0072B2'),  # blue
        ('Temperature',   norm(df['Temperature']), '#D55E00'),  # vermillion
        ('Humidity',      norm(df['Humidity']),    '#009E73'),  # bluish green
        ('Wind Speed',    norm(df['Wind_Speed']),  '#E69F00'),  # orange
        ('WD sin',        norm(df['WD_sin']),      '#CC79A7'),  # reddish purple
        ('WD cos',        norm(df['WD_cos']),      '#56B4E9'),  # sky blue
    ]

    N = min(len(power), 150000)
    idx = np.random.RandomState(0).choice(len(power), size=N, replace=False)
    power = power[idx]
    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.0))
    for ax, (name, x, c) in zip(axes.ravel(), channels):
        xs = x[idx]
        ax.scatter(xs, power, s=4, alpha=0.4, color=c, edgecolors='none')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=11)
        ax.set_xlabel(name, fontsize=13)

    fig.supylabel('Power (normalized)', fontsize=13)
    for r in range(2):
        for c in range(1, 3):
            axes[r, c].set_yticklabels([])
    fig.tight_layout(rect=[0.03, 0, 1, 1])

    os.makedirs(os.path.dirname(PNG_OUT), exist_ok=True)
    fig.savefig(PNG_OUT, dpi=300, bbox_inches='tight')
    print('saved', PNG_OUT)

    os.makedirs(EPS_OUT, exist_ok=True)
    base = 'fig_intro_channels'
    for ext in ('eps', 'pdf', 'svg'):
        dst = os.path.join(EPS_OUT, base + '.' + ext)
        fig.savefig(dst, format=ext, dpi=300)
        print('wrote', os.path.basename(dst))


if __name__ == '__main__':
    main()
