"""Ablation figure for the KANConv paper draft (Fig. 6), dual-panel.

(a) Replacement-style spline ablation: per-configuration mean delta% vs FC
    across the 10 main-table sites (HKUST 5 + DKASC 5). Shows what is missing
    when the learnable shape is removed/weakened.
(b) Removal-style component ablation: loss (%) when a single component is
    removed from the full KANConv_M5 (relative to M5's mean MSE), 10 sites.

Data sources (10 seeds each):
  HKUST Zone_J1/UG3/SQ2/Zone_D: kanconv_4sig_hkust.csv (FC/M5) + supplement
  HKUST SQ1: kanconv_8site_design.csv ; DKASC 5: kanconv_8site_design.csv
  NoLSTM: s_data/cleaned/ablation/ablation_C8.csv

Output: s_data/cleaned/figs/fig_ablation_delta10site.png
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

FIG = 's_data/cleaned/figs'
os.makedirs(FIG, exist_ok=True)

SCREEN = 's_data/cleaned/kanconv_hkust_screen.csv'
SUPP = 's_data/cleaned/kanconv_hkust_supplement.csv'
A4 = 's_data/cleaned/kanconv_4sig_hkust.csv'
D8 = 's_data/cleaned/kanconv_8site_design.csv'
NL = 's_data/cleaned/ablation/ablation_C8.csv'

HKUST5 = ['Zone_J1', 'UG3', 'SQ2', 'Zone_D', 'SQ1']
DKASC5 = ['11', '56', '67', '73', '79']
SITES = HKUST5 + DKASC5

# (a) replacement-style configs (FC is the baseline and is not drawn)
ABL_CONFIGS = ['FC', 'LinearCalib', 'Frozen', 'MLPConv', 'KANConv_M5']
ABL_PLOT = ['LinearCalib', 'Frozen', 'MLPConv', 'KANConv_M5']
# Okabe-Ito colorblind-safe, mutually distinguishable
ABL_COLORS = ['#E69F00', '#009E73', '#D55E00', '#0072B2']

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.9,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})


def load_all():
    # Use the same data source as the main table (Section 4.2): HKUST from
    # kanconv_4sig_hkust.csv (SQ1 falls back to kanconv_8site_design.csv),
    # DKASC from kanconv_8site_design.csv.
    a4 = pd.read_csv(A4) if os.path.exists(A4) else pd.DataFrame()
    d8 = pd.read_csv(D8) if os.path.exists(D8) else pd.DataFrame()
    frames = []
    for st in HKUST5:
        for cfg in ABL_CONFIGS:
            sub = a4[(a4['site'] == st) & (a4['config'] == cfg)]
            if len(sub) == 0:
                sub = d8[(d8['site'] == st) & (d8['config'] == cfg)]
            if len(sub):
                frames.append(sub.assign(site=st, config=cfg))
    for st in DKASC5:
        for cfg in ABL_CONFIGS:
            sub = d8[(d8['site'] == st) & (d8['config'] == cfg)]
            if len(sub):
                frames.append(sub.assign(site=st, config=cfg))
    return pd.concat(frames, ignore_index=True)


def site_delta_pct(df):
    """config -> [delta% vs FC per site]."""
    sm = df.groupby(['site', 'config'])['norm_MSE'].mean()
    out = {}
    for st in SITES:
        fc = sm.get((st, 'FC'))
        if fc is None:
            continue
        for cfg in ABL_CONFIGS:
            if (st, cfg) in sm.index:
                out.setdefault(cfg, []).append((sm[(st, cfg)] - fc) / fc * 100.0)
    return out


def component_loss():
    """Removal-style: relative loss (%) to full KANConv_M5 for removing
    spline (M5->FC), gate (M5->KANConv_base), LSTM (M5->M5_NoLSTM)."""
    a = pd.read_csv('s_data/cleaned/kanconv_4sig_hkust.csv'); a['site'] = a['site'].astype(str)
    b = pd.read_csv(D8); b['site'] = b['site'].astype(str)
    comb = pd.concat([a, b], ignore_index=True)
    df = pd.read_csv(NL); df['site'] = df['site'].astype(str)
    out = {}
    for site in SITES:
        m5 = comb[(comb.site == site) & (comb.config == 'KANConv_M5')].sort_values('seed')['norm_MSE'].values
        fc = comb[(comb.site == site) & (comb.config == 'FC')].sort_values('seed')['norm_MSE'].values
        base = comb[(comb.site == site) & (comb.config == 'KANConv_base')].sort_values('seed')['norm_MSE'].values
        m5nl = df[(df.site == site) & (df.config == 'M5_NoLSTM')].sort_values('seed')['norm_MSE'].values
        if len(m5) == 10 and len(fc) == 10:
            out.setdefault('spline', []).append((fc.mean() - m5.mean()) / m5.mean() * 100)
        if len(m5) == 10 and len(base) == 10:
            out.setdefault('gate', []).append((base.mean() - m5.mean()) / m5.mean() * 100)
        if len(m5) == 10 and len(m5nl) == 10:
            out.setdefault('lstm', []).append((m5nl.mean() - m5.mean()) / m5.mean() * 100)
    return out


def style_closed(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3.5, labelsize=8.5)


def main():
    df = load_all()
    deltas = site_delta_pct(df)
    comp = component_loss()

    # ---- (a) replacement-style spline ablation (gains vs FC, downward) ----
    # display label mapping: code/data name -> figure label (PC-KAN renamed in paper)
    LABEL_MAP = {'LinearCalib': 'LinearCalib', 'Frozen': 'Frozen',
                 'MLPConv': 'MLPConv', 'KANConv_M5': 'PC-KAN'}
    labels_a = [LABEL_MAP.get(c, c) for c in ABL_PLOT]
    means_a = [np.mean(deltas[c]) for c in ABL_PLOT]

    # ---- (b) removal-style component ablation (harm when removed, upward) ----
    labels_b = ['Spline', 'Gate', 'LSTM']
    means_b = [np.mean(comp['spline']), np.mean(comp['gate']), np.mean(comp['lstm'])]
    # (b) Okabe-Ito set, fully disjoint from (a){orange,green,red,blue}
    colors_b = ['#CC79A7', '#56B4E9', '#F0E442']

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), dpi=300,
                             gridspec_kw={'wspace': 0.32})

    # (a)
    ax = axes[0]
    x = np.arange(len(labels_a))
    bars = ax.bar(x, means_a, width=0.62, color=ABL_COLORS)
    ax.axhline(0, color='#666666', lw=0.8, ls='--')
    for b, val in zip(bars, means_a):
        ax.text(b.get_x() + b.get_width() / 2, val - 1.0, f'{val:+.1f}%',
                ha='center', va='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels_a, fontsize=8)
    ax.set_ylabel('Gain vs FC (%)', fontsize=9)
    ax.set_ylim(-9.5, 1.5)
    style_closed(ax)
    ax.text(0.5, -0.10, '(a)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', ha='center', va='top')

    # (b)
    ax = axes[1]
    x = np.arange(len(labels_b))
    bars = ax.bar(x, means_b, width=0.55, color=colors_b)
    for b, val in zip(bars, means_b):
        ax.text(b.get_x() + b.get_width() / 2, val + 1.0, f'+{val:.1f}%',
                ha='center', va='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels_b, fontsize=8)
    ax.set_ylabel('Harm when removed (%)', fontsize=9)
    ax.set_ylim(0, 22)
    style_closed(ax)
    ax.text(0.5, -0.10, '(b)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', ha='center', va='top')

    fig.tight_layout()
    out = os.path.join(FIG, 'fig_ablation_delta10site.png')
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
    print('(a)', [f'{LABEL_MAP.get(c, c)}:{np.mean(deltas[c]):+.2f}' for c in ABL_PLOT])
    print('(b)', [f'{l}:{m:+.1f}' for l, m in zip(labels_b, means_b)])


if __name__ == '__main__':
    main()
