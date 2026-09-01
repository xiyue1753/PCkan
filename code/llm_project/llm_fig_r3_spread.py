"""Fig. 8: R=3 cross-run spread of best validation MSE (scatter + boxplot).

For each method, a boxplot summarises the R=3 distribution with the three
jittered raw best-validation values overlaid. The y-axis is piecewise-linear
compressed so every true statistic (including HEBO's off-scale outliers at
~0.19 / ~0.28) is visible in one continuous axis with labelled ticks.

Output: s_data/cleaned/figs/fig_llm_r3_spread.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.scale import FuncScale

_LLM = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_LLM)
os.chdir(_PROJ)

CLEANED = 's_data/cleaned'
OUT = os.path.join(CLEANED, 'figs', 'fig_llm_r3_spread.png')

METHODS = [
    ('a_llm_aug', 'LLM'),
    ('c_tpe_aug', 'TPE'),
    ('f_random_aug', 'Random'),
    ('i_dehb_aug', 'DEHB'),
    ('h_hebo_aug', 'HEBO'),
]
COLORS = {
    'a_llm_aug': '#0072B2',
    'c_tpe_aug': '#E69F00',
    'f_random_aug': '#009E73',
    'i_dehb_aug': '#D55E00',
    'h_hebo_aug': '#CC79A7',
}

# piecewise-linear mapping data->display (display: 0=bottom,1=top)
_SEG = [(0.045, 0.0), (0.080, 0.72), (0.10, 0.80), (0.30, 1.0)]


def _fwd(y):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    for i in range(len(_SEG) - 1):
        y0, d0 = _SEG[i]
        y1, d1 = _SEG[i + 1]
        m = (y >= y0) & (y < y1)
        out[m] = d0 + (y[m] - y0) / (y1 - y0) * (d1 - d0)
    out[y >= _SEG[-1][0]] = _SEG[-1][1]
    out[y < _SEG[0][0]] = _SEG[0][1]
    return out


def _inv(d):
    d = np.asarray(d, dtype=float)
    out = np.empty_like(d)
    for i in range(len(_SEG) - 1):
        y0, d0 = _SEG[i]
        y1, d1 = _SEG[i + 1]
        m = (d >= d0) & (d < d1)
        out[m] = y0 + (d[m] - d0) / (d1 - d0) * (y1 - y0)
    out[d >= _SEG[-1][1]] = _SEG[-1][0]
    out[d < _SEG[0][1]] = _SEG[0][0]
    return out


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.9,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})


def best_val(group, site, suffix):
    p = f'{CLEANED}/llm_e2_small_{group}_{site}{suffix}.csv'
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)['objective'].min()


def collect(group, site):
    return [best_val(group, site, s) for s in ['', '_run1', '_run2']]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9),
                             gridspec_kw={'wspace': 0.30})

    rng = np.random.default_rng(0)
    for ax, site, label in [(axes[0], 212, 'DKASC-212'), (axes[1], 213, 'DKASC-213')]:
        vals = [collect(g, site) for g, _ in METHODS]
        vals = [v for v in vals if all(x is not None for x in v)]

        ax.set_yscale(FuncScale(ax, (_fwd, _inv)))
        xpos = np.arange(len(METHODS))

        # boxplot summary on the compressed axis
        bp = ax.boxplot(vals, positions=xpos, widths=0.5, patch_artist=True,
                        boxprops=dict(linewidth=0.8, edgecolor='black'),
                        medianprops=dict(color='black', linewidth=0.8),
                        whiskerprops=dict(color='black', linewidth=0.8),
                        capprops=dict(color='black', linewidth=0.8),
                        showfliers=False)
        for patch, (g, _) in zip(bp['boxes'], METHODS):
            patch.set_facecolor(COLORS[g])
            patch.set_alpha(0.35)

        # jittered scatter overlay (true values, on compressed axis)
        for xi, ((g, name), vv) in enumerate(zip(METHODS, vals)):
            jitter = rng.uniform(-0.12, 0.12, len(vv))
            ax.scatter(xi + jitter, vv, s=35, color=COLORS[g],
                       edgecolor='black', linewidth=0.4, zorder=3)

        ax.set_xticks(xpos)
        ax.set_xticklabels([m[1] for m in METHODS], fontsize=8)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.set_title(f'{label}', fontsize=10, pad=6)
        ax.set_xlabel('Method')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3.5, labelsize=8)
        # labelled ticks across all segments
        ax.set_yticks([0.045, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.30])
        ax.set_yticklabels(['0.045', '0.05', '0.06', '0.07', '0.08',
                            '0.10', '0.15', '0.20', '0.30'], fontsize=7)
        if site == 212:
            ax.set_ylabel('Best val. MSE (3 runs)', fontsize=9)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
