"""Fig. 7: best-so-far validation-MSE convergence (methods only, HEBO removed).

Methods: LLM(a), TPE(c), random(f), DEHB(i), from llm_e2_small_*_{212,213}.csv.
Each curve is the mean best-so-far over R=3 runs (10 trials each, fixed temp 0.1).
The y-axis is piecewise-linear compressed (as in Fig. 8) so the early high
best-so-far values (up to ~0.2) and the converged range are both visible.

Output: s_data/cleaned/figs/fig_llm_joint_search.png
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
OUT = os.path.join(CLEANED, 'figs', 'fig_llm_joint_search.png')

GROUPS = {
    'a_llm_aug': 'LLM',
    'c_tpe_aug': 'TPE',
    'f_random_aug': 'Random',
    'i_dehb_aug': 'DEHB',
}
COLORS = {
    'a_llm_aug': '#0072B2',
    'c_tpe_aug': '#E69F00',
    'f_random_aug': '#009E73',
    'i_dehb_aug': '#D55E00',
}
MARKERS = {
    'a_llm_aug': 'o',
    'c_tpe_aug': '^',
    'f_random_aug': 's',
    'i_dehb_aug': 'D',
}

# piecewise-linear mapping data->display (display: 0=bottom,1=top)
_SEG = [(0.045, 0.0), (0.080, 0.72), (0.10, 0.80), (0.35, 1.0)]


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
    'legend.fontsize': 8,
})


def load_mean(group, site):
    runs = []
    for suffix in ['', '_run1', '_run2']:
        p = os.path.join(CLEANED, f'llm_e2_small_{group}_{site}{suffix}.csv')
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p).sort_values('trial')
        runs.append((df['trial'].values, np.minimum.accumulate(df['objective'].values)))
    if not runs:
        return None, None
    t0 = runs[0][0]
    curves = np.array([np.interp(t0, r[0], r[1]) for r in runs])
    return t0, curves.mean(axis=0)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9),
                             gridspec_kw={'wspace': 0.28})

    for ax, site, label in [(axes[0], 212, 'DKASC-212'), (axes[1], 213, 'DKASC-213')]:
        ax.set_yscale(FuncScale(ax, (_fwd, _inv)))
        for g, name in GROUPS.items():
            t0, mean = load_mean(g, site)
            if mean is None:
                continue
            ax.plot(t0, mean, '-', marker=MARKERS[g], color=COLORS[g], lw=1.7, ms=4,
                    label=name, markerfacecolor=COLORS[g],
                    markeredgecolor=COLORS[g])

        ax.set_xlim(0, 10.5)
        ax.tick_params(direction='out', length=3.5, labelsize=8)
        ax.set_title(f'{label}', fontsize=10, pad=6)
        ax.set_xlabel('Trial')
        ax.set_yticks([0.045, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.30])
        ax.set_yticklabels(['0.045', '0.05', '0.06', '0.07', '0.08',
                            '0.10', '0.15', '0.20', '0.30'], fontsize=7)
        ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=7.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if site == 212:
            ax.set_ylabel('Best-so-far val. MSE', fontsize=9)
        else:
            ax.set_yticklabels([])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
