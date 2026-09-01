"""Fig. 4 (aux): enhancement-family illustration, 1x2 schematic.

Two categories of data-quality enhancement, shown as waveforms (x = time step,
channel-agnostic):
  (1) Mixup family: convex combination of two sample waveforms into a new one;
      a note indicates that the sample pair may be restricted to the same
      GHI-state or hour-of-day bin (mixup_phys / mixup_temporal).
  (2) GHI-targeted jitter: i.i.d. noise added to the irradiance-channel signal.

Style: Times New Roman, Okabe-Ito colours.
Output: s_data/cleaned/figs/fig_enh_family.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJ)
OUT = 's_data/cleaned/figs/fig_enh_family.png'

GREY, BLUE = '#999999', '#0072B2'

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})


def base_wave(n=48, seed=1, peak=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    tn = t / n
    w = 0.8 * np.exp(-((tn - peak) * 4.2) ** 2)
    w = w + rng.normal(0, 0.03, n)
    w = np.clip(w, 0, 1)
    return t, w


def style(ax, title):
    ax.set_xlabel('Time step')
    ax.set_ylim(-0.15, 1.25)
    ax.tick_params(direction='out', length=3, labelsize=7.5)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6),
                             gridspec_kw={'wspace': 0.42})

    # ---- (1) Mixup family ----
    ax = axes[0]
    t, wA = base_wave(peak=0.32, seed=1)
    _, wB = base_wave(peak=0.68, seed=4)
    ax.plot(t, wA, color='#0072B2', lw=1.0, label='sample A')
    ax.plot(t, wB, color='#E69F00', lw=1.0, label='sample B')
    lam = 0.5
    wm = lam * wA + (1 - lam) * wB
    ax.plot(t, wm, color='#D55E00', lw=1.0, label='mixed')
    ax.set_ylabel('Channel value')
    style(ax, 'Mixup family')
    ax.set_xlim(-3, 51)
    ax.legend(frameon=False, loc='upper right', fontsize=7.5, ncol=1)

    # ---- (2) GHI-targeted jitter ----
    ax = axes[1]
    t, w1 = base_wave(peak=0.5, seed=1)
    rng = np.random.default_rng(9)
    wj = np.clip(w1 + rng.normal(0, 0.09, len(w1)), 0, 1)
    ax.plot(t, w1, color=GREY, lw=1.0, alpha=0.8, label='original')
    ax.plot(t, wj, color=BLUE, lw=1.0, alpha=0.9, label='jittered')
    ax.fill_between(t, w1, wj, color=BLUE, alpha=0.12, lw=0)
    ax.set_ylabel('Channel value')
    style(ax, 'GHI-targeted jitter')
    ax.set_xlim(-3, 51)
    ax.legend(frameon=False, loc='upper right', fontsize=7.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
