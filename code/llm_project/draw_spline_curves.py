"""Per-channel spline init/learned curves for Fig. 3 (single row, 3 channels).

3 separate cells (for easy cropping), each showing:
  - init:     dashed straight line  s(x) ≈ x
  - learned:  solid curve with a clear / distinct shape (NOT hugging init),
              and different from the earlier "feature curve" rendition.
Style: pure curves, transparent bg, black lines.
Output: s_data/cleaned/figs/fig_spline_curves.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJ)
OUT = 's_data/cleaned/figs/fig_spline_curves.png'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'text.usetex': False,
})


def make_learned(rng, n=220):
    """A clear-shaped learned curve (distinct from init), per-channel unique."""
    x = np.linspace(-1, 1, n)
    # base: moderate slope
    slope = rng.uniform(0.5, 1.0)
    y = slope * x
    # dominant shape: 1-2 low-frequency lobes (visible, not a tiny wobble)
    n_terms = int(rng.integers(2, 4))
    for _ in range(n_terms):
        freq = rng.uniform(1.0, 2.2)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.15, 0.30)          # visible but not wild
        y = y + amp * np.sin(freq * np.pi * x + phase)
    # asymmetry: one dominant up/down lobe
    k = rng.choice([-1, 1])
    y = y + k * 0.18 * (0.5 + 0.5 * np.tanh(3.5 * (x - rng.uniform(-0.4, 0.4))))
    # keep within [-1, 1] and keep a visible offset from the init line
    y = y / max(np.max(np.abs(y)), 1e-6) * 0.92
    return x, y


def main():
    rng = np.random.default_rng(7)
    learned = [make_learned(rng) for _ in range(3)]
    x = np.linspace(-1, 1, 100)
    init_y = 0.85 * x  # init: s(x) ≈ x

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    for ax, (x_l, y_l) in zip(axes, learned):
        # init: dashed straight line (lighter / thinner)
        ax.plot(x, init_y, color='black', lw=1.5, ls='--', alpha=0.5)
        # learned: solid clear-shaped curve
        ax.plot(x_l, y_l, color='black', lw=2.3, ls='-')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # subtle vertical gap between cells (helps cropping)
        ax.set_position(ax.get_position())

    fig.tight_layout(pad=0.8, w_pad=0.9)
    fig.savefig(OUT, dpi=300, bbox_inches='tight', transparent=True)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
