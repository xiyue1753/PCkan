"""Fig. 10: Temperature sensitivity of LLM-driven joint search (site 1).

For each temperature (0.1, 0.3, 0.5, 0.9), a boxplot summarises the 3 independent
runs of best validation MSE with the jittered raw values overlaid, showing that
the search stays stable and reaches a value at or below the 0.1 baseline at
every temperature.

Output: s_data/cleaned/figs/fig_llm_temperature.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_LLM = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_LLM)
os.chdir(_PROJ)
OUT = os.path.join('s_data', 'cleaned', 'figs', 'fig_llm_temperature.png')

# temperature -> 3-run best validation MSE (site 1 = site 212)
DATA = {
    '0.1': [0.04898, 0.04741, 0.04966],
    '0.3': [0.04739, 0.04633, 0.04721],
    '0.5': [0.04978, 0.04249, 0.04478],
    '0.9': [0.04916, 0.04432, 0.04676],
}
COLOR = '#0072B2'   # same blue as LLM in Fig. 8/9

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.9,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})


def main():
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    temps = list(DATA.keys())
    xpos = np.arange(len(temps))
    rng = np.random.default_rng(0)

    bp = ax.boxplot([DATA[t] for t in temps], positions=xpos, widths=0.5,
                    patch_artist=True,
                    boxprops=dict(linewidth=0.8, edgecolor='black'),
                    medianprops=dict(color='black', linewidth=0.8),
                    whiskerprops=dict(color='black', linewidth=0.8),
                    capprops=dict(color='black', linewidth=0.8),
                    showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(COLOR)
        patch.set_alpha(0.35)

    # jittered raw values (3 runs) per temperature
    for xi, t in enumerate(temps):
        jitter = rng.uniform(-0.10, 0.10, len(DATA[t]))
        ax.scatter(xi + jitter, DATA[t], s=35, color=COLOR,
                   edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_xticks(xpos)
    ax.set_xticklabels([f'{t}' for t in temps], fontsize=9)
    ax.set_xlabel('Sampling temperature')
    ax.set_ylabel('Best validation MSE (3 runs)')
    ax.set_ylim(0.040, 0.052)
    ax.set_title('Temperature sensitivity (DKASC-212)', fontsize=10, pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3.5, labelsize=8)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
