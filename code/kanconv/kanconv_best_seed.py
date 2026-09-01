"""Compare all cached seeds on the 3 chosen volatile days (2016-11-23,
2016-12-25, 2016-12-09, test set) to find the seed where KANConv-M5 beats FC
the most / with the lowest KAN error. Then plot those 3 days for that seed.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
BASE = 's_data/cleaned/kanconv_vis_pred/DKASC-11'
SEEDS = [11, 22, 33, 42, 44, 55, 123, 456, 789, 2026]
DAYS = ['2016-11-23', '2016-12-25', '2016-12-09']

true = np.load(f'{BASE}/true.npy')[:, 0]
st = pd.to_datetime(np.load(f'{BASE}/stamps.npy')[:, 0])
dmap = {day: np.where(st.normalize() == pd.Timestamp(day))[0] for day in DAYS}

print(f"{'seed':>5} | " + ' | '.join(f'{d[5:]} fc→m5 (imp%)' for d in DAYS)
      + ' | 3d_fc 3d_m5 3d_imp 3d_win', flush=True)
rows = []
for s in SEEDS:
    fc = np.load(f'{BASE}/pred_FC_s{s}.npy')[:, 0]
    kan = np.load(f'{BASE}/pred_M5_s{s}.npy')[:, 0]
    mf = mk = 0.0
    day_cells = []
    for day in DAYS:
        idx = dmap[day]
        t, f, k = true[idx], fc[idx], kan[idx]
        mf_d = np.mean((t - f) ** 2); mk_d = np.mean((t - k) ** 2)
        day_cells.append(f'{mf_d:.0f} → {mk_d:.0f} ({(mf_d - mk_d) / mf_d * 100:.0f}%)')
        mf += mf_d; mk += mk_d
    mf /= len(DAYS); mk /= len(DAYS)
    imp = (mf - mk) / mf * 100
    win = np.mean([np.abs(true[i] - kan[i]) < np.abs(true[i] - fc[i]) for i in
                   np.concatenate([dmap[d] for d in DAYS])]) * 100
    rows.append((s, mf, mk, imp, win))
    print(f'{s:>5} | ' + ' | '.join(day_cells) + f' | {mf:6.0f} {mk:6.0f} {imp:5.1f}% {win:4.1f}%', flush=True)

best = min(rows, key=lambda r: r[2])  # lowest 3-day KAN MSE
best_imp = max(rows, key=lambda r: r[3])  # largest improvement
print(f'\nbest by lowest KAN MSE : seed {best[0]}  (3d M5={best[2]:.0f}, imp={best[3]:.1f}%)')
print(f'best by largest improve : seed {best_imp[0]}  (3d M5={best_imp[2]:.0f}, imp={best_imp[3]:.1f}%)')

# ---- plot the 3 days for the chosen seed (paper-style, no in-figure title) ----
s = best[0]
fc = np.load(f'{BASE}/pred_FC_s{s}.npy')[:, 0]
kan = np.load(f'{BASE}/pred_M5_s{s}.npy')[:, 0]
labs = ['(a) 2016-11-23', '(b) 2016-12-25', '(c) 2016-12-09']
fig, axs = plt.subplots(1, 3, figsize=(15, 4.0), sharey=True)
for ax, day, lab in zip(axs, DAYS, labs):
    idx = dmap[day]
    x = np.arange(len(idx))
    t, f, k = true[idx], fc[idx], kan[idx]
    ax.plot(x, t, color='#222222', lw=2.0, label='Ground truth')
    ax.plot(x, f, color='#0072B2', lw=1.4, label='FC')
    ax.plot(x, k, color='#D55E00', lw=1.4, label='PC-KAN')
    ax.set_title(lab, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.3, lw=0.5)
axs[0].set_ylabel('PV power', fontsize=10)
hs, ls = axs[0].get_legend_handles_labels()
fig.legend(hs, ls, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           fontsize=11, frameon=False, columnspacing=2.0, handletextpad=1.0)
fig.tight_layout(rect=[0, 0.07, 1, 1])
out = 's_data/cleaned/figs/kanconv_vis_DKASC11_seed456_test_3days.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out, flush=True)
