"""Causal analysis: WHY does KANConv help? Cross-site spline-activity vs gain.

For every HKUST station (34): train KANConv_M5 (1 seed, same protocol), then
on the TEST split compute:
  1. per-channel mean |spline(x)-x| activation (how much the calibration
     actually moves each input channel's values)
  2. the spline gate value (init 1.0; >1 amplifies, <1 shrinks)
  3. test MSE
Merge with the FC-vs-KAN screen deltas and look for the correlation between
spline activity and KAN gain across sites.

Hypothesis being tested: KANConv pays off where the learned per-channel
calibration is actually ACTIVATED on the input distribution (values in the
nonlinear regime are present AND the spline learns to move them); on
near-linear or noise-dominated sites the spline stays near identity and the
gain vanishes.

Output: s_data/cleaned/kanconv_cause_analysis.csv
"""
import sys, os, shutil, glob, torch, numpy as np, pandas as pd, warnings
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from config import *
from utils import set_seed
from mode_code import CNN_LSTM_Head, prepare_multistep_data, Trainer
from scipy import stats

SITE6_HK = ['GHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
HK_DIR = 's_data/cleaned/hkust'
H = 24; SEQLEN = 24; EP = 100; BATCH = 4096; PATIENCE = 30
SEED = 42


def load_test(site_file, feats):
    df = pd.read_csv(site_file, index_col=0, parse_dates=True)
    X_all, Y_full = prepare_multistep_data(df, feature_cols=feats, target_col=TARGET_COL,
                                            seq_len=SEQLEN, max_horizon=H)
    Y = Y_full[:, :H]
    n = len(X_all); tr = int(n*0.64); vl = int(n*0.80)
    x_t = torch.FloatTensor(X_all[:tr]); y_t = torch.FloatTensor(Y[:tr])
    x_v = torch.FloatTensor(X_all[tr:vl]); y_v = torch.FloatTensor(Y[tr:vl])
    x_ts = torch.FloatTensor(X_all[vl:]); y_ts = torch.FloatTensor(Y[vl:])
    sx = MinMaxScaler((-1,1)).fit(x_t.reshape(-1, X_all.shape[2]))
    sy = MinMaxScaler((-1,1)).fit(y_t.reshape(-1, 1))
    def xs(t): return torch.FloatTensor(sx.transform(t.reshape(-1, t.shape[-1])).reshape(t.shape))
    def ys(t): return torch.FloatTensor(sy.transform(t.reshape(-1, 1)).reshape(t.shape))
    return (xs(x_t).to(DEVICE), ys(y_t).to(DEVICE),
            xs(x_v).to(DEVICE), ys(y_v).to(DEVICE),
            xs(x_ts).to(DEVICE), ys(y_ts).to(DEVICE))


def train_kan(site_file, feats, seed=SEED):
    set_seed(seed); shutil.rmtree('./model', ignore_errors=True)
    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts = load_test(site_file, feats)
    set_seed(seed)
    model = CNN_LSTM_Head(in_features=len(feats), horizon=H, head_type='FC',
                          device=DEVICE,
                          use_kanconv=True, kan_gate_init=1.0).to(DEVICE)
    trn = Trainer(model, lr=0.01, device=DEVICE)
    trn.train(x_tr, y_tr, x_vl, y_vl, epochs=EP, batch_size=BATCH,
              patience=PATIENCE)
    model.eval()
    with torch.no_grad():
        pred = model(x_ts).cpu().numpy()
    true = y_ts.cpu().numpy()
    mse = float(np.mean((pred - true) ** 2))
    # per-channel spline activity on the (normalized) test inputs
    spline = model.kanconv1.spline
    gate = float(spline.gate.detach().cpu()) if spline.gate is not None else None
    x_ts_cpu = x_ts.permute(0, 2, 1)              # (W, C, T)
    with torch.no_grad():
        out = spline(x_ts_cpu)
    act = (out - x_ts_cpu).abs().mean(dim=(0, 2)).cpu().numpy()   # (C,)
    return mse, gate, act


def main():
    sites = []
    for f in sorted(glob.glob(os.path.join(HK_DIR, 'hkust_*.csv'))):
        sites.append((os.path.basename(f)[len('hkust_'):-len('.csv')], f))

    # merge with the existing screen deltas (FC mean from the 27-site screen +
    # the 8site/SHHo tables for the previously-tested stations)
    screen = pd.read_csv('s_data/cleaned/kanconv_hkust_screen_summary.csv') \
        if os.path.exists('s_data/cleaned/kanconv_hkust_screen_summary.csv') else pd.DataFrame()
    extra = {}
    if os.path.exists('s_data/cleaned/kanconv_8site_design.csv'):
        d8 = pd.read_csv('s_data/cleaned/kanconv_8site_design.csv')
        for s in ['LSKN', 'SQ1', 'UG7']:
            sub = d8[d8.site == s]
            fc = sub[sub.config == 'FC'].norm_MSE.mean()
            kan = sub[sub.config == 'KANConv_M5'].norm_MSE.mean()
            v = sub[sub.config == 'KANConv_M5'].sort_values('seed').norm_MSE.values
            fcv = sub[sub.config == 'FC'].sort_values('seed').norm_MSE.values
            t, p = stats.ttest_rel(v, fcv)
            extra[s] = dict(FC=fc, KAN=kan, delta_pct=(kan/fc-1)*100, p=p)
    if os.path.exists('s_data/cleaned/kanconv_shho_fc_kan.csv'):
        d = pd.read_csv('s_data/cleaned/kanconv_shho_fc_kan.csv')
        fc = d[d.config == 'FC'].norm_MSE.mean()
        kan = d[d.config == 'KANConv_M5'].norm_MSE.mean()
        extra['SHHo'] = dict(FC=fc, KAN=kan, delta_pct=(kan/fc-1)*100, p=np.nan)

    rows = []
    for site_name, site_file in sites:
        try:
            mse, gate, act = train_kan(site_file, SITE6_HK)
        except Exception as e:
            print(f'  {site_name} ERR {e}', flush=True)
            continue
        delta = np.nan
        fc_mse = np.nan
        if site_name in extra:
            fc_mse = extra[site_name]['FC']
            delta = extra[site_name]['delta_pct']
        elif len(screen) and site_name in screen.site.values:
            r = screen[screen.site == site_name].iloc[0]
            fc_mse = r['FC']
            delta = r['delta_pct']
        rows.append({'site': site_name, 'kan_mse': mse, 'fc_mse': fc_mse,
                     'delta_pct': delta, 'gate': gate,
                     'act_GHI': act[0], 'act_Temp': act[1], 'act_Hum': act[2],
                     'act_WS': act[3], 'act_WDsin': act[4], 'act_WDcos': act[5],
                     'act_total': float(act.sum())})
        print(f'  {site_name:12s} gate={gate:.3f} act_total={act.sum():.4f} '
              f'delta={delta:+.2f}%', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('s_data/cleaned/kanconv_cause_analysis.csv', index=False)

    print('\n===== correlation: spline activity vs KAN gain =====')
    ok = df.dropna(subset=['delta_pct'])
    for col in ['act_total', 'act_GHI', 'act_Temp']:
        if len(ok) > 3:
            r, p = stats.spearmanr(ok[col], ok['delta_pct'])
            print(f'  {col:10s} spearman r={r:+.3f} p={p:.4f} (n={len(ok)})')
    if len(ok) > 3:
        r, p = stats.spearmanr(ok['gate'], ok['delta_pct'])
        print(f'  {"gate":10s} spearman r={r:+.3f} p={p:.4f} (n={len(ok)})')
    print('\n===== top active vs bottom active (delta comparison) =====')
    med = ok['act_total'].median()
    hi = ok[ok.act_total >= med]
    lo = ok[ok.act_total < med]
    print(f'  high-activity sites: mean delta={hi.delta_pct.mean():+.2f}% (n={len(hi)})')
    print(f'  low-activity sites:  mean delta={lo.delta_pct.mean():+.2f}% (n={len(lo)})')
    print('\n===== per-site table (sorted by delta) =====')
    print(df.sort_values('delta_pct').round(4).to_string(index=False))


if __name__ == '__main__':
    main()
