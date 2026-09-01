"""Removal ablation (NoLSTM) on the main-table 10 sites (HKUST 5 + DKASC 5).

Extends the earlier 8-site run (backup/ga_archive/kanconv_ablation.py mode C) to
the ten sites used in the main comparison (Section 4.2), so the ablation-site
scope matches the main ablation. Only mode C is implemented here.

Run (from cnn_lstm_kan/):  D:/.conda/envs/pytorch_env/python.exe kanconv_ablation_c10.py
Protocol: 24in/24out, 100ep, patience 30, batch 4096, MinMax(-1,1), 10 seeds.
Metrics: normalized MSE + MAE. Results appended to ablation_C8.csv (resume).
"""
import sys, os, shutil, torch, numpy as np, pandas as pd, warnings
sys.path.insert(0, '.')
sys.path.insert(0, 'kanconv')
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from config import *
from utils import set_seed
from mode_code import prepare_multistep_data
from kan_layers import KANConv1d
from scipy import stats

RAW7 = ['GHI', 'DHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
SITE6_HK = ['GHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
SITE6_LD = ['GHI', 'Temperature', 'Wind_Speed', 'WD_sin', 'WD_cos', 'Air_Pressure']

# Main-table ten sites (HKUST 5 + DKASC 5)
SITES10 = {
    'Zone_J1': ('s_data/cleaned/hkust/hkust_Zone_J1.csv', SITE6_HK),
    'UG3':     ('s_data/cleaned/hkust/hkust_UG3.csv', SITE6_HK),
    'SQ2':     ('s_data/cleaned/hkust/hkust_SQ2.csv', SITE6_HK),
    'Zone_D':  ('s_data/cleaned/hkust/hkust_Zone_D.csv', SITE6_HK),
    'SQ1':     ('s_data/cleaned/hkust/hkust_SQ1.csv', SITE6_HK),
    '11':      ('s_data/cleaned/site11_2016_15min.csv', SITE6_LD),
    '56':      ('s_data/cleaned/site56_2016_15min.csv', RAW7),
    '67':      ('s_data/cleaned/site67_2016_15min.csv', RAW7),
    '73':      ('s_data/cleaned/site73_2016_15min.csv', RAW7),
    '79':      ('s_data/cleaned/site79_2016_15min.csv', RAW7),
}
H = 24; SEQLEN = 24; EP = 100; BATCH = 4096; PATIENCE = 30
SEEDS10 = [42, 123, 456, 789, 2026, 11, 22, 33, 44, 55]
OUT_DIR = 's_data/cleaned/ablation'
OUT = f'{OUT_DIR}/ablation_C8.csv'
os.makedirs(OUT_DIR, exist_ok=True)


class CNNOnlyModel(torch.nn.Module):
    """CNN-only (NoLSTM) ablation: same conv front-end as CNN_LSTM_Head,
    LSTM replaced by mean-pool over time + linear head."""
    def __init__(self, in_features=7, out_channels=48, horizon=24,
                 use_kan=False, gate_init=None):
        super().__init__()
        if use_kan:
            self.kanconv1 = KANConv1d(in_features, out_channels, 3, padding=1,
                                      grid_size=3, k=3, gate_init=gate_init)
            self.bn1 = torch.nn.BatchNorm1d(out_channels)
            self.conv2 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv1d(out_channels, out_channels, 3, padding=1),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2))
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_features, out_channels, 3, padding=1),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv1d(out_channels, out_channels, 3, padding=1),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2))
        self.head = torch.nn.Linear(out_channels, horizon)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if hasattr(self, 'kanconv1'):
            c = self.bn1(self.kanconv1(x))
            c = self.conv2(c)
        else:
            c = self.conv(x)
        c = c.mean(dim=2)
        return self.head(c)


def load_scaled(site_file, feats):
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
    def xs(t,s): return torch.FloatTensor(s.transform(t.reshape(-1,t.shape[-1])).reshape(t.shape))
    x_tr_s = xs(x_t,sx); y_tr_s = torch.FloatTensor(sy.transform(y_t.reshape(-1,1)).reshape(y_t.shape))
    x_vl_s = xs(x_v,sx); y_vl_s = torch.FloatTensor(sy.transform(y_v.reshape(-1,1)).reshape(y_v.shape))
    x_ts_s = xs(x_ts,sx)
    y_ts_s = torch.FloatTensor(sy.transform(y_ts.reshape(-1,1)).reshape(y_ts.shape))
    return x_tr_s, y_tr_s, x_vl_s, y_vl_s, x_ts_s, y_ts_s, y_ts, sy


def train_one(model, x_tr, y_tr, x_vl, y_vl):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    crit = torch.nn.MSELoss()
    n = len(x_tr)
    best_val, best_ep, no_improve, best_state = float('inf'), 0, 0, None
    for e in range(EP):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            crit(model(x_tr[idx].to(DEVICE)), y_tr[idx].to(DEVICE)).backward()
            opt.step()
        model.eval()
        vsum = 0.0
        with torch.no_grad():
            for i in range(0, len(x_vl), BATCH):
                vsum += crit(model(x_vl[i:i+BATCH].to(DEVICE)),
                             y_vl[i:i+BATCH].to(DEVICE)).item() * min(BATCH, len(x_vl)-i)
        vloss = vsum / len(x_vl)
        sched.step(vloss)
        if vloss < best_val - 1e-5:
            best_val, best_ep, no_improve = vloss, e, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    if best_state: model.load_state_dict(best_state)
    return best_val, best_ep


def evaluate(model, x_ts_s, y_ts_s):
    model.eval()
    with torch.no_grad():
        pred_s = model(x_ts_s.to(DEVICE)).cpu().numpy()
    true_s = y_ts_s.numpy()
    norm_mse = np.mean((pred_s - true_s) ** 2)
    norm_mae = np.mean(np.abs(pred_s - true_s))
    return norm_mse, norm_mae


def load_resume():
    if os.path.exists(OUT) and os.path.getsize(OUT) > 1:
        old = pd.read_csv(OUT)
        old['site'] = old['site'].astype(str)
        results = old.to_dict('records')
        keys = [c for c in ['site', 'config', 'seed'] if c in old.columns]
        skip = set(zip(*[old[c].values for c in keys]))
        print(f'  resume: {len(skip)} done')
        return results, skip
    return [], set()


def run_C():
    results, skip = load_resume()
    for site_name, (site_file, feats) in SITES10.items():
        x_tr, y_tr, x_vl, y_vl, x_ts_s, y_ts_s, y_ts, sy = load_scaled(site_file, feats)
        for cfg_name, use_kan in [('FC_NoLSTM', False), ('M5_NoLSTM', True)]:
            for seed in SEEDS10:
                if (site_name, cfg_name, seed) in skip: continue
                set_seed(seed); shutil.rmtree('./model', ignore_errors=True)
                model = CNNOnlyModel(in_features=len(feats), horizon=H, use_kan=use_kan,
                                     gate_init=(1.0 if use_kan else None)).to(DEVICE)
                _, _ = train_one(model, x_tr, y_tr, x_vl, y_vl)
                nm, nma = evaluate(model, x_ts_s, y_ts_s)
                results.append({'site': site_name, 'config': cfg_name, 'seed': seed,
                                'norm_MSE': nm, 'norm_MAE': nma})
                pd.DataFrame(results).to_csv(OUT, index=False)
                print(f'  C {site_name:8s} {cfg_name:10s} s={seed:3d} nMSE={nm:.5f}', flush=True)
                torch.cuda.empty_cache()
    report_C10(pd.DataFrame(results))


def report_C10(df):
    # Merge full (LSTM) FC/KANConv_M5 baselines from the two main-table sources:
    # HKUST Zone_J1/UG3/SQ2/Zone_D -> kanconv_4sig_hkust.csv; SQ1 + DKASC -> kanconv_8site_design.csv
    def baseline():
        out = {}
        a = pd.read_csv('s_data/cleaned/kanconv_4sig_hkust.csv'); a['site'] = a['site'].astype(str)
        b = pd.read_csv('s_data/cleaned/kanconv_8site_design.csv'); b['site'] = b['site'].astype(str)
        comb = pd.concat([a, b], ignore_index=True)
        for site in SITES10:
            for cfg in ['FC', 'KANConv_M5']:
                v = comb[(comb['site'] == site) & (comb['config'] == cfg)]\
                    .sort_values('seed')['norm_MSE'].values
                out[(site, cfg)] = v
        return out
    bl = baseline()
    print(f"\n{'='*80}\n  C10: REMOVAL — LSTM component (10 sites x 10 seeds, nMSE)\n{'='*80}")
    base10, nl10, m5_10, m5nl_10 = [], [], [], []
    n_loss_fc = n_sig_fc = n_loss_m5 = n_sig_m5 = 0
    for site_name, (site_file, feats) in SITES10.items():
        sub = df[df['site'] == site_name]
        fc_nl = sub[sub['config'] == 'FC_NoLSTM'].sort_values('seed')['norm_MSE'].values
        m5_nl = sub[sub['config'] == 'M5_NoLSTM'].sort_values('seed')['norm_MSE'].values
        fc = bl[(site_name, 'FC')]
        m5 = bl[(site_name, 'KANConv_M5')]
        if len(fc_nl) != 10 or len(fc) != 10: continue
        d_fc = (fc_nl.mean() - fc.mean()) / fc.mean() * 100
        n_loss_fc += int(d_fc > 0)
        _, p = stats.ttest_rel(fc_nl, fc); n_sig_fc += int(p < 0.05 and d_fc > 0)
        line = f'  {site_name:8s} FC {fc.mean():.5f} -> NoLSTM {fc_nl.mean():.5f} ({d_fc:+.1f}%)'
        if len(m5_nl) == 10 and len(m5) == 10:
            d_m5 = (m5_nl.mean() - m5.mean()) / m5.mean() * 100
            n_loss_m5 += int(d_m5 > 0)
            _, p2 = stats.ttest_rel(m5_nl, m5); n_sig_m5 += int(p2 < 0.05 and d_m5 > 0)
            line += f' | M5 {m5.mean():.5f} -> {m5_nl.mean():.5f} ({d_m5:+.1f}%)'
        print(line)
        base10.append(fc.mean()); nl10.append(fc_nl.mean())
        m5_10.append(m5.mean()); m5nl_10.append(m5_nl.mean())
    print(f"\n  10-site mean: FC {np.mean(base10):.5f} -> NoLSTM {np.mean(nl10):.5f} "
          f"({(np.mean(nl10)-np.mean(base10))/np.mean(base10)*100:+.1f}%) | "
          f"M5 {np.mean(m5_10):.5f} -> {np.mean(m5nl_10):.5f} "
          f"({(np.mean(m5nl_10)-np.mean(m5_10))/np.mean(m5_10)*100:+.1f}%)")
    print(f"  LSTM contribution: FC loss {n_loss_fc}/10 sites, {n_sig_fc} significant; "
          f"M5 loss {n_loss_m5}/10 sites, {n_sig_m5} significant")


if __name__ == '__main__':
    run_C()
