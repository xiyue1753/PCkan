"""Clean attribution: separate structure / bias / learned-shape contributions.

The Frozen control in the 8-site table is NOT clean: KANConv1d uses
conv(bias=False) while the FC branch uses conv(bias=True) - so the
Frozen-vs-FC gap mixes a bias difference with the spline structure.

Configs (4 significant sites x 3 seeds):
  FC          - plain conv (bias=True), no spline        [reference]
  FC_noBias   - same conv structure, bias=False          [isolates bias]
  Frozen      - KANConv1d, spline frozen at identity     [structure + bias]
  M5          - KANConv1d, spline learned + gate         [full method]

Decomposition:
  bias effect      = FC - FC_noBias
  structure effect = FC_noBias - Frozen
  shape effect     = Frozen - M5
"""
import sys, os, shutil, torch, numpy as np, pandas as pd, warnings
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from config import *
from utils import set_seed
from mode_code import CNN_LSTM_Head, prepare_multistep_data, Trainer
from scipy import stats

SITE6_HK = ['GHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
HK_DIR = 's_data/cleaned/hkust'
SITES = {k: f'{HK_DIR}/hkust_{k}.csv' for k in ['Zone_J1', 'UG3', 'SQ2', 'Zone_D']}
H = 24; SEQLEN = 24; EP = 100; BATCH = 4096; PATIENCE = 30
SEEDS3 = [42, 123, 456]
OUT_CSV = 's_data/cleaned/kanconv_bias_probe.csv'


class FCBlock(nn.Module):
    """Plain conv block, mirroring the FC branch of CNN_LSTM_Head, with a
    bias switch (the only structural difference vs KANConv1d)."""
    def __init__(self, in_channels, out_channels, kernel=3, pad=1, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=pad, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel, padding=pad, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2))

    def forward(self, x):
        return self.conv(x)


def build_model(feats, variant):
    model = CNN_LSTM_Head(in_features=len(feats), horizon=H, head_type='FC',
                          device=DEVICE)
    if variant == 'FC':
        pass                                  # default FC branch (bias=True)
    elif variant == 'FC_noBias':
        model.conv = FCBlock(len(feats), 48, bias=False)
    elif variant == 'Frozen':
        model = CNN_LSTM_Head(in_features=len(feats), horizon=H, head_type='FC',
                              device=DEVICE, use_kanconv=True,
                              freeze_spline=True)
    elif variant == 'M5':
        model = CNN_LSTM_Head(in_features=len(feats), horizon=H, head_type='FC',
                              device=DEVICE, use_kanconv=True,
                              kan_gate_init=1.0)
    return model.to(DEVICE)


def train(site_file, feats, variant, seed):
    set_seed(seed); shutil.rmtree('./model', ignore_errors=True)
    df = pd.read_csv(site_file, index_col=0, parse_dates=True)
    X_all, Y_full = prepare_multistep_data(df, feature_cols=feats, target_col=TARGET_COL,
                                            seq_len=SEQLEN, max_horizon=H)
    Y = Y_full[:, :H]
    n = len(X_all); tr = int(n*0.64); vl = int(n*0.80)
    x_t = torch.FloatTensor(X_all[:tr]).to(DEVICE); y_t = torch.FloatTensor(Y[:tr]).to(DEVICE)
    x_v = torch.FloatTensor(X_all[tr:vl]).to(DEVICE); y_v = torch.FloatTensor(Y[tr:vl]).to(DEVICE)
    x_ts = torch.FloatTensor(X_all[vl:]).to(DEVICE); y_ts = torch.FloatTensor(Y[vl:]).to(DEVICE)
    sx = MinMaxScaler((-1,1)).fit(x_t.cpu().reshape(-1, X_all.shape[2]))
    sy = MinMaxScaler((-1,1)).fit(y_t.cpu().reshape(-1, 1))
    def xs(t): return torch.FloatTensor(sx.transform(t.cpu().reshape(-1, t.shape[-1])).reshape(t.shape)).to(DEVICE)
    def ys(t): return torch.FloatTensor(sy.transform(t.cpu().reshape(-1, 1)).reshape(t.shape)).to(DEVICE)
    x_tr_s, y_tr_s = xs(x_t), ys(y_t)
    x_vl_s, y_vl_s = xs(x_v), ys(y_v)
    x_ts_s, y_ts_s = xs(x_ts), ys(y_ts)
    set_seed(seed)
    model = build_model(feats, variant)
    trn = Trainer(model, lr=0.01, device=DEVICE)
    trn.train(x_tr_s, y_tr_s, x_vl_s, y_vl_s, epochs=EP, batch_size=BATCH,
              patience=PATIENCE)
    model.eval()
    with torch.no_grad():
        pred = model(x_ts_s).cpu().numpy()
    true = y_ts_s.cpu().numpy()
    return float(np.mean((pred - true) ** 2))


def main():
    results = []
    skip = set()
    if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 1:
        old = pd.read_csv(OUT_CSV)
        results = old.to_dict('records')
        skip = set(zip(old['site'], old['config'], old['seed']))
    total = len(SITES) * 4 * len(SEEDS3)
    done = len(skip)
    for site_name, site_file in SITES.items():
        for variant in ['FC', 'FC_noBias', 'Frozen', 'M5']:
            for seed in SEEDS3:
                if (site_name, variant, seed) in skip:
                    continue
                mse = train(site_file, SITE6_HK, variant, seed)
                results.append({'site': site_name, 'config': variant,
                                'seed': seed, 'norm_MSE': mse})
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)
                done += 1
                print(f'  {site_name:8s} {variant:10s} s={seed:3d} '
                      f'nMSE={mse:.5f} [{done}/{total}]', flush=True)
    df = pd.DataFrame(results)
    print(f"\n{'='*88}")
    print(f"  CLEAN ATTRIBUTION (4 sites x 3 seeds)")
    print(f"{'='*88}")
    for site in SITES:
        sub = df[df.site == site]
        fc = sub[sub.config == 'FC'].sort_values('seed').norm_MSE.values
        fnb = sub[sub.config == 'FC_noBias'].sort_values('seed').norm_MSE.values
        fr = sub[sub.config == 'Frozen'].sort_values('seed').norm_MSE.values
        m5 = sub[sub.config == 'M5'].sort_values('seed').norm_MSE.values
        bias = (fnb.mean() - fc.mean()) / fc.mean() * 100
        struct = (fr.mean() - fnb.mean()) / fnb.mean() * 100
        shape = (m5.mean() - fr.mean()) / fr.mean() * 100
        print(f'-- {site}: FC={fc.mean():.5f}')
        print(f'   FC_noBias={fnb.mean():.5f} ({bias:+.1f}%) | '
              f'Frozen={fr.mean():.5f} ({struct:+.1f}%) | '
              f'M5={m5.mean():.5f} ({shape:+.1f}%)')
        for a, b, name in [(fc, fnb, 'bias'), (fnb, fr, 'structure'),
                           (fr, m5, 'shape')]:
            t, p = stats.ttest_rel(a, b)
            print(f'      {name:10s} delta={((b.mean()-a.mean())/a.mean()*100):+.1f}% '
                  f'p={p:.3f} {"*" if p<0.05 else ""}')


if __name__ == '__main__':
    main()
