"""Main-table full config set for the PC-KAN paper (10-seed).

Reproduces the paper's ten-site main comparison:
  - HKUST-5: Zone_J1, UG3, SQ2, Zone_D (from this file when SQ1 is included) plus SQ1
  - DKASC-5: 11, 56, 67, 73, 79
This file only trains the sites NOT covered by kanconv_4sig_hkust.py, i.e.
HKUST-SQ1 and the five DKASC sites, so that the union of the two scripts covers
exactly the ten main-table sites used in the paper. Any site outside the paper
table (e.g. LSKN, UG7) is intentionally excluded.

Configs (identical to kanconv_4sig_hkust.py):
  FC, KANConv_base, KANConv_M5, Shared, Kernel, Frozen, LinearCalib, MLPConv

Protocol: 24in/24out, 100ep, patience 30, batch 4096, MinMax(-1,1),
canonical init, 10 seeds, normalized MSE. Resume-friendly.
"""
import sys, os, shutil, time, torch, numpy as np, pandas as pd, warnings
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

# HKUST uses Humidity; DKASC (SITE6_LD) uses Air_Pressure instead.
SITE6_HK = ['GHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
SITE6_LD = ['GHI', 'Temperature', 'Wind_Speed', 'WD_sin', 'WD_cos', 'Air_Pressure']

HK_DIR = 's_data/cleaned/hkust'
# Only the paper main-table sites that this file covers:
# SQ1 (HKUST) + the five DKASC sites.
SITES = {
    'SQ1': (f'{HK_DIR}/hkust_SQ1.csv', SITE6_HK),
    '11':  ('s_data/cleaned/site11_2016_15min.csv', SITE6_LD),
    '56':  ('s_data/cleaned/site56_2016_15min.csv', SITE6_LD),
    '67':  ('s_data/cleaned/site67_2016_15min.csv', SITE6_LD),
    '73':  ('s_data/cleaned/site73_2016_15min.csv', SITE6_LD),
    '79':  ('s_data/cleaned/site79_2016_15min.csv', SITE6_LD),
}
CONFIGS = [
    ('FC',            dict()),
    ('KANConv_base',  dict(use_kanconv=True)),
    ('KANConv_M5',    dict(use_kanconv=True, kan_gate_init=1.0)),
    ('Shared',        dict(use_shared_kanconv=True)),
    ('Kernel',        dict(use_kernel_kanconv=True)),
    ('Frozen',        dict(use_kanconv=True, freeze_spline=True)),
    ('LinearCalib',   dict(use_lin_calib=True)),
    ('MLPConv',       dict(use_mlp_calib=True)),
]
H = 24; SEQLEN = 24; EP = 100; BATCH = 4096; PATIENCE = 30
SEEDS10 = [42, 123, 456, 789, 2026, 11, 22, 33, 44, 55]
OUT_CSV = 's_data/cleaned/kanconv_8site_design.csv'


def train_one(site_file, feats, m_kw, seed):
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
    def xs(t,s): return torch.FloatTensor(s.transform(t.cpu().reshape(-1,t.shape[-1])).reshape(t.shape)).to(DEVICE)
    x_tr_s = xs(x_t,sx); y_tr_s = torch.FloatTensor(
        sy.transform(y_t.cpu().reshape(-1,1)).reshape(y_t.shape)).to(DEVICE)
    x_vl_s = xs(x_v,sx); y_vl_s = torch.FloatTensor(
        sy.transform(y_v.cpu().reshape(-1,1)).reshape(y_v.shape)).to(DEVICE)
    x_ts_s = xs(x_ts,sx)
    y_ts_s = torch.FloatTensor(sy.transform(y_ts.cpu().reshape(-1,1)).reshape(y_ts.shape)).to(DEVICE)
    set_seed(seed)
    model = CNN_LSTM_Head(in_features=len(feats), horizon=H, head_type='FC',
                          device=DEVICE, **m_kw).to(DEVICE)
    trn = Trainer(model, lr=0.01, device=DEVICE)
    _, _, best_ep, _, _ = trn.train(x_tr_s, y_tr_s, x_vl_s, y_vl_s,
                                    epochs=EP, batch_size=BATCH, patience=PATIENCE)
    model.eval()
    with torch.no_grad():
        pred_s = model(x_ts_s).cpu().numpy()
        true_s = y_ts_s.cpu().numpy()
    norm_mse = np.mean((pred_s - true_s) ** 2)
    norm_mae = np.mean(np.abs(pred_s - true_s))
    return norm_mse, norm_mae, best_ep


def main():
    set_seed(MASTER_SEED)
    results = []
    skip = set()
    if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 1:
        old = pd.read_csv(OUT_CSV)
        # keep only the paper main-table sites in any existing partial output
        keep_sites = set(SITES.keys())
        old = old[old['site'].isin(keep_sites)]
        results = old.to_dict('records')
        skip = set(zip(old['site'], old['config'], old['seed']))
    t0 = time.time()
    total = len(SITES) * len(CONFIGS) * len(SEEDS10)
    done = len(skip)
    for site_name, (site_file, feats) in SITES.items():
        for cfg_name, m_kw in CONFIGS:
            for seed in SEEDS10:
                if (site_name, cfg_name, seed) in skip: continue
                nm, nma, ep = train_one(site_file, feats, m_kw, seed)
                results.append({'site': site_name, 'config': cfg_name, 'seed': seed,
                                'norm_MSE': nm, 'norm_MAE': nma, 'best_epoch': ep})
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)
                done += 1
                print(f'  {site_name:8s} {cfg_name:12s} s={seed:3d} nMSE={nm:.5f} '
                      f'ep={ep:2d}  [{done}/{total}] {time.time()-t0:.0f}s', flush=True)
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"  MAIN-TABLE SITES (SQ1 + DKASC 5) - FULL CONFIG SET (10 seeds)")
    print(f"{'='*80}")
    for site_name in SITES:
        sub = df[df['site'] == site_name]
        fc = sub[sub['config'] == 'FC'].sort_values('seed')['norm_MSE'].values
        print(f"\n  == {site_name} (FC={fc.mean():.5f}) ==")
        for cfg_name, _ in CONFIGS:
            v = sub[sub['config'] == cfg_name].sort_values('seed')['norm_MSE'].values
            if len(v) == 0: continue
            line = f'    {cfg_name:12s} nMSE={v.mean():.5f}+/-{v.std():.5f}'
            if cfg_name != 'FC':
                t, p = stats.ttest_rel(v, fc)
                wins = int((v < fc).sum())
                line += f'  delta={((v.mean()-fc.mean())/fc.mean()*100):+.1f}%  p={p:.4f}  wins {wins}/{len(v)}'
            print(line)


if __name__ == '__main__':
    main()
