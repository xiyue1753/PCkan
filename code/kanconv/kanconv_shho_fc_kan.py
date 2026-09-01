"""FC vs KANConv_M5 comparison on HK_SHHo (new HKUST station, 10 seeds).

Replaces UG7 in the HKUST trio (UG7 showed ~0% KAN gain; SHHo has the
highest GHI->power nonlinearity, poly_gain=0.161 vs SQ1 0.002 / UG7 0.016).

Same protocol as kanconv_8site_design.py: 24in/24out, 100ep, patience 30,
batch 4096, MinMax(-1,1), canonical init, 10 seeds, normalized MSE.
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

SITE6_HK = ['GHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
SITES = {
    'HK_SHHo': ('s_data/cleaned/hkust/hkust_SHHo.csv', SITE6_HK),
}
CONFIGS = [
    ('FC',          dict()),
    ('KANConv_M5',  dict(use_kanconv=True, kan_gate_init=1.0)),
]
H = 24; SEQLEN = 24; EP = 100; BATCH = 4096; PATIENCE = 30
SEEDS10 = [42, 123, 456, 789, 2026, 11, 22, 33, 44, 55]
OUT_CSV = 's_data/cleaned/kanconv_shho_fc_kan.csv'


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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trn = Trainer(model, lr=0.01, device=DEVICE)
    _, _, best_ep, _, _ = trn.train(x_tr_s, y_tr_s, x_vl_s, y_vl_s,
                                    epochs=EP, batch_size=BATCH, patience=PATIENCE)
    model.eval()
    with torch.no_grad():
        pred_s = model(x_ts_s).cpu().numpy()
        true_s = y_ts_s.cpu().numpy()
    norm_mse = np.mean((pred_s - true_s) ** 2)
    norm_mae = np.mean(np.abs(pred_s - true_s))
    return norm_mse, norm_mae, best_ep, n_params


def main():
    set_seed(MASTER_SEED)
    results = []
    t0 = time.time()
    for site_name, (site_file, feats) in SITES.items():
        for cfg_name, m_kw in CONFIGS:
            for seed in SEEDS10:
                nm, nma, ep, np_ = train_one(site_file, feats, m_kw, seed)
                results.append({'site': site_name, 'config': cfg_name, 'seed': seed,
                                'norm_MSE': nm, 'norm_MAE': nma,
                                'best_epoch': ep, 'n_params': np_})
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)
                print(f'  {site_name:5s} {cfg_name:12s} s={seed:3d} nMSE={nm:.5f} '
                      f'ep={ep:2d}  {time.time()-t0:.0f}s', flush=True)
                torch.cuda.empty_cache()
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    for site_name in SITES:
        sub = df[df['site'] == site_name]
        fc = sub[sub['config'] == 'FC'].sort_values('seed')['norm_MSE'].values
        for cfg_name, _ in CONFIGS:
            v = sub[sub['config'] == cfg_name].sort_values('seed')['norm_MSE'].values
            if len(v) == 0: continue
            line = f'  {cfg_name:12s} nMSE={v.mean():.5f}+/-{v.std():.5f}'
            if cfg_name != 'FC':
                t, p = stats.ttest_rel(v, fc)
                wins = int((v < fc).sum())
                line += f'  delta={((v.mean()-fc.mean())/fc.mean()*100):+.1f}%  p={p:.4f}  wins {wins}/{len(v)}'
            print(line)


if __name__ == '__main__':
    main()
