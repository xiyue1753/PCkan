"""Electricity generalization: fill remaining design configs (10 seeds).

Fills Shared / Kernel / LinearCalib for the electricity LTSF benchmark
(321ch -> OT, 96in/96out, Autoformer protocol). FC + KANConv_M5 already
exist in kanconv_ltsf_electricity.csv and are NOT rerun here.

Frozen is deliberately EXCLUDED (generalization experiment, not ablation).

Kernel feasibility gate: Kernel-side KAN on 321 channels has
out*in*(G+k)*kernel = 48*321*6*3 = 277k coef params (vs 46k plain conv) and
builds a (B*T, C*n_basis) basis tensor -> slow / high VRAM. If a single
Kernel seed takes > KERNEL_TIME_LIMIT min or OOMs, Kernel is skipped with a
warning (design-space comparison on the PV sites already covers it).

Usage:
  python kanconv/kanconv_ltsf_electricity_supp.py smoke  # 1 seed each, timing
  python kanconv/kanconv_ltsf_electricity_supp.py        # full (resume-safe)
"""
import sys, os, shutil, time, torch, numpy as np, pandas as pd, warnings, gc
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from config import *
from utils import set_seed
from mode_code import CNN_LSTM_Head
from scipy import stats

DATA = '../Dataset/all_six_datasets/electricity/electricity.csv'
SEQLEN = 96; PRED = 96; BATCH = 1024
MAX_EP = 50; PATIENCE = 10; LR = 0.001
SEEDS = [42, 123, 456, 789, 2026, 11, 22, 33, 44, 55]
OUT_CSV = 's_data/cleaned/kanconv_ltsf_electricity.csv'
SMOKE = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
KERNEL_TIME_LIMIT = 60  # minutes; skip Kernel if a seed exceeds this


def load_data():
    df = pd.read_csv(DATA, encoding='utf-8')
    cols = [c for c in df.columns if c != 'date']
    data = df[cols].values.astype(np.float32)
    n = len(data)
    num_train = int(n * 0.7); num_test = int(n * 0.2); num_val = n - num_train - num_test
    scaler = StandardScaler().fit(data[:num_train])
    data = scaler.transform(data).astype(np.float32)
    X_all, Y_all = [], []
    tgt = data[:, -1]
    for i in range(n - SEQLEN - PRED + 1):
        X_all.append(data[i:i + SEQLEN])
        Y_all.append(tgt[i + SEQLEN:i + SEQLEN + PRED])
    X_all = np.stack(X_all); Y_all = np.stack(Y_all)
    b_vl = num_train - SEQLEN
    b_ts = n - num_test - SEQLEN
    tr = (0, b_vl); vl = (b_vl, b_vl + num_val - SEQLEN); ts = (b_ts, b_ts + num_test - SEQLEN)
    def s(x, r): return torch.FloatTensor(x[r[0]:r[1]])
    out = (s(X_all, tr), s(Y_all, tr), s(X_all, vl), s(Y_all, vl),
           s(X_all, ts), s(Y_all, ts), scaler)
    del X_all, Y_all; gc.collect()
    return out


def train_batched(model, x_tr, y_tr, x_vl, y_vl):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    crit = torch.nn.MSELoss()
    n = len(x_tr)
    best_val, best_ep, no_improve = float('inf'), 0, 0
    best_state = None
    for e in range(MAX_EP):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = x_tr[idx].to(DEVICE); yb = y_tr[idx].to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        vsum = 0.0
        with torch.no_grad():
            for i in range(0, len(x_vl), BATCH):
                xb = x_vl[i:i + BATCH].to(DEVICE); yb = y_vl[i:i + BATCH].to(DEVICE)
                vsum += crit(model(xb), yb).item() * len(xb)
        vloss = vsum / len(x_vl)
        sched.step(vloss)
        if vloss < best_val - 1e-5:
            best_val, best_ep, no_improve = vloss, e, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, best_ep


def train_one(m_kw, seed, x_tr, y_tr, x_vl, y_vl, x_ts, y_ts, scaler):
    set_seed(seed); shutil.rmtree('./model', ignore_errors=True)
    model = CNN_LSTM_Head(in_features=321, horizon=PRED, head_type='FC',
                          device=DEVICE, **m_kw).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.time()
    best_val, best_ep = train_batched(model, x_tr, y_tr, x_vl, y_vl)
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred_s = np.concatenate([
            model(x_ts[i:i + BATCH].to(DEVICE)).cpu().numpy()
            for i in range(0, len(x_ts), BATCH)])
        true_s = y_ts.numpy()
    scale_ot = float(scaler.scale_[-1]); mean_ot = float(scaler.mean_[-1])
    pred = pred_s * scale_ot + mean_ot
    true = true_s * scale_ot + mean_ot
    mse = np.mean((true.ravel() - pred.ravel()) ** 2)
    mae = np.mean(np.abs(true.ravel() - pred.ravel()))
    return mse, mae, best_val, best_ep, dt, n_params


def main():
    set_seed(MASTER_SEED)
    t0 = time.time()
    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts, scaler = load_data()
    print(f'  data: train {x_tr.shape[0]} val {x_vl.shape[0]} test {x_ts.shape[0]}  '
          f'RAM {x_tr.numel()*4/1e9:.2f}GB  ({time.time()-t0:.0f}s)', flush=True)

    results = []
    skip = set()
    if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 1:
        old = pd.read_csv(OUT_CSV)
        results = old.to_dict('records')
        skip = set(zip(old['model'], old['seed']))
    seeds = [42] if SMOKE else SEEDS
    models = [
        ('Shared',      dict(use_shared_kanconv=True)),
        ('Kernel',      dict(use_kernel_kanconv=True)),
        ('LinearCalib', dict(use_lin_calib=True)),
    ]
    kernel_too_slow = False
    for m_label, m_kw in models:
        for seed in seeds:
            if (m_label, seed) in skip: continue
            if m_label == 'Kernel' and kernel_too_slow:
                print(f'  [skip] Kernel (previous seed > {KERNEL_TIME_LIMIT}min)')
                continue
            try:
                mse, mae, best_val, best_ep, dt, n_params = train_one(
                    m_kw, seed, x_tr, y_tr, x_vl, y_vl, x_ts, y_ts, scaler)
            except torch.cuda.OutOfMemoryError:
                print(f'  [OOM] {m_label} s={seed} -> skipping Kernel entirely')
                kernel_too_slow = True
                torch.cuda.empty_cache()
                continue
            results.append({'model': m_label, 'seed': seed, 'MSE': mse, 'MAE': mae,
                            'val_loss': best_val, 'best_epoch': best_ep, 'time_s': dt,
                            'n_params': n_params})
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)
            print(f'  {m_label:12s} s={seed:3d} MSE={mse:.4f} MAE={mae:.4f} ep={best_ep+1:2d} '
                  f'{dt/60:.1f}min params={n_params}  {time.time()-t0:.0f}s', flush=True)
            torch.cuda.empty_cache()
            if m_label == 'Kernel' and dt / 60 > KERNEL_TIME_LIMIT:
                kernel_too_slow = True
                print(f'  [warn] Kernel seed time {dt/60:.1f}min > {KERNEL_TIME_LIMIT}min '
                      f'-> stop Kernel')

    if SMOKE:
        print('\n  SMOKE DONE. If per-seed time is acceptable, run full: '
              'python kanconv/kanconv_ltsf_electricity_supp.py')
        return

    df = pd.DataFrame(results)
    fc = df[df['model'] == 'FC'].sort_values('seed')['MSE'].values
    if len(fc) == 0:
        fc = None
    print(f"\n{'='*72}")
    print("  ELECTRICITY SUPP (Shared/Kernel/LinearCalib vs existing FC+KANConv_M5)")
    print(f"{'='*72}")
    for m_label, _ in models + [('FC', dict()), ('KANConv_M5', dict())]:
        v = df[df['model'] == m_label].sort_values('seed')
        if len(v) == 0: continue
        k = v['MSE'].values
        if fc is not None and m_label != 'FC' and len(k) == len(fc):
            t, p = stats.ttest_rel(k, fc)
            wins = int((k < fc).sum())
            print(f'  {m_label:12s} MSE={k.mean():.4f}+/-{k.std():.4f}  '
                  f'delta={((k.mean()-fc.mean())/fc.mean()*100):+.1f}%  p={p:.3f}  wins {wins}/{len(k)}  '
                  f'mean_time={v["time_s"].mean()/60:.1f}min')
        else:
            print(f'  {m_label:12s} MSE={k.mean():.4f}+/-{k.std():.4f}  '
                  f'mean_time={v["time_s"].mean()/60:.1f}min')


if __name__ == '__main__':
    main()
