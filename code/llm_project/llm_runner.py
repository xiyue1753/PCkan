"""Unified HPO trial evaluation + search orchestration.

Protocol (identical to the main-table harness kanconv_8site_design.py):
  24in/24out, MinMax(-1,1) normalization, canonical protocol-B constructor
  init, Trainer with lr / 100ep / patience 30 / batch 4096, ReduceLROnPlateau.

Objective per trial: BEST VALIDATION MSE (normalized scale), median over
TRIAL_SEEDS = [42, 123, 456, 789, 2026]   # 5-seed eval (plan: 3-5) (fixed per site so every search method sees the
exact same evaluation -> fair and cacheable).

Trial registry: s_data/cleaned/llm_trial_registry.json keyed by
(site, config_hash) -> per-seed results. Shared across ALL methods and groups
(a config is evaluated once; different search methods re-read the cached
result, eliminating evaluation noise between methods).
"""
import os, sys
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_LLM_DIR)
sys.path.insert(0, _LLM_DIR)
sys.path.insert(0, _PROJ_DIR)
os.chdir(_PROJ_DIR)          # cnn_lstm_kan/ keeps s_data/ paths working

import hashlib
import json
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

warnings.filterwarnings('ignore')
from config import *
from utils import set_seed
from mode_code import CNN_LSTM_Head, prepare_multistep_data, Trainer
from llm_search_space import (validate, config_hash, DEFAULT_CONFIG,
                              FIXED_PROTOCOL, aug_enabled)
from llm_augment import augment_dataset

RAW7 = ['GHI', 'DHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
SITE6_HK = ['GHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
SITE6_LD = ['GHI', 'Temperature', 'Wind_Speed', 'WD_sin', 'WD_cos', 'Air_Pressure']
SITES = {
    '56':   ('s_data/cleaned/site56_2016_15min.csv', RAW7),
    '73':   ('s_data/cleaned/site73_2016_15min.csv', RAW7),
    '11':   ('s_data/cleaned/site11_2016_15min.csv', SITE6_LD),
    '67':   ('s_data/cleaned/site67_2016_15min.csv', RAW7),
    '79':   ('s_data/cleaned/site79_2016_15min.csv', RAW7),
    # natural half-year sites (2016-07-02 -> 12-31, ~11.5K rows): the
    # small-data regime for the augmentation study
    '212':  ('s_data/cleaned/site212_2016_15min.csv', RAW7),
    '213':  ('s_data/cleaned/site213_2016_15min.csv', RAW7),
    'LSKN': ('s_data/cleaned/hkust/hkust_LSKN.csv', SITE6_HK),
    'SQ1':  ('s_data/cleaned/hkust/hkust_SQ1.csv', SITE6_HK),
    'UG7':  ('s_data/cleaned/hkust/hkust_UG7.csv', SITE6_HK),
}

TRIAL_SEEDS = [42, 123, 456, 789, 2026]   # 5-seed eval (plan: 3-5)
EP = FIXED_PROTOCOL['epochs']
BATCH = FIXED_PROTOCOL['batch_size']
PATIENCE = FIXED_PROTOCOL['patience']
SEQLEN = FIXED_PROTOCOL['seqlen']
H = FIXED_PROTOCOL['horizon']

# Fixed-budget protocol (early stopping disabled): every model trains exactly
# FIXED_EPOCHS epochs, where FIXED_EPOCHS = mean early-stop epoch of the main
# table KANConv-M5 at the study sites (56/73/11: 29.8/24.2/32.4, mean 28.8
# -> 30). Augmentation benefits show under full training; the early-stop
# schedule masks them.
FIXED_EPOCHS = 30

# Preprocessing cache: read_csv + prepare_multistep_data are pure functions of
# (site_file, feats, subset, subset_mode) and are recomputed once per
# (seed, trial) otherwise. Results are numpy, so no device issues.
_DATA_CACHE = {}


def _load_windows(site_file, feats, subset, subset_mode):
    key = (site_file, tuple(feats), subset, subset_mode)
    if key not in _DATA_CACHE:
        df = pd.read_csv(site_file, index_col=0, parse_dates=True)
        X_all, Y_full = prepare_multistep_data(
            df, feature_cols=feats, target_col=TARGET_COL,
            seq_len=SEQLEN, max_horizon=H)
        Y = Y_full[:, :H]
        n = len(X_all)
        if subset < 1.0:
            n_sub = int(n * subset)
            if subset_mode == 'uniform':
                idx = np.linspace(0, n - 1, n_sub).astype(int)
                X_all = X_all[idx]
                Y = Y[idx]
            else:                                   # 'prefix'
                X_all = X_all[:n_sub]
                Y = Y[:n_sub]
        _DATA_CACHE[key] = (X_all, Y)
    return _DATA_CACHE[key]


def registry_path(site_key, fixed_epochs=None, subset=1.0,
                  subset_mode='prefix'):
    """Registry file per (site, protocol, subset): concurrent multi-site runs
    never clobber each other, different fixed-budget values never share a file
    (epochs=60 would otherwise silently hit the fixed-30 cache), and the
    data-scarcity regime (subset<1) is keyed separately from full data.
    Directory can be overridden via env LLM_REGISTRY_DIR."""
    tag = f'fixed{fixed_epochs}' if fixed_epochs else 'earlystop'
    if subset < 1.0:
        tag += f'_sub{subset}'
        if subset_mode == 'uniform':
            tag += '_uni'
    base = os.environ.get('LLM_REGISTRY_DIR', 's_data/cleaned')
    return os.path.join(base, f'llm_trial_registry_{site_key}_{tag}.json')


def load_registry(path):
    """Load a registry file. Returns (data, meta). Missing/invalid files give
    ({}, None); a meta header that does not match the requested protocol or
    augmentation version is rejected (stale caches reset automatically)."""
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except Exception:
        return {}, None
    meta = raw.get('meta')
    if not isinstance(meta, dict):
        return {}, meta            # headerless legacy file: reject
    return raw.get('data', {}), meta


def save_registry(reg, path, meta):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'data': reg}, f, ensure_ascii=False, indent=1)


def _registry_meta(site_key, fixed_epochs, subset=1.0, subset_mode='prefix'):
    from llm_augment import AUG_VERSION
    return {'protocol_tag': f'fixed{fixed_epochs}' if fixed_epochs else 'earlystop',
            'objective': 'final_val' if fixed_epochs else 'best_val',
            'subset': subset, 'subset_mode': subset_mode,
            'aug_version': AUG_VERSION, 'site': site_key}


def train_one_val(site_file, feats, m_kw, seed, lr=0.01, aug_config=None,
                  epochs=EP, fixed_epochs=None, subset=1.0,
                  subset_mode='prefix', batch=BATCH, aug_mode='append'):
    """Train one model and return (best_val, test_norm_mse, best_ep).

    Mirrors kanconv_8site_design.py::train_one but records the validation
    objective (which the CSV does not store) and optionally augments the
    train split.

    fixed_epochs > 0: disable early stopping, train exactly that many epochs
    (Trainer returns the LAST state; best_val = final-epoch val, matching the
    evaluated model).

    subset < 1.0: keep a fraction of the series. 'prefix' = chronological
    prefix (cold-start scenario); 'uniform' = evenly spaced windows (keeps the
    seasonal distribution — required for clean data-scale comparisons).
    """
    set_seed(seed)
    shutil.rmtree('./model', ignore_errors=True)
    X_all, Y = _load_windows(site_file, feats, subset, subset_mode)
    n = len(X_all)
    tr = int(n * 0.64)
    vl = int(n * 0.80)
    x_t = torch.FloatTensor(X_all[:tr]).to(DEVICE)
    y_t = torch.FloatTensor(Y[:tr]).to(DEVICE)
    x_v = torch.FloatTensor(X_all[tr:vl]).to(DEVICE)
    y_v = torch.FloatTensor(Y[tr:vl]).to(DEVICE)
    x_ts = torch.FloatTensor(X_all[vl:]).to(DEVICE)
    y_ts = torch.FloatTensor(Y[vl:]).to(DEVICE)

    sx = MinMaxScaler((-1, 1)).fit(x_t.cpu().reshape(-1, X_all.shape[2]))
    sy = MinMaxScaler((-1, 1)).fit(y_t.cpu().reshape(-1, 1))

    def xs(t, s):
        return torch.FloatTensor(s.transform(t.cpu().reshape(-1, t.shape[-1]))
                                 .reshape(t.shape)).to(DEVICE)

    x_tr_s = xs(x_t, sx)
    y_tr_s = torch.FloatTensor(sy.transform(y_t.cpu().reshape(-1, 1)).reshape(y_t.shape)).to(DEVICE)
    x_vl_s = xs(x_v, sx)
    y_vl_s = torch.FloatTensor(sy.transform(y_v.cpu().reshape(-1, 1)).reshape(y_v.shape)).to(DEVICE)
    x_ts_s = xs(x_ts, sx)
    y_ts_s = torch.FloatTensor(sy.transform(y_ts.cpu().reshape(-1, 1)).reshape(y_ts.shape)).to(DEVICE)

    # augmentation applies to the (normalized) train input windows only;
    # labels are replicated to stay aligned with the enlarged input set
    x_tr_s, y_tr_s = augment_dataset(x_tr_s, y_tr_s, aug_config, seed=seed,
                                  aug_mode=aug_mode)

    set_seed(seed)
    model = CNN_LSTM_Head(in_features=len(feats), horizon=H, head_type='FC',
                          device=DEVICE, **m_kw).to(DEVICE)
    trn = Trainer(model, lr=lr, device=DEVICE)
    if fixed_epochs:
        # fixed budget: train exactly fixed_epochs, the evaluated model is the
        # FINAL state, so the objective must be the FINAL-epoch val (not the
        # best val over the schedule — best-val vs final-state mismatch made
        # val rankings meaningless for the evaluated model).
        _, best_val, best_ep, _, val_losses = trn.train(
            x_tr_s, y_tr_s, x_vl_s, y_vl_s, epochs=fixed_epochs,
            batch_size=batch, patience=PATIENCE, fixed_epochs=fixed_epochs)
        best_val = float(val_losses[-1])          # final-epoch val = objective
    else:
        try:
            _, best_val, best_ep, _, _ = trn.train(x_tr_s, y_tr_s, x_vl_s, y_vl_s,
                                                   epochs=epochs, batch_size=batch,
                                                   patience=PATIENCE)
        except torch.cuda.OutOfMemoryError:
            # extreme configs (very wide/hidden) may not fit 4GB at batch 4096;
            # grad_accum=4 splits each batch into 4 micro-batches and accumulates
            # gradients — mathematically equivalent to the full batch.
            torch.cuda.empty_cache()
            set_seed(seed)
            model = CNN_LSTM_Head(in_features=len(feats), horizon=H, head_type='FC',
                                  device=DEVICE, **m_kw).to(DEVICE)
            trn = Trainer(model, lr=lr, device=DEVICE)
            _, best_val, best_ep, _, _ = trn.train(x_tr_s, y_tr_s, x_vl_s, y_vl_s,
                                                   epochs=epochs, batch_size=batch,
                                                   patience=PATIENCE, grad_accum=4)
    model.eval()
    with torch.no_grad():
        pred_s = model(x_ts_s).cpu().numpy()
        true_s = y_ts_s.cpu().numpy()
    norm_mse = float(np.mean((pred_s - true_s) ** 2))
    return best_val, norm_mse, best_ep


def evaluate_config(site_key, config, use_registry=True, fixed_epochs=None,
                    subset=1.0, subset_mode='prefix', batch=None):
    """Evaluate one config on one site: median val MSE over TRIAL_SEEDS.

    Reads/writes the per-(site, protocol, subset) trial registry so identical
    configs are never re-evaluated (fairness across methods + compute saving).
    Cached entries are rejected unless their meta header matches the current
    protocol tag, subset and augmentation version.
    """
    config = validate(config)
    if config is None:
        raise ValueError('invalid config')
    if batch is None:
        batch = max(256, int(4096 * subset)) if subset < 1.0 else 4096
    h = config_hash(config)
    path = registry_path(site_key, fixed_epochs, subset, subset_mode)
    meta = _registry_meta(site_key, fixed_epochs, subset, subset_mode)
    reg, reg_meta = load_registry(path)
    if reg_meta != meta:
        reg = {}                                  # stale protocol/version: reset
    key = f'{site_key}|{h}'
    if use_registry and key in reg and len(reg[key].get('per_seed', {})) >= len(TRIAL_SEEDS):
        return reg[key]

    site_file, feats = SITES[site_key]
    m_kw = dict(use_kanconv=True, kan_gate_init=1.0,
                hidden_size=config['hidden_dim'], num_layers=config['num_layers'],
                out_channels=config['out_channels'], dropout=config['dropout'])
    per_seed = {}
    for seed in TRIAL_SEEDS:
        val_mse, test_mse, best_ep = train_one_val(
            site_file, feats, m_kw, seed, lr=config['learning_rate'],
            aug_config=config if aug_enabled(config) else None,
            fixed_epochs=fixed_epochs, subset=subset,
            subset_mode=subset_mode, batch=batch)
        per_seed[str(seed)] = {'val_mse': val_mse, 'test_mse': test_mse,
                               'best_epoch': best_ep}
    vals = [per_seed[str(s)]['val_mse'] for s in TRIAL_SEEDS]
    entry = {
        'config': config,
        'per_seed': per_seed,
        'median_val_mse': float(np.median(vals)),
        'mean_val_mse': float(np.mean(vals)),
        'test_mse_median': float(np.median([per_seed[str(s)]['test_mse'] for s in TRIAL_SEEDS])),
    }
    reg[key] = entry
    save_registry(reg, path, meta)
    return entry


def alignment_check(site_key='56', seed=42):
    """Protocol-alignment red line: default config must reproduce the
    main-table KANConv_M5 row (site 56, seed 42) test norm_MSE from
    kanconv_8site_design.csv."""
    csv = 's_data/cleaned/kanconv_8site_design.csv'
    if not os.path.exists(csv):
        return None
    old = pd.read_csv(csv)
    row = old[(old['site'] == site_key) & (old['config'] == 'KANConv_M5')
              & (old['seed'] == seed)]
    if len(row) == 0:
        return None
    expected = float(row.iloc[0]['norm_MSE'])
    site_file, feats = SITES[site_key]
    m_kw = dict(use_kanconv=True, kan_gate_init=1.0,
                hidden_size=64, num_layers=2, out_channels=48, dropout=0.2)
    _, test_mse, _ = train_one_val(site_file, feats, m_kw, seed, lr=0.01)
    return expected, test_mse


def run_search(site_key, method, n_trials, seed=0, log_csv=None, fixed_epochs=None,
               subset=1.0, subset_mode='prefix', batch=None):
    """Run one search: n_trials proposals from `method` with cached evaluation.

    method: object with propose(history) -> config and report(config, objective).
    history: list of {'config':..., 'objective':..., 'analysis':...}.
    Returns list of trial records.
    """
    rng = np.random.RandomState(seed)
    history = []
    t0 = time.time()
    for i in range(n_trials):
        config = method.propose(history)
        if config is None:
            print(f'  [{method.name}] propose returned None at trial {i}; stop')
            break
        entry = evaluate_config(site_key, config, fixed_epochs=fixed_epochs,
                                subset=subset, subset_mode=subset_mode,
                                batch=batch)
        obj = entry['median_val_mse']
        method.report(config, obj)
        rec = {'trial': i + 1, 'config': config, 'objective': obj,
               'test_mse_median': entry['test_mse_median'],
               'analysis': getattr(method, 'last_analysis', ''),
               'time': round(time.time() - t0, 1)}
        history.append(rec)
        print(f'  [{method.name}] t={i+1}/{n_trials} obj={obj:.5f} '
              f'test={rec["test_mse_median"]:.5f} cfg={json.dumps(config, ensure_ascii=False)}')
    return history


def summarize_groups(groups, site_keys, fixed_epochs=None):
    """groups: {name: [trial records]}; print best config per group vs default."""
    default_entry = {}
    for sk in site_keys:
        default_entry[sk] = evaluate_config(sk, DEFAULT_CONFIG,
                                            fixed_epochs=fixed_epochs)
    print('\n===== GROUP SUMMARY (median val MSE) =====')
    for name, hist in groups.items():
        if not hist:
            continue
        best = min(hist, key=lambda r: r['objective'])
        best_cfg = best['config']
        print(f'-- {name}: best obj={best["objective"]:.5f} test={best["test_mse_median"]:.5f}')
        print(f'   cfg={json.dumps(best_cfg, ensure_ascii=False)}')
    print('\n-- default anchor:')
    for sk in site_keys:
        e = default_entry[sk]
        print(f'   site {sk}: val={e["median_val_mse"]:.5f} test={e["test_mse_median"]:.5f}')
