"""Final validation (R2#5): retrain each group's best config with 10 seeds.

For every E2 group (plus the default anchor), take the best-val config found
by the search and retrain it with SEEDS10 (full protocol), reporting
mean +/- std on val and test (normalized scale) plus paired t-tests:
  - every config vs the default anchor
  - the key E2 contrasts: b vs d, a vs c, a vs b, c vs d, a vs g, a vs sep

The training protocol matches the SEARCH protocol (--epochs N trains every
config exactly N epochs without early stopping; default None = early stop).

Usage:
  python llm_final_validate.py --sites 212 213 --epochs 30
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_LLM_DIR)
sys.path.insert(0, _LLM_DIR)
sys.path.insert(0, _PROJ_DIR)
os.chdir(_PROJ_DIR)          # cnn_lstm_kan/ keeps s_data/ paths working

from llm_search_space import DEFAULT_CONFIG, validate
from llm_runner import SITES, train_one_val

SEEDS = [42, 123, 456, 789, 2026, 11, 22, 33, 44, 55]

GROUP_NAMES = ['a_llm_aug', 'b_llm_noaug', 'c_tpe_aug', 'd_tpe_noaug',
               'f_random_aug', 'g_llm_noctx', 'sep_llm']


def group_csv(site_key, gname, space_tag='small'):
    return f's_data/cleaned/llm_e2_{space_tag}_{gname}_{site_key}.csv'


def best_config_from_csv(path):
    df = pd.read_csv(path)
    row = df.loc[df['objective'].idxmin()]
    cfg = {k: row[k] for k in ('learning_rate', 'hidden_dim', 'num_layers',
                               'out_channels', 'enable_augmentation',
                               'strategy_type', 'sigma', 'knot',
                               'augmentation_factor')}
    return validate(cfg), float(row['objective'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', nargs='*', default=['56'])
    ap.add_argument('--epochs', type=int, default=None,
                    help='fixed budget (matches the search protocol); '
                         'None = early stopping')
    ap.add_argument('--subset', type=float, default=1.0,
                    help='data-scarcity regime: use first fraction of the series')
    args = ap.parse_args()
    fixed = args.epochs
    sub = args.subset

    for site_key in args.sites:
        print(f'\n########## site {site_key} (epochs={fixed or "earlystop"}) ##########')
        site_file, feats = SITES[site_key]
        configs = {'anchor_default': (DEFAULT_CONFIG, None)}
        for gname in GROUP_NAMES:
            path = group_csv(site_key, gname)
            if os.path.exists(path):
                cfg, obj = best_config_from_csv(path)
                configs[gname] = (cfg, obj)
                print(f'  loaded {gname}: best-val config (val={obj:.5f})')

        rows = []
        t0 = time.time()
        for name, (cfg, _) in configs.items():
            per_seed = {}
            for seed in SEEDS:
                m_kw = dict(use_kanconv=True, kan_gate_init=1.0,
                            hidden_size=cfg['hidden_dim'],
                            num_layers=cfg['num_layers'],
                            out_channels=cfg['out_channels'],
                            dropout=cfg['dropout'])
                val, test, ep = train_one_val(
                    site_file, feats, m_kw, seed, lr=cfg['learning_rate'],
                    aug_config=cfg if cfg['enable_augmentation'] else None,
                    fixed_epochs=fixed, subset=sub)
                per_seed[str(seed)] = {'val': val, 'test': test, 'ep': ep}
                rows.append({'site': site_key, 'config': name, 'seed': seed,
                             'val_mse': val, 'test_mse': test, 'best_epoch': ep})
            vals = [per_seed[str(s)]['val'] for s in SEEDS]
            tests = [per_seed[str(s)]['test'] for s in SEEDS]
            print(f'  [{name}] val={np.mean(vals):.5f}+/-{np.std(vals):.5f} '
                  f'test={np.mean(tests):.5f}+/-{np.std(tests):.5f} '
                  f'({time.time()-t0:.0f}s)')
            tag = f'fixed{fixed}' if fixed else 'earlystop'
            if sub < 1.0:
                tag += f'_sub{sub}'
            pd.DataFrame(rows).to_csv(
                f's_data/cleaned/llm_final_validate_{site_key}_{tag}.csv',
                index=False)

        data = pd.DataFrame(rows)
        names = list(configs)
        res = {}
        for a in names:
            for b in names:
                if a >= b:
                    continue
                xa = data[data['config'] == a].sort_values('seed')['test_mse'].values
                xb = data[data['config'] == b].sort_values('seed')['test_mse'].values
                if len(xa) == len(xb) == len(SEEDS):
                    t, p = stats.ttest_rel(xa, xb)
                    res[f'{a} vs {b}'] = p
        print(f'\n=== paired t-test (TEST MSE, 10 seeds, site {site_key}) ===')
        anchor = 'anchor_default'
        keys = [k for k in res
                if k.split(' vs ')[0] == anchor or k.split(' vs ')[1] == anchor
                or (set(k.split(' vs ')) == {'b_llm_noaug', 'd_tpe_noaug'})
                or (set(k.split(' vs ')) == {'a_llm_aug', 'c_tpe_aug'})
                or (set(k.split(' vs ')) == {'a_llm_aug', 'b_llm_noaug'})
                or (set(k.split(' vs ')) == {'c_tpe_aug', 'd_tpe_noaug'})
                or (set(k.split(' vs ')) == {'a_llm_aug', 'g_llm_noctx'})
                or (set(k.split(' vs ')) == {'a_llm_aug', 'sep_llm'})]
        for k in sorted(keys, key=lambda kv: res[kv]):
            print(f'  {k:38s} p={res[k]:.4f} {"*" if res[k] < 0.05 else ""}')


if __name__ == '__main__':
    main()
