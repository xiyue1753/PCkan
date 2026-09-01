"""Cold-start 10-seed confirmation (the money result).

Sites 212/213, prefix 0.25, fixed 100ep, batch 4096, 10 seeds:
  none vs mixup_s05_append vs mixup_s05_rep25 vs jitter_append
Paired t-tests per site and pooled.
Usage: python llm_coldstart_confirm.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_LLM_DIR)
sys.path.insert(0, _LLM_DIR)
sys.path.insert(0, _PROJ_DIR)
os.chdir(_PROJ_DIR)

from llm_search_space import DEFAULT_CONFIG, validate
from llm_runner import SITES, train_one_val

SITES2 = ['212', '213']
SCALE = 0.25
EPOCHS = 100
BATCH = 4096
SEEDS10 = [42, 123, 456, 789, 2026, 11, 22, 33, 44, 55]

CONFIGS = [
    ('none',               {}, 'append'),
    ('mixup_s05_append',   dict(enable_augmentation=True,
                                strategy_type='mixup', sigma=0.5,
                                knot=4, augmentation_factor=2.0), 'append'),
    ('mixup_s05_rep25',    dict(enable_augmentation=True,
                                strategy_type='mixup', sigma=0.5,
                                knot=4, augmentation_factor=1.333), 'replace'),
    ('jitter_append',      dict(enable_augmentation=True,
                                strategy_type='jitter_ghi', sigma=0.05,
                                knot=4, augmentation_factor=2.0), 'append'),
]


def main():
    rows = []
    print(f'===== cold-start 10-seed confirm | prefix {SCALE} | {EPOCHS}ep =====')
    for site in SITES2:
        site_file, feats = SITES[site]
        for name, over, mode in CONFIGS:
            cfg = validate({**DEFAULT_CONFIG, **over})
            m_kw = dict(use_kanconv=True, kan_gate_init=1.0,
                        hidden_size=cfg['hidden_dim'],
                        num_layers=cfg['num_layers'],
                        out_channels=cfg['out_channels'],
                        dropout=cfg['dropout'])
            for seed in SEEDS10:
                val, test, _ = train_one_val(
                    site_file, feats, m_kw, seed, lr=cfg['learning_rate'],
                    aug_config=cfg if cfg['enable_augmentation'] else None,
                    fixed_epochs=EPOCHS, subset=SCALE, subset_mode='prefix',
                    batch=BATCH, aug_mode=mode)
                rows.append({'site': site, 'config': name, 'seed': seed,
                             'val_mse': val, 'test_mse': test})
        d = pd.DataFrame([r for r in rows if r['site'] == site])
        base = d[d['config'] == 'none']['val_mse'].mean()
        print(f'-- {site}:')
        for name, _, _ in CONFIGS:
            v = d[d['config'] == name]['val_mse']
            print(f'    {name:18s} val={v.mean():.5f}+/-{v.std():.5f} '
                  f'({(v.mean()/base-1)*100:+.1f}%)')
        pd.DataFrame(rows).to_csv('s_data/cleaned/llm_coldstart_confirm.csv',
                                  index=False)

    print('\n===== PAIRED t-TESTS (val, 10 seeds) =====')
    df = pd.DataFrame(rows)
    for site in SITES2:
        d = df[df['site'] == site]
        none = d[d['config'] == 'none'].sort_values('seed')['val_mse'].values
        for name, _, _ in CONFIGS:
            if name == 'none':
                continue
            x = d[d['config'] == name].sort_values('seed')['val_mse'].values
            t, p = stats.ttest_rel(none, x)
            dlt = (x.mean() / none.mean() - 1) * 100
            print(f'  {site}: none vs {name:18s} {dlt:+.1f}% p={p:.4f} '
                  f'{"*" if p < 0.05 else ""}')
    # pooled
    for name, _, _ in CONFIGS:
        if name == 'none':
            continue
        none = df[df['config'] == 'none'].sort_values(['site', 'seed'])['val_mse'].values
        x = df[df['config'] == name].sort_values(['site', 'seed'])['val_mse'].values
        t, p = stats.ttest_rel(none, x)
        print(f'  POOLED: none vs {name:18s} {(x.mean()/none.mean()-1)*100:+.1f}% '
              f'p={p:.4f} {"*" if p < 0.05 else ""}')


if __name__ == '__main__':
    main()
