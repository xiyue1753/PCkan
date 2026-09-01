"""Cold-start re-verification of the augmentation scheme (direction A).

Cold-start protocol (prefix subsetting, the real new-plant scenario):
  sites 212/213, subset 0.25 (chronological prefix, ~1.8-2.9K windows),
  fixed 100 epochs (long training — the regime where the old mixup result
  was significant: -13.2%, p=0.0004 at 212-25%), batch 4096.

Configs:
  none                     baseline
  mixup_s05_append         sigma=0.5 factor=2 (the old significant config)
  mixup_s05_replace25      sigma=0.5, 25% replaced (selection-style, no dilution)
  mixup_s02_replace25      sigma=0.2, 25% replaced
  jitter_ghi_append        sigma=0.05 factor=2 (label-preserving reference)

Usage: python llm_coldstart_verify.py
"""
import os
import sys

import numpy as np
import pandas as pd

_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_LLM_DIR)
sys.path.insert(0, _LLM_DIR)
sys.path.insert(0, _PROJ_DIR)
os.chdir(_PROJ_DIR)

from llm_search_space import DEFAULT_CONFIG, validate
from llm_runner import SITES, train_one_val, TRIAL_SEEDS

SITES2 = ['212', '213']
SCALE = 0.25
EPOCHS = 100
BATCH = 4096

CONFIGS = [
    ('none',               {}, 'append'),
    ('mixup_s05_append',   dict(enable_augmentation=True,
                                strategy_type='mixup', sigma=0.5,
                                knot=4, augmentation_factor=2.0), 'append'),
    ('mixup_s05_rep25',    dict(enable_augmentation=True,
                                strategy_type='mixup', sigma=0.5,
                                knot=4, augmentation_factor=1.333), 'replace'),
    ('mixup_s02_rep25',    dict(enable_augmentation=True,
                                strategy_type='mixup', sigma=0.2,
                                knot=4, augmentation_factor=1.333), 'replace'),
    ('jitter_append',      dict(enable_augmentation=True,
                                strategy_type='jitter_ghi', sigma=0.05,
                                knot=4, augmentation_factor=2.0), 'append'),
]


def main():
    rows = []
    print(f'===== cold-start verification | prefix {SCALE} | {EPOCHS}ep | '
          f'batch {BATCH} | 3 seeds =====')
    for site in SITES2:
        site_file, feats = SITES[site]
        results = {}
        for name, over, mode in CONFIGS:
            cfg = validate({**DEFAULT_CONFIG, **over})
            vals, tests = [], []
            m_kw = dict(use_kanconv=True, kan_gate_init=1.0,
                        hidden_size=cfg['hidden_dim'],
                        num_layers=cfg['num_layers'],
                        out_channels=cfg['out_channels'],
                        dropout=cfg['dropout'])
            for seed in TRIAL_SEEDS:
                val, test, _ = train_one_val(
                    site_file, feats, m_kw, seed, lr=cfg['learning_rate'],
                    aug_config=cfg if cfg['enable_augmentation'] else None,
                    fixed_epochs=EPOCHS, subset=SCALE, subset_mode='prefix',
                    batch=BATCH, aug_mode=mode)
                vals.append(val); tests.append(test)
                rows.append({'site': site, 'config': name, 'seed': seed,
                             'val_mse': val, 'test_mse': test})
            results[name] = (np.mean(vals), np.mean(tests))
            pd.DataFrame(rows).to_csv('s_data/cleaned/llm_coldstart_verify.csv',
                                      index=False)
        base = results['none'][0]
        print(f'-- {site}:')
        for name, (m, t) in results.items():
            print(f'    {name:18s} val={m:.5f} ({(m/base-1)*100:+.1f}%) '
                  f'test={t:.5f}')

    print('\n===== SUMMARY =====')
    df = pd.DataFrame(rows)
    for site in SITES2:
        d = df[df['site'] == site]
        base = d[d['config'] == 'none']['val_mse'].mean()
        print(f'-- {site}:')
        for name, _, _ in CONFIGS:
            v = d[d['config'] == name]['val_mse'].mean()
            print(f'    {name:18s} {(v/base-1)*100:+.1f}% vs none')


if __name__ == '__main__':
    main()
