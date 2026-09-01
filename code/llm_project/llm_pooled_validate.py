"""Pooled final-validation analysis across sites.

Reads llm_final_validate_{site}_fixed{epochs}.csv files and runs PAIRED
t-tests on the pooled TEST MSE (pairs = site x 10 seeds), giving the
cross-site statistical picture the single-site numbers cannot provide.

Usage: python llm_pooled_validate.py --sites 212 213 --epochs 30
"""
import argparse
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

CONFIGS = ['anchor_default', 'a_llm_aug', 'b_llm_noaug', 'c_tpe_aug',
           'd_tpe_noaug', 'f_random_aug', 'g_llm_noctx', 'sep_llm']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', nargs='*', default=['212', '213'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--subset', type=float, default=1.0)
    args = ap.parse_args()
    tag = f'fixed{args.epochs}' if args.epochs else 'earlystop'
    if args.subset < 1.0:
        tag += f'_sub{args.subset}'

    frames = []
    for sk in args.sites:
        path = f's_data/cleaned/llm_final_validate_{sk}_{tag}.csv'
        if not os.path.exists(path):
            print(f'  missing {path}')
            continue
        df = pd.read_csv(path)
        frames.append(df)
        print(f'  loaded {sk}: {len(df)} rows')
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        print('no data')
        return

    print(f'\n=== pooled mean (test MSE, n = {len(data)} per pair set) ===')
    for c in CONFIGS:
        sub = data[data['config'] == c]
        if len(sub):
            print(f'  {c:18s} test={sub["test_mse"].mean():.5f}+/-{sub["test_mse"].std():.5f} '
                  f'(val {sub["val_mse"].mean():.5f})')

    print('\n=== pooled paired t-tests (test MSE) ===')
    res = {}
    for a in CONFIGS:
        for b in CONFIGS:
            if a >= b:
                continue
            xa = data[data['config'] == a].sort_values(['site', 'seed'])['test_mse'].values
            xb = data[data['config'] == b].sort_values(['site', 'seed'])['test_mse'].values
            if len(xa) == len(xb) and len(xa) >= 10:
                t, p = stats.ttest_rel(xa, xb)
                res[f'{a} vs {b}'] = p
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
