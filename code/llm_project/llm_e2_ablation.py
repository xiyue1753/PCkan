"""E2: LLM fairness ablation (R2#6) — groups a/b/c/d/f/g + separate.

Groups (manual group deferred by user decision):
  a_llm_aug      LLM (full context) + augmentation in space   [R2#6: LLM+aug]
  b_llm_noaug    LLM (full context), model-only space         [R2#6: LLM-aug]
  c_tpe_aug      optuna TPE + augmentation in space           [R2#6: HPO+aug]
  d_tpe_noaug    TPE, model-only space                        [R2#6: HPO-aug]
  f_random_aug   random search, full space, same budget       [R2#6: random floor]
  g_llm_noctx    LLM WITHOUT context (numeric history only)   [R2#6: context]
  sep_llm        LLM two-stage: HP (5) then aug (5), equal budget
                 [T1: joint vs separate optimization]

All groups: same search space, same trial budget, same objective (median val
MSE over 3 fixed seeds), shared trial registry (a config is evaluated once).

Usage:
  python llm_e2_ablation.py --smoke            # site 56, 3 trials/group
  python llm_e2_ablation.py --budget 10 --sites 56 73 11
"""
import os, sys
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_LLM_DIR)
sys.path.insert(0, _LLM_DIR)
sys.path.insert(0, _PROJ_DIR)
os.chdir(_PROJ_DIR)          # cnn_lstm_kan/ keeps s_data/ paths working

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LLM_DIR)
os.chdir(os.path.dirname(_LLM_DIR))          # cnn_lstm_kan/ keeps s_data/ paths

import numpy as np
import pandas as pd

from llm_search_space import (SPACE, MODEL_ONLY_SPACE, AUG_ONLY_SPACE,
                              AUG_FORCED_SPACE, COLD_START_FORCED_SPACE,
                              force_aug_on, DEFAULT_CONFIG, MANUAL_CONFIG,
                              sample_random, config_hash, validate, SPACES)
from llm_runner import (SITES, run_search, evaluate_config, load_registry,
                        TRIAL_SEEDS, FIXED_EPOCHS)
from llm_baselines import TPESearch, RandomSearch, ManualSearch, HEBOSearch, DEHBSearch
from llm_agent import LLMAgent

OUT_DIR = 's_data/cleaned'

# fixed sensible augmentation used by the separate group's HP-tuning stage
SEP_FIXED_AUG = {k: MANUAL_CONFIG[k] for k in
                 ('use_mixup', 'use_mixup_phys', 'use_mixup_temporal',
                  'use_jitter', 'sigma_mixup', 'sigma_phys', 'sigma_temporal',
                  'sigma_jitter', 'augmentation_factor')}


class LLMAdapter:
    """Wrap LLMAgent.propose (needs stats/space/base) into the SearchMethod
    interface consumed by run_search.

    Dedup: if the LLM re-proposes a configuration that was already evaluated,
    it gets prompt feedback ("already evaluated with objective X; propose a
    different one") and retries (up to MAX_DUP retries) so the trial budget is
    never wasted on repeats. Falls back to a random sample if the API fails.
    """
    MAX_DUP = 3

    def __init__(self, agent, stats, space, base, seed=0, force_aug=False):
        self.agent = agent
        self.stats = stats
        self.space = space
        self.base = dict(base)
        self.name = agent.name
        self.rng = np.random.RandomState(seed)
        self.force_aug = force_aug
        self.last_analysis = ''

    def propose(self, history):
        seen = {config_hash(h['config']): h['objective'] for h in history}
        feedback = None
        for attempt in range(self.MAX_DUP):
            out = self.agent.propose(history, self.stats, self.space, self.base,
                                     feedback=feedback)
            if out is None:
                cfg = sample_random(self.rng, self.space, self.base)
                return force_aug_on(cfg) if self.force_aug else cfg
            cfg = out['config']
            if self.force_aug:
                cfg = force_aug_on(cfg)
            h = config_hash(cfg)
            if h not in seen:
                self.last_analysis = out['analysis']
                return cfg
            obj = seen[h]
            print(f'    [dedup] config {h[:8]} already evaluated obj={obj:.5f}; '
                  f'asking LLM for a new point')
            feedback = (f'Your proposed config (hash {h[:8]}) was already '
                        f'evaluated with objective {obj:.5f}. Propose a '
                        f'DIFFERENT configuration in a different region of '
                        f'the space.')
        cfg = sample_random(self.rng, self.space, self.base)
        return force_aug_on(cfg) if self.force_aug else cfg

    def report(self, config, objective):
        pass


class SeparateAdapter:
    """Two-stage LLM search with equal total budget:
    stage 1 (n1 trials): model HP only, augmentation fixed at SEP_FIXED_AUG;
    stage 2 (n2 trials): augmentation only, model HP fixed at stage-1 best.
    """
    name = 'llm_separate'

    def __init__(self, agent, stats, n1, n2, seed=0):
        self.agent = agent
        self.stats = stats
        self.n1 = n1
        self.n2 = n2
        self.rng = np.random.RandomState(seed)
        self.base1 = dict(DEFAULT_CONFIG)
        self.base1.update(SEP_FIXED_AUG)
        self._best1 = None
        self.last_analysis = ''

    def propose(self, history):
        n = len(history)
        if n < self.n1:
            out = self.agent.propose(history, self.stats, MODEL_ONLY_SPACE,
                                     self.base1)
        else:
            if self._best1 is None:
                st1 = history[:self.n1]
                if st1:
                    self._best1 = min(st1, key=lambda r: r['objective'])['config']
                else:
                    self._best1 = dict(self.base1)
            out = self.agent.propose(history, self.stats, AUG_ONLY_SPACE,
                                     self._best1)
        if out is None:
            space = MODEL_ONLY_SPACE if len(history) < self.n1 else AUG_ONLY_SPACE
            base = self.base1 if len(history) < self.n1 else self._best1 or self.base1
            return sample_random(self.rng, space, base)
        self.last_analysis = out['analysis']
        return out['config']

    def report(self, config, objective):
        pass


def site_stats(site_key, subset=1.0, mode='prefix'):
    """Cheap raw data statistics for the LLM prompt (no model training)."""
    import torch
    from config import TARGET_COL
    from mode_code import prepare_multistep_data
    df = pd.read_csv(SITES[site_key][0], index_col=0, parse_dates=True)
    feats = SITES[site_key][1]
    X, _ = prepare_multistep_data(df, feature_cols=feats, target_col=TARGET_COL,
                                  seq_len=24, max_horizon=24)
    n = len(X)
    if subset < 1.0:
        n_sub = int(n * subset)
        if mode == 'uniform':
            idx = np.linspace(0, n - 1, n_sub).astype(int)
            X = X[idx]
        else:
            X = X[:n_sub]
        n = n_sub
    tr = int(n * 0.64)
    Xtr = X[:tr]
    ranges = {f: [round(float(Xtr[:, 0, i].min()), 3),
                  round(float(Xtr[:, 0, i].max()), 3)]
              for i, f in enumerate(feats)}
    return {'site': site_key, 'samples': int(tr),
            'features': len(feats),
            'ranges': json.dumps(ranges, ensure_ascii=False)}


GROUPS = {
    'a_llm_aug':      dict(kind='llm', with_context=True,  space='full',
                          force_aug=True),
    'b_llm_noaug':    dict(kind='llm', with_context=True,  space='model'),
    'c_tpe_aug':      dict(kind='tpe', space='full', force_aug=True),
    'd_tpe_noaug':    dict(kind='tpe', space='model'),
    'e_manual_aug':   dict(kind='manual', space='full', force_aug=True),
    'f_random_aug':   dict(kind='random', space='full', force_aug=True),
    'h_hebo_aug':     dict(kind='hebo', space='full', force_aug=True),
    'i_dehb_aug':     dict(kind='dehb', space='full', force_aug=True),
    'g_llm_noctx':    dict(kind='llm', with_context=False, space='full',
                          force_aug=True),
    'sep_llm':        dict(kind='separate'),
}

LLM_GROUPS = ['a_llm_aug', 'b_llm_noaug', 'g_llm_noctx', 'sep_llm']


def build_method(gname, gdef, stats, seed, temperature, space):
    g_space = gdef.get('space', 'full')
    if g_space == 'model':
        space_use = MODEL_ONLY_SPACE
    elif g_space == 'aug':
        space_use = AUG_ONLY_SPACE
    elif gdef.get('force_aug'):
        space_use = AUG_FORCED_SPACE            # augmentation mandatory
        if AUG_SPACE_NAME == 'coldstart':
            space_use = COLD_START_FORCED_SPACE
    else:
        space_use = space                       # full space: small or large
    if gdef['kind'] == 'llm':
        agent = LLMAgent(temperature=temperature, with_context=gdef['with_context'],
                         log_dir='llm_logs')
        return LLMAdapter(agent, stats, space_use, DEFAULT_CONFIG, seed=seed,
                          force_aug=gdef.get('force_aug', False))
    if gdef['kind'] == 'tpe':
        return TPESearch(seed=seed, space=space_use, base=DEFAULT_CONFIG,
                         force_aug=gdef.get('force_aug', False))
    if gdef['kind'] == 'manual':
        return ManualSearch(seed=seed, space=space_use, base=DEFAULT_CONFIG,
                            force_aug=gdef.get('force_aug', False))
    if gdef['kind'] == 'random':
        return RandomSearch(seed=seed, space=space_use, base=DEFAULT_CONFIG,
                            force_aug=gdef.get('force_aug', False))
    if gdef['kind'] == 'hebo':
        return HEBOSearch(seed=seed, space=space_use, base=DEFAULT_CONFIG,
                          force_aug=gdef.get('force_aug', False))
    if gdef['kind'] == 'dehb':
        return DEHBSearch(seed=seed, space=space_use, base=DEFAULT_CONFIG,
                          force_aug=gdef.get('force_aug', False))
    if gdef['kind'] == 'separate':
        agent = LLMAgent(temperature=temperature, with_context=True,
                         log_dir='llm_logs')
        n1 = max(1, BUDGET // 2)
        return SeparateAdapter(agent, stats, n1, BUDGET - n1, seed=seed)
    raise ValueError(gdef['kind'])


def save_group(site_key, gname, history, space_tag, run_id=None, temp=None):
    rows = []
    for r in history:
        rows.append({'site': site_key, 'group': gname, 'space': space_tag,
                     'trial': r['trial'], 'objective': r['objective'],
                     'test_mse_median': r['test_mse_median'],
                     'hash': config_hash(r['config']), **r['config']})
    if rows:
        stem = f'llm_e2_{space_tag}_{gname}_{site_key}'
        fname = f'{stem}.csv'
        if temp is not None:
            fname = f'{stem}_t{temp}.csv'
        if run_id is not None:
            fname = f'{stem}_t{temp}_run{run_id}.csv' if temp is not None \
                else f'{stem}_run{run_id}.csv'
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, fname), index=False)


def main():
    global BUDGET
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--budget', type=int, default=10)
    ap.add_argument('--sites', nargs='*', default=['56', '73', '11'])
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=None,
                    help='fixed budget: disable early stop, train exactly N epochs')
    ap.add_argument('--subset', type=float, default=1.0,
                    help='data-scarcity regime: use first fraction of the series')
    ap.add_argument('--batch', type=int, default=None,
                    help='override batch size (cold-start protocol: 4096)')
    ap.add_argument('--aug-space', choices=['forced', 'coldstart'],
                    default='forced',
                    help='augmentation space for the aug arms')
    ap.add_argument('--space', choices=['small', 'large'], default='small')
    ap.add_argument('--llm-only', action='store_true',
                    help='only rerun the LLM groups (small-space rerun with '
                         'the final method; TPE/random reuse the registry)')
    ap.add_argument('--groups', nargs='*', default=None,
                    help='explicit group list to run (overrides --llm-only)')
    ap.add_argument('--fresh', action='store_true',
                    help='delete the per-(site, protocol) registry files '
                         'before running (clear cache)')
    ap.add_argument('--registry-dir', default=None,
                    help='override registry directory (default s_data/cleaned)')
    ap.add_argument('--seed', type=int, default=42,
                    help='search random seed (multi-run: vary per run)')
    ap.add_argument('--run-id', type=int, default=None,
                    help='when set, results are saved with a _run{id} suffix '
                         '(multi-run repeats do not clobber the main CSVs)')
    args = ap.parse_args()

    global FIXED_EP
    BUDGET = 3 if args.smoke else args.budget
    sites = ['56'] if args.smoke else args.sites
    temp = args.temperature
    FIXED_EP = args.epochs if args.epochs is not None else None
    global SUB, BATCH_OVR, AUG_SPACE_NAME
    SUB = args.subset
    BATCH_OVR = args.batch
    AUG_SPACE_NAME = args.aug_space
    space_name = args.space
    space_full = SPACES[space_name]
    e2_json = os.path.join(OUT_DIR, f'llm_e2_{space_name}_results.json')
    if args.groups is not None:
        groups = {k: v for k, v in GROUPS.items() if k in args.groups}
    elif args.llm_only:
        groups = {k: v for k, v in GROUPS.items() if k in LLM_GROUPS}
    else:
        groups = dict(GROUPS)
    print(f'E2 ablation | sites={sites} budget={BUDGET} temp={temp} '
          f'space={space_name} groups={list(groups)} '
          f'fixed_epochs={FIXED_EP} subset={SUB}')

    if args.registry_dir:
        os.environ['LLM_REGISTRY_DIR'] = args.registry_dir
    if args.fresh:
        import glob as _glob
        for sk in sites:
            tag = f'fixed{FIXED_EP}' if FIXED_EP else 'earlystop'
            for f in _glob.glob(os.path.join(
                    os.environ.get('LLM_REGISTRY_DIR', OUT_DIR),
                    f'llm_trial_registry_{sk}_{tag}.json')):
                os.remove(f)
                print(f'  [fresh] removed {f}')

    results = {}
    for site_key in sites:
        stats = site_stats(site_key, subset=SUB, mode='prefix')
        for gname, gdef in groups.items():
            print(f'\n===== {gname} @ site {site_key} ({space_name}) '
                  f'seed={args.seed} run={args.run_id} =====')
            method = build_method(gname, gdef, stats, seed=args.seed,
                                  temperature=temp, space=space_full)
            t0 = time.time()
            hist = run_search(site_key, method, BUDGET, fixed_epochs=FIXED_EP, subset=SUB,
                  batch=BATCH_OVR)
            save_group(site_key, gname, hist, space_name, run_id=args.run_id,
                       temp=temp)
            print(f'  [{gname}] done in {time.time()-t0:.0f}s')
            if hist:
                best = min(hist, key=lambda r: r['objective'])
                results.setdefault(gname, {})[site_key] = {
                    'best_obj': best['objective'],
                    'best_config': best['config'],
                    'n_trials': len(hist),
                    'best_test': best['test_mse_median'],
                }
                print(f'  best obj={best["objective"]:.5f} '
                      f'test={best["test_mse_median"]:.5f}')

    # anchor (default config) for delta reporting — SAME protocol as the
    # trials (fixed_epochs must match), and stored in the results json
    print('\n===== ANCHOR (default config) =====')
    anchors = {}
    for sk in sites:
        e = evaluate_config(sk, DEFAULT_CONFIG, fixed_epochs=FIXED_EP, subset=SUB,
                    batch=BATCH_OVR)
        anchors[sk] = {'val': e['median_val_mse'], 'test': e['test_mse_median']}
        print(f'  site {sk}: val={e["median_val_mse"]:.5f} test={e["test_mse_median"]:.5f}')

    with open(e2_json, 'w', encoding='utf-8') as f:
        json.dump({'budget': BUDGET, 'sites': sites, 'temperature': temp,
                   'space': space_name, 'llm_only': args.llm_only,
                   'fixed_epochs': FIXED_EP,
                   'anchor': anchors, 'results': results},
                  f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
