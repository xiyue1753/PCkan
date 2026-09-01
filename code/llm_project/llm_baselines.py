"""Classical HPO baselines behind one SearchMethod interface.

All baselines receive ONLY numeric objectives (config + val MSE) — never the
text context given to the LLM (R2#6 fairness condition).

Methods:
  random  — uniform sampling over the joint space
  tpe     — optuna TPESampler
  cmaes   — optuna CmaEsSampler
  hebo    — HEBO (Bayesian optimization, hebo 0.3.6)
  dehb    — simplified DEHB ported from the old paper notebook
  manual  — greedy coordinate-wise tuning simulating a human expert:
            one parameter is varied per trial; if it improves the objective
            the next candidate of the SAME parameter is tried, otherwise the
            search falls back and moves to the next parameter in a fixed
            expert order (mimics a practitioner's manual tuning loop).
"""
import random as _random

import numpy as np
import optuna
from optuna.samplers import TPESampler, CmaEsSampler

from llm_search_space import (sample_random, validate, SPACE, force_aug_on,
                              optuna_config_from_trial,
                              hebo_space, config_from_hebo_row,
                              pd_row_from_config, DEFAULT_CONFIG)


class ManualSearch:
    """Greedy coordinate-wise tuning (simulated human expert).

    Starts from the default config (augmentation forced on for the aug arms)
    and walks an expert-ordered list of parameters. Per trial it changes ONE
    parameter to the next candidate value: an improvement keeps the change and
    continues with the next candidate of the same parameter; a regression
    reverts to the previous best and moves on to the next parameter.
    """

    name = 'manual'

    # expert tuning order: most impactful / most commonly tuned first
    PARAM_ORDER = [
        ('learning_rate', [0.005, 0.003, 0.002, 0.001, 0.02]),
        ('hidden_dim', [96, 128, 160, 192]),
        ('out_channels', [96, 128, 160, 192]),
        ('dropout', [0.1, 0.3, 0.4]),
        ('num_layers', [1, 3]),
        ('augmentation_factor', [2.0, 2.5, 3.0]),
        ('sigma_mixup', [0.3, 0.5, 0.2]),
        ('use_mixup_phys', [True]),
        ('use_mixup_temporal', [True]),
        ('use_jitter', [True]),
    ]

    def __init__(self, seed=0, space=None, base=None, force_aug=False):
        self.space = space or SPACE
        self.base = dict(base or DEFAULT_CONFIG)
        self.force_aug = force_aug
        self._param_i = 0
        self._cand_i = 0
        self._best_cfg = None
        self._best_obj = float('inf')
        self._last_cfg = None
        self._last_obj = None
        self._first = True
        if force_aug:
            self.base = force_aug_on(self.base)

    def _current_cfg(self):
        """Config to evaluate this trial (before knowing the previous outcome):
        the previous trial's result decides, so the state is resolved lazily in
        propose() via _last_obj."""
        base = self._best_cfg if self._best_cfg is not None else self.base
        if self._param_i >= len(self.PARAM_ORDER):
            return None
        key, cands = self.PARAM_ORDER[self._param_i]
        if self._cand_i >= len(cands):
            return None
        cfg = dict(base)
        cfg[key] = cands[self._cand_i]
        return validate(cfg)

    def propose(self, history):
        if self._first:
            self._first = False
            cfg = force_aug_on(dict(self.base)) if self.force_aug \
                else validate(dict(self.base))
            self._best_cfg = cfg
            return cfg
        # consume the previous trial's outcome exactly once
        if self._last_obj is not None:
            if self._last_obj < self._best_obj - 1e-9:
                # improvement: keep the change, continue same parameter
                self._best_cfg = self._last_cfg
                self._best_obj = self._last_obj
                self._cand_i += 1
            else:
                # no improvement: revert, move to the next parameter
                self._cand_i = 0
                self._param_i += 1
            self._last_obj = None
        # skip parameters with no remaining candidates
        while self._param_i < len(self.PARAM_ORDER) and \
                self._cand_i >= len(self.PARAM_ORDER[self._param_i][1]):
            self._cand_i = 0
            self._param_i += 1
        if self._param_i >= len(self.PARAM_ORDER):
            return None
        key, cands = self.PARAM_ORDER[self._param_i]
        base = self._best_cfg if self._best_cfg is not None else self.base
        cfg = validate(dict(base))
        cfg[key] = cands[self._cand_i]
        if self.force_aug:
            cfg = force_aug_on(cfg)
        self._last_cfg = validate(cfg)
        return self._last_cfg

    def report(self, config, objective):
        self._last_obj = objective
        self._last_cfg = validate(config)


class RandomSearch:
    name = 'random'

    def __init__(self, seed=0, space=None, base=None, force_aug=False):
        self.rng = _random.Random(seed)
        self.space = space or SPACE
        self.base = dict(base or DEFAULT_CONFIG)
        self.force_aug = force_aug

            

    def propose(self, history):
        cfg = sample_random(self.rng, self.space, self.base)
        return force_aug_on(cfg) if self.force_aug else cfg

    def report(self, config, objective):
        pass


class _OptunaSearch:
    sampler = None

    def __init__(self, seed=0, n_startup=5, space=None, base=None,
                 force_aug=False):
        self.space = space or SPACE
        self.base = dict(base or DEFAULT_CONFIG)
        self.force_aug = force_aug

            
        self.study = optuna.create_study(
            direction='minimize',
            sampler=self.sampler(seed=seed, n_startup_trials=n_startup))
        self._pending = {}

    def _key(self, cfg):
        return tuple(cfg[k] for k in self.space)

    def propose(self, history):
        trial = self.study.ask()
        cfg = optuna_config_from_trial(trial, self.space, self.base)
        if self.force_aug:
            cfg = force_aug_on(cfg)
        self._pending[self._key(cfg)] = trial
        return cfg

    def report(self, config, objective):
        cfg = validate(config)
        trial = self._pending.pop(self._key(cfg))
        self.study.tell(trial, objective)


class TPESearch(_OptunaSearch):
    name = 'tpe'
    sampler = TPESampler

    def __init__(self, seed=0, space=None, base=None, force_aug=False):
        super().__init__(seed=seed, n_startup=5, space=space, base=base,
                         force_aug=force_aug)


class CMAESSearch(_OptunaSearch):
    name = 'cmaes'
    sampler = CmaEsSampler

    def __init__(self, seed=0, space=None, base=None, force_aug=False):
        super().__init__(seed=seed, n_startup=5, space=space, base=base,
                         force_aug=force_aug)


class HEBOSearch:
    name = 'hebo'

    def __init__(self, seed=0, space=None, base=None, force_aug=False):
        from hebo.optimizers.hebo import HEBO
        self.space = space or SPACE
        self.base = dict(base or DEFAULT_CONFIG)
        self.force_aug = force_aug

            
        self.opt = HEBO(hebo_space(self.space), scramble_seed=seed)
        self.n_obs = 0

    def propose(self, history):
        rec = self.opt.suggest(n_suggestions=1)
        cfg = config_from_hebo_row(rec, self.space, self.base)
        return force_aug_on(cfg) if self.force_aug else cfg

    def report(self, config, objective):
        cfg = validate(config)
        row = pd_row_from_config(cfg, self.space)
        self.opt.observe(row, np.array([objective]))
        self.n_obs += 1


class DEHBSearch:
    """Simplified DEHB (ported from the old paper's hand-rolled version):
    population of pop_size individuals; each trial either initializes the
    population (uniform random) or creates a child by crossover + mutation
    of the current best and a random parent; fitness replaces the worst.
    """
    name = 'dehb'

    def __init__(self, seed=0, pop_size=5, mut_rate=0.3, cross_rate=0.7,
                 space=None, base=None, force_aug=False):
        self.rng = np.random.RandomState(seed)
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.space = space or SPACE
        self.base = dict(base or DEFAULT_CONFIG)
        self.force_aug = force_aug

            
        self.pop = []            # list of configs
        self.fitness = []        # aligned val objectives
        self._seed = seed

    def propose(self, history):
        if len(self.pop) < self.pop_size:
            cfg = sample_random(_random.Random(self.rng.randint(0, 2**31)),
                                self.space, self.base)
            if self.force_aug:
                cfg = force_aug_on(cfg)
            self.pop.append(cfg)
            self.fitness.append(float('inf'))
            return cfg
        # child = crossover(best, random parent) + mutation
        best_i = int(np.argmin(self.fitness))
        parent2 = int(self.rng.randint(0, self.pop_size - 1))
        if parent2 >= best_i:
            parent2 += 1
        p1, p2 = self.pop[best_i], self.pop[parent2]
        child = {}
        for k in p1:
            child[k] = p1[k] if self.rng.random() < self.cross_rate else p2[k]
        child = self._mutate(child)
        child = validate(child) or sample_random(space=self.space, base=self.base)
        self.pop.append(child)
        self.fitness.append(float('inf'))
        return child

    def _mutate(self, cfg):
        out = dict(cfg)
        for k, d in self.space.items():
            if self.rng.random() >= self.mut_rate:
                continue
            if d['type'] == 'log_float':
                lo, hi = np.log10(d['lb']), np.log10(d['ub'])
                out[k] = 10 ** self.rng.uniform(lo, hi)
            elif d['type'] == 'float':
                out[k] = self.rng.uniform(d['lb'], d['ub'])
            elif d['type'] == 'int':
                out[k] = int(self.rng.randint(d['lb'], d['ub']))
            elif d['type'] == 'bool':
                out[k] = self.rng.random() < 0.5
            else:  # int_choice / enum
                out[k] = self.rng.choice(d['values'])
        return validate(out) or cfg

    def report(self, config, objective):
        cfg = validate(config)
        if len(self.fitness) > self.pop_size:
            # find the pending child slot
            for i in range(len(self.fitness) - 1, -1, -1):
                if self.fitness[i] == float('inf') and self.pop[i] == cfg:
                    self.fitness[i] = objective
                    break
            else:
                self.fitness[-1] = objective
        else:
            i = len(self.fitness) - 1
            self.fitness[i] = objective
        # prune: keep pop_size best when population full
        while len(self.pop) > self.pop_size:
            worst = int(np.argmax(self.fitness))
            self.pop.pop(worst)
            self.fitness.pop(worst)


BASELINES = {'random': RandomSearch, 'tpe': TPESearch, 'cmaes': CMAESSearch,
             'hebo': HEBOSearch, 'dehb': DEHBSearch}


def make_baseline(name, seed=0, space=None, base=None, force_aug=False):
    return BASELINES[name](seed=seed, space=space, base=base,
                           force_aug=force_aug)
