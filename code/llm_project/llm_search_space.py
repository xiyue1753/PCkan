"""Unified HPO search space for the LLM + augmentation joint-optimization study.

Flat config format shared by all search methods (LLM / random / TPE / CMA-ES /
HEBO / DEHB) and the trial registry:

    {
      "learning_rate":      float in [1e-5, 0.5]   (log-uniform)
      "hidden_dim":         int   multiple of 16 in [16, 256]
      "num_layers":         int   in {1, 2, 3}
      "out_channels":       int   multiple of 16 in [16, 256]
      "dropout":            float in [0, 0.8]
      "use_mixup":          bool  (may be combined with any of the others)
      "use_mixup_phys":     bool
      "use_mixup_temporal": bool
      "use_jitter":         bool
      "sigma_mixup":        float in [0.01, 0.5]  (Beta shape for the mixup family)
      "sigma_phys":         float in [0.01, 0.5]
      "sigma_temporal":     float in [0.01, 0.5]
      "sigma_jitter":       float in [0.01, 0.5]  (noise std, irradiance channels)
      "augmentation_factor": float in [1.0, 3.0]   (dataset multiplier)
    }

Augmentation is expressed per-strategy: each of the four strategies has its
own on/off switch (use_*) and its own sigma. ANY subset may be enabled at
once (single strategy, several strategies, or none) - the search method
decides the composition; nothing is pre-combined.

Training protocol is FIXED (matches the main-table protocol so HPO results are
directly comparable): seqlen 24, horizon 24, MinMax(-1,1) normalization,
normalized-scale validation MSE as the objective.
"""
import hashlib
import json
import math
import random

MODEL_KEYS = ('learning_rate', 'hidden_dim', 'num_layers', 'out_channels',
              'dropout')
AUG_USE_KEYS = ('use_mixup', 'use_mixup_phys', 'use_mixup_temporal',
                'use_jitter')
AUG_SIGMA_KEYS = ('sigma_mixup', 'sigma_phys', 'sigma_temporal', 'sigma_jitter')
AUG_KEYS = AUG_USE_KEYS + AUG_SIGMA_KEYS + ('augmentation_factor',)

FIXED_PROTOCOL = dict(epochs=100, patience=30, batch_size=4096,
                      seqlen=24, horizon=24)

# Hyperparameter bounds: LEGAL bounds (nothing pre-trimmed to "reasonable"
# values except the ones that are hard constraints of the implementation:
# hidden/out are channel counts, must be multiples of 16; num_layers > 3 is
# excluded for compute cost, not for quality).
MULT16 = list(range(16, 257, 16))               # 16..256 step 16
STRATEGIES = ('mixup', 'mixup_phys', 'mixup_temporal', 'jitter_ghi')
STRATEGY_VALUES = ['none'] + list(STRATEGIES)
LEGACY_STRATEGIES = ['masking', 'mixup+jitter_ghi', 'jitter', 'scaling',
                     'time_warp', 'magnitude_warp', 'jitter+scaling']
VALIDATE_STRATEGIES = STRATEGY_VALUES + LEGACY_STRATEGIES

SPACE = {
    'learning_rate':      {'type': 'log_float', 'lb': 1e-5, 'ub': 0.5},
    'hidden_dim':         {'type': 'int_choice', 'values': MULT16},
    'num_layers':         {'type': 'int_choice', 'values': [1, 2, 3]},
    'out_channels':       {'type': 'int_choice', 'values': MULT16},
    'dropout':            {'type': 'float', 'lb': 0.0, 'ub': 0.8},
    'use_mixup':          {'type': 'bool'},
    'use_mixup_phys':     {'type': 'bool'},
    'use_mixup_temporal': {'type': 'bool'},
    'use_jitter':         {'type': 'bool'},
    'sigma_mixup':        {'type': 'float', 'lb': 0.01, 'ub': 0.5},
    'sigma_phys':         {'type': 'float', 'lb': 0.01, 'ub': 0.5},
    'sigma_temporal':     {'type': 'float', 'lb': 0.01, 'ub': 0.5},
    'sigma_jitter':       {'type': 'float', 'lb': 0.01, 'ub': 0.5},
    'augmentation_factor': {'type': 'float', 'lb': 1.0, 'ub': 3.0},
}

# Reduced sub-spaces used by the ablation groups:
#   MODEL_ONLY  — groups b/d (no-augmentation): search model dims only
#   AUG_ONLY    — stage-2 of the separate-optimization group: search
#                 augmentation dims only with model dims held fixed
MODEL_ONLY_SPACE = {k: SPACE[k] for k in MODEL_KEYS}
AUG_ONLY_SPACE = {k: SPACE[k] for k in AUG_KEYS}

# Forced-augmentation space: the "with augmentation" arms (LLM group a,
# TPE group c, random group f, no-context group g, manual group e) MUST apply
# augmentation - validate() guarantees at least one use_* is True, so the
# aug arms cannot degenerate to "no augmentation". factor >= 1.5 guarantees
# at least one REAL augmented copy: factor=1.0 would round to 1 copy and
# degenerate augmentation to a no-op (the "cheat" the constraint prevents).
AUG_FORCED_SPACE = dict(SPACE)
AUG_FORCED_SPACE['augmentation_factor'] = {'type': 'float', 'lb': 1.5, 'ub': 3.0}

# Cold-start forced space (alias): under the cold-start protocol the same
# legal space applies; no narrower bounds are imposed.
COLD_START_FORCED_SPACE = dict(AUG_FORCED_SPACE)


def aug_enabled(config):
    """True when at least one augmentation strategy is switched on."""
    return any(bool(config.get(k)) for k in AUG_USE_KEYS)


def force_aug_on(config):
    """Lock augmentation ON for the forced arms: ensure at least one use_*
    is True and factor >= 1.5 (at least one real augmented copy)."""
    c = dict(config)
    if not aug_enabled(c):
        c['use_mixup'] = True
    c = validate(c)
    if c is None:
        return None
    c['augmentation_factor'] = max(c['augmentation_factor'],
                                   AUG_FORCED_SPACE['augmentation_factor']['lb'])
    return validate(c)


def _snap_mult16(v):
    best = min(MULT16, key=lambda x: abs(x - int(v)))
    return best


def validate(config):
    """Check a raw config dict against the space; return normalized config
    or None. Accepts both the per-strategy (use_*/sigma_*) format and the
    legacy single-strategy (enable_augmentation/strategy_type/sigma) format,
    mapping the latter onto the per-strategy one."""
    if not isinstance(config, dict):
        return None
    out = {}
    try:
        lr = float(config.get('learning_rate'))
        if not (SPACE['learning_rate']['lb'] <= lr <= SPACE['learning_rate']['ub']):
            return None
        out['learning_rate'] = lr
        hd = int(config.get('hidden_dim'))
        if hd not in SPACE['hidden_dim']['values']:
            hd = _snap_mult16(hd)
        out['hidden_dim'] = hd
        nl = int(config.get('num_layers'))
        if nl not in SPACE['num_layers']['values']:
            return None
        out['num_layers'] = nl
        oc = int(config.get('out_channels'))
        if oc not in SPACE['out_channels']['values']:
            oc = _snap_mult16(oc)
        out['out_channels'] = oc
        dp = float(config.get('dropout', 0.2))
        out['dropout'] = min(max(dp, SPACE['dropout']['lb']),
                             SPACE['dropout']['ub'])
        # per-strategy switches: from new fields, or legacy strategy_type
        st = str(config.get('strategy_type', 'none'))
        if st not in VALIDATE_STRATEGIES:
            return None
        legacy_on = bool(config.get('enable_augmentation', False))
        sig = float(config.get('sigma', 0.2))
        sig = min(max(sig, SPACE['sigma_mixup']['lb']),
                  SPACE['sigma_mixup']['ub'])
        use_map = {'mixup': 'use_mixup', 'mixup_phys': 'use_mixup_phys',
                   'mixup_temporal': 'use_mixup_temporal',
                   'jitter_ghi': 'use_jitter'}
        for sk, uk in use_map.items():
            if uk in config:
                out[uk] = bool(config[uk])
            elif legacy_on and st == sk:
                out[uk] = True
            else:
                out[uk] = False
        for uk, sk in zip(AUG_USE_KEYS, ('sigma_mixup', 'sigma_phys',
                                         'sigma_temporal', 'sigma_jitter')):
            v = float(config.get(sk, 0.2))
            v = min(max(v, SPACE[sk]['lb']), SPACE[sk]['ub'])
            out[sk] = round(v, 6)
        f = float(config.get('augmentation_factor', 1.0))
        f = min(max(f, SPACE['augmentation_factor']['lb']),
                SPACE['augmentation_factor']['ub'])
        out['augmentation_factor'] = round(f, 4)
    except (TypeError, ValueError):
        return None
    if not aug_enabled(out):
        out['augmentation_factor'] = 1.0
        for sk in AUG_SIGMA_KEYS:
            out[sk] = round(SPACE[sk]['lb'], 6)
    return out


def sample_random(rng=None, space=None, base=None):
    """Uniformly sample one config from the (sub)space.

    space=None -> full SPACE; a reduced space (MODEL_ONLY_SPACE / AUG_ONLY_SPACE)
    only samples the given keys; the remaining fields are inherited from
    `base` (default DEFAULT_CONFIG) and validate() normalizes them.
    """
    rng = rng or random
    space = space or SPACE
    base = base or DEFAULT_CONFIG
    cand = dict(base)
    for k, d in space.items():
        if d['type'] == 'log_float':
            cand[k] = 10 ** rng.uniform(math.log10(d['lb']), math.log10(d['ub']))
        elif d['type'] == 'float':
            cand[k] = rng.uniform(d['lb'], d['ub'])
        elif d['type'] == 'int':
            cand[k] = rng.randint(d['lb'], d['ub'])
        elif d['type'] == 'bool':
            cand[k] = rng.random() < 0.5
        elif d['type'] in ('int_choice', 'enum', 'choice'):
            cand[k] = rng.choice(d['values'])
    return validate(cand)


def config_hash(config):
    """Stable hash over the canonical (sorted) config JSON."""
    c = validate(config)
    if c is None:
        raise ValueError(f'config does not validate: {config}')
    blob = json.dumps(c, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def space_schema_str(space=None):
    """JSON schema description handed to the LLM in the prompt."""
    return json.dumps(space or SPACE, indent=2, ensure_ascii=False)


def config_to_flat(config):
    return validate(config)


def flat_to_optuna_kwargs(config):
    """Map a config to optuna suggest_* kwargs (used by TPE / CMA-ES)."""
    return {k: config[k] for k in SPACE}


def optuna_config_from_trial(trial, space=None, base=None):
    space = space or SPACE
    base = base or DEFAULT_CONFIG
    cand = dict(base)
    for k, d in space.items():
        if d['type'] == 'log_float':
            cand[k] = trial.suggest_float(k, d['lb'], d['ub'], log=True)
        elif d['type'] == 'float':
            cand[k] = trial.suggest_float(k, d['lb'], d['ub'])
        elif d['type'] == 'int':
            cand[k] = trial.suggest_int(k, d['lb'], d['ub'])
        elif d['type'] == 'bool':
            cand[k] = trial.suggest_categorical(k, [True, False])
        elif d['type'] in ('int_choice', 'enum', 'choice'):
            cand[k] = trial.suggest_categorical(k, d['values'])
    return validate(cand)


def hebo_space(space=None):
    """HEBO DesignSpace definition mirroring (a sub-space of) SPACE."""
    from hebo.design_space.design_space import DesignSpace
    space = space or SPACE
    hb = []
    for k, d in space.items():
        if d['type'] == 'log_float':
            hb.append({'name': k, 'type': 'num', 'lb': d['lb'], 'ub': d['ub'],
                       'transform': 'log'})
        elif d['type'] == 'float':
            hb.append({'name': k, 'type': 'num', 'lb': d['lb'], 'ub': d['ub']})
        elif d['type'] == 'int':
            hb.append({'name': k, 'type': 'int', 'lb': d['lb'], 'ub': d['ub']})
        elif d['type'] == 'bool':
            hb.append({'name': k, 'type': 'int', 'lb': 0, 'ub': 1})
        elif d['type'] in ('int_choice', 'enum', 'choice'):
            hb.append({'name': k, 'type': 'int', 'lb': 0,
                       'ub': len(d['values']) - 1})         # index
    return DesignSpace().parse(hb)


def _hebo_scalar(d, k):
    if hasattr(d, 'iloc'):
        return float(d.iloc[0][k])
    return float(d[k])


def config_from_hebo_row(row, space=None, base=None):
    space = space or SPACE
    base = base or DEFAULT_CONFIG
    cand = dict(base)
    for k, d in space.items():
        v = _hebo_scalar(row, k)
        if d['type'] == 'bool':
            cand[k] = bool(round(v))
        elif d['type'] in ('int_choice', 'enum', 'choice'):
            cand[k] = d['values'][int(round(v))]
        elif d['type'] == 'int':
            cand[k] = int(round(v))
        else:
            cand[k] = v
    return validate(cand)


def pd_row_from_config(c, space=None):
    space = space or SPACE
    import pandas as pd
    d = {}
    for k, meta in space.items():
        v = c[k]
        if meta['type'] == 'bool':
            d[k] = 1 if v else 0
        elif meta['type'] in ('int_choice', 'enum', 'choice'):
            d[k] = meta['values'].index(v)
        else:
            d[k] = v
    return pd.DataFrame([d])


# Default protocol anchor config (no augmentation): reference for delta reports.
DEFAULT_CONFIG = {
    'learning_rate': 0.01,
    'hidden_dim': 64,
    'num_layers': 2,
    'out_channels': 48,
    'dropout': 0.2,
    'use_mixup': False,
    'use_mixup_phys': False,
    'use_mixup_temporal': False,
    'use_jitter': False,
    'sigma_mixup': 0.2,
    'sigma_phys': 0.2,
    'sigma_temporal': 0.2,
    'sigma_jitter': 0.05,
    'augmentation_factor': 1.0,
}

# Sensible hand-picked augmentation (group e "manual") with default model HP:
# mixup sigma 0.5 x factor 2 (strong label-consistent interpolation).
MANUAL_CONFIG = dict(DEFAULT_CONFIG)
MANUAL_CONFIG.update(use_mixup=True, sigma_mixup=0.5, augmentation_factor=2.0)

# Wide search space kept for reference (early exploratory runs).
LARGE_SPACE = dict(SPACE)

SPACES = {'small': SPACE, 'large': LARGE_SPACE,
          'model': MODEL_ONLY_SPACE, 'aug': AUG_ONLY_SPACE}
