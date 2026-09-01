import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
HISTORY_DIR = BASE_DIR / 'history'
MODEL_DIR = BASE_DIR / 'model'

MASTER_SEED = 42
EXPERIMENT_SEEDS = [42, 123, 456, 789, 2026]

SPLIT_RATIOS = (0.64, 0.16, 0.20)  # train / val / test
LOOKBACK = 20
TARGET_COL = 'profit'

PRICE_SCHEDULE = {
    0: 0.15, 1: 0.12, 2: 0.09, 3: 0.08, 4: 0.08, 5: 0.08,
    6: 0.10, 7: 0.15, 8: 0.20, 9: 0.22, 10: 0.25, 11: 0.30,
    12: 0.28, 13: 0.25, 14: 0.22, 15: 0.20, 16: 0.20, 17: 0.22,
    18: 0.25, 19: 0.24, 20: 0.22, 21: 0.20, 22: 0.18, 23: 0.15
}

CONSUMPTION_SCHEDULE = [
    ((0, 5), 5),
    ((5, 6), 6),
    ((6, 7), 10),
    ((7, 8), 15),
    ((8, 9), 20),
    ((9, 10), 25),
    ((10, 17), 30),
    ((17, 18), 20),
    ((18, 19), 15),
    ((19, 20), 10),
    ((20, 21), 8),
    ((21, 22), 6),
    ((22, 24), 5),
]

ENERGY_CONVERSION = 5 / 60  # 5-min sampling interval → kWh
I_BASE = 0  # base revenue (explicitly zero for this study)

LLM_BACKEND = 'deepseek'  # 'deepseek' | 'gpt' | 'glm'

# ---------------------------------------------------------------
# DeepSeek API interface — key is NEVER hardcoded here.
# Fill in one of the following (in priority order):
#   1. environment variable  DEEPSEEK_API_KEY
#   2. file  cnn_lstm_kan/.deepseek_api_key  (plain text, first line)
#      (this file is gitignored — safe to write the key locally)
#   3. file  cnn_lstm_kan/local_secrets.py  (DEEPSEEK_API_KEY = 'sk-...')
# The file is read once and cached; no key ever gets logged or committed.
# ---------------------------------------------------------------
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1'
DEEPSEEK_MODEL = 'deepseek-v4-flash'   # V4-Flash-0731 (docs: api.deepseek.com)
LLM_TEMPERATURE = 0.1      # main experiments; E8 sweeps this
LLM_MAX_TOKENS = 8192      # thinking tokens + JSON config output
DEEPSEEK_THINKING = True   # v4 thinking mode (default); E8 semantics noted
LLM_REASONING_EFFORT = 'low'   # 'low' | 'medium' | 'high' (v4 thinking)

_api_key_cache = None


def get_deepseek_api_key():
    """Return the DeepSeek API key from env var / local file, or None.

    Priority: DEEPSEEK_API_KEY env var > .deepseek_api_key file
    > local_secrets.py. The key is cached after first resolution.
    """
    global _api_key_cache
    if _api_key_cache is not None:
        return _api_key_cache

    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if key:
        _api_key_cache = key
        return key

    key_file = Path(__file__).parent / '.deepseek_api_key'
    if key_file.exists():
        for line in key_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                _api_key_cache = line
                return line

    secrets_file = Path(__file__).parent / 'local_secrets.py'
    if secrets_file.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location('local_secrets', secrets_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        key = getattr(mod, 'DEEPSEEK_API_KEY', '').strip()
        if key:
            _api_key_cache = key
            return key

    return None

HPO_RANGES = {
    'model': {
        'learning_rate': [1e-5, 0.15],
        'hidden_dim': [32, 48, 64, 80, 96, 128],
        'num_layers': [1, 3],
        'num_epochs': [25, 100],
        'out_channels': [16, 32, 64, 80, 96, 128],
    },
    'ga': {
        'population_size': [2, 15],
        'mutation_rate': [0.1, 0.5],
        'crossover_rate': [0.6, 0.9],
        'generations': [30, 45],
    },
    'augmentation': {
        'enable_augmentation': [True, False],
        'strategy_type': ['none', 'jitter', 'scaling', 'time_warp', 'magnitude_warp'],
        'sigma_range': [0.01, 0.3],
        'knot_range': [3, 8],
        'augmentation_factor': [1.0, 3.0],
    }
}

DEFAULT_HYPERPARAMS = {
    'learning_rate': 0.01,
    'hidden_dim': 64,
    'num_layers': 2,
    'out_channels': 64,
    'num_epochs': 100,
}

DEFAULT_GA_PARAMS = {
    'population_size': 8,
    'mutation_rate': 0.25,
    'crossover_rate': 0.8,
    'generations': 40,
}

DEFAULT_AUGMENTATION = {
    'enable_augmentation': True,
    'strategy_type': '混合策略：jitter + scaling',
    'methods': [
        {'method': 'jitter', 'params': {'sigma': 0.05}},
        {'method': 'scaling', 'params': {'sigma': 0.1}},
    ],
    'augmentation_factor': 1.8,
}

SEQLEN = 72
HORIZONS = [36, 72]  # 3h, 6h at 5-min resolution
HORIZON_NAMES = {36: '3h', 72: '6h'}

# Phase 10: 15-min resolution
P10_SEQLEN = 24
P10_HORIZONS = [12, 24]  # 3h, 6h at 15-min resolution
P10_HORIZON_NAMES = {12: '3h', 24: '6h'}

# v2 — Feature engineered with nighttime compression
FEATURE_COLS_V2 = [
    'Active_Power', 'GHI', 'DHI', 'Temperature', 'Humidity', 'Wind_Speed',
    'WD_sin', 'WD_cos', 'hour_sin', 'hour_cos',
    'clear_sky_idx', 'PR', 'has_wind', 'delta_t',
]
TARGET_COL = 'Active_Power'

CLEANED_SITES = {
    'site214': 's_data/cleaned/site214_201701.csv',
    'site56':  's_data/cleaned/site56_201701.csv',
    'site63':  's_data/cleaned/site63_201701.csv',
    'site64':  's_data/cleaned/site64_201701.csv',
    'site73':  's_data/cleaned/site73_201701.csv',
}

# Phase 10: 15-min adaptive compressed, full 2017 year
P10_SITES = {
    '214': 's_data/cleaned/site214_2017_15min_adaptive.csv',
    '56':  's_data/cleaned/site56_2017_15min_adaptive.csv',
    '63':  's_data/cleaned/site63_2017_15min_adaptive.csv',
    # '64', '73' to be added when expanding to 5 sites
}

# Phase 11: 2016 data, 15-min adaptive compressed, 7 input cols (no derived features)
P11_SEQLEN = 24
P11_HORIZONS = [12, 24]
P11_HORIZON_NAMES = {12: '3h', 24: '6h'}

P11_FEATURE_COLS = [
    'Active_Power', 'GHI', 'DHI', 'Temperature', 'Humidity',
    'Wind_Speed', 'WD_sin', 'WD_cos',
]

# P11B: with delta_t for long-horizon time-alignment test
P11B_FEATURE_COLS = [
    'GHI', 'DHI', 'Temperature', 'Humidity',
    'Wind_Speed', 'WD_sin', 'WD_cos', 'delta_t',
]

P11_SITES = {
    '214': 's_data/cleaned/site214_2016_15min.csv',
    '56':  's_data/cleaned/site56_2016_15min.csv',
    '63':  's_data/cleaned/site63_2016_15min.csv',
    '64':  's_data/cleaned/site64_2014_15min.csv',
    '73':  's_data/cleaned/site73_2016_15min.csv',
}

# Uncompressed versions (no nighttime compression, uniform 15-min grid)
P11_SITES_UNCOMP = {
    '56': 's_data/cleaned/site56_2016_15min_uncomp.csv',
    '63': 's_data/cleaned/site63_2016_15min_uncomp.csv',
    '73': 's_data/cleaned/site73_2016_15min_uncomp.csv',
}

# Phase 12: Yulara (DKASC Yulara, NT). 6 features, no DHI/Humidity available.
# Air_Pressure added as 6th met feature. Same 15min adaptive compression.
P12_SEQLEN = 24
P12_HORIZONS = [12, 24]
P12_HORIZON_NAMES = {12: '3h', 24: '6h'}

P12_FEATURE_COLS = [
    'Active_Power', 'GHI', 'Temperature', 'Wind_Speed',
    'WD_sin', 'WD_cos', 'Air_Pressure',
]

P12_SITES = {
    'LDPV1_2017': 's_data/cleaned/yulara_LDPV1_2017_15min.csv',
    'sub8_5_2021': 's_data/cleaned/yulara_sub8_5_2021_15min.csv',
    'sub9_3B_2021': 's_data/cleaned/yulara_sub9_3B_2021_15min.csv',
}

# Phase 13: ASF (Adaptive Solar Forecasting, US). Hourly, 24h in / 12h out.
# Henderson dropped (insufficient contiguous hourly coverage for 24h windows).
# Weather time-zone corrected (-5h cocoa, -7h golden); negatives clipped to 0.
P13_SEQLEN = 24
P13_HORIZONS = [12]
P13_HORIZON_NAMES = {12: '12h'}

P13_FEATURE_COLS = [
    'Active_Power', 'GHI', 'DHI', 'DNI', 'Temperature', 'Humidity',
    'Wind_Speed', 'WD_sin', 'WD_cos',
]

P13_SITES = {
    'ASF_Cocoa': 's_data/cleaned/asf_cocoa_2019_hourly.csv',
    'ASF_Golden': 's_data/cleaned/asf_golden_2020_hourly.csv',
}

ARCH_VARIANTS = {
    'V0':  {'dilated': False, 'highway': False, 'lw_seq': False, 'wide_kan': False},
    'V1a': {'dilated': True,  'highway': False, 'lw_seq': False, 'wide_kan': False},
    'V1b': {'dilated': False, 'highway': True,  'lw_seq': False, 'wide_kan': False},
    'V1c': {'dilated': False, 'highway': False, 'lw_seq': True,  'wide_kan': False},
    'V1d': {'dilated': False, 'highway': False, 'lw_seq': False, 'wide_kan': True},
}

GA_FITNESS_VARIANTS = ['baseline', 'F1', 'F2', 'F3', 'F4', 'F5']
GA_FITNESS_K = 5
GA_SMALL_POP = 5
GA_SMALL_GEN = 15
GA_LAMBDA_GRAD = 1.0
GA_LAMBDA_CORR = 0.3

try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'

MODEL_NAMES = ['LSTM', 'LSTM_KAN', 'CNN_LSTM', 'CNN_LSTM_MLP', 'CNN_LSTM_KAN', 'GA_CNN_LSTM_KAN']
