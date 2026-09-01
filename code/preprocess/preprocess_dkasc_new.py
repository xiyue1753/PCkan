"""Preprocess new DKASC sites 79/92/213/11 for 2016 (same pipeline as
preprocess_2016.py, extended).

Protocol differences:
  - 79, 92, 11: standard adaptive night compression (MIN_GAP_STEPS=8)
  - 213: half-year data -> LIGHT compression (MIN_GAP_STEPS=4) per user
  - site 11: 2016 per suggestion (2017 fallback if results poor)
All: 5min -> 15min resample, same cleaning as site56/63/73.

Output: s_data/cleaned/site{79,92,213,11}_2016_15min.csv
"""
import pandas as pd, numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))

SITES = {
    '79':  ('s_data/79-Site_DKA-M6_A-Phase.csv', 8),
    '92':  ('s_data/92-Site_DKA-M6_B-Phase.csv', 8),
    '213': ('s_data/213-Site_DKA-M16_A-Phase_II.csv', 4),
    '11':  ('s_data/11-Site_LD-PV1-DB-LD-1A.csv', 8),
    '78':  ('s_data/78-Site_DKA-M11_3-Phase.csv', 8),
    '67':  ('s_data/67-Site_DKA-M8_A-Phase.csv', 8),
    '85':  ('s_data/85-Site_DKA-M7_A-Phase.csv', 8),
    '212': ('s_data/212-Site_DKA-M15_C-Phase_II.csv', 4),
}
YEAR = 2016
POWER_THRESHOLD = 0.001
MIN_NIGHT_SEGMENT = 4
EDGE_KEEP = 2

KEEP_COLS = ['Active_Power', 'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation',
             'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Wind_Speed',
             'Wind_Direction', 'Performance_Ratio', 'Air_Pressure']

OUTPUT_DIR = Path('s_data/cleaned')
OUTPUT_DIR.mkdir(exist_ok=True)


def compress_night(df, min_gap_steps):
    is_night = df['Active_Power'] < POWER_THRESHOLD
    keep = np.ones(len(df), dtype=bool)
    start = None
    for i in range(len(df)):
        if is_night.iloc[i] and start is None:
            start = i
        elif (not is_night.iloc[i]) and start is not None:
            seg_len = i - start
            if seg_len > MIN_NIGHT_SEGMENT:
                interior_start = start + EDGE_KEEP
                interior_end = i - EDGE_KEEP
                if interior_end > interior_start:
                    for j in range(interior_start, interior_end):
                        if (j - interior_start) % min_gap_steps != 0:
                            keep[j] = False
            start = None
    if start is not None:
        seg_len = len(df) - start
        if seg_len > MIN_NIGHT_SEGMENT:
            interior_start = start + EDGE_KEEP
            interior_end = len(df) - EDGE_KEEP
            if interior_end > interior_start:
                for j in range(interior_start, interior_end):
                    if (j - interior_start) % min_gap_steps != 0:
                        keep[j] = False
    return df.iloc[keep].copy()


for site_id, (raw_path, gap) in SITES.items():
    print(f"\n{'='*60}\n  Site {site_id}: {raw_path} (gap={gap})\n{'='*60}")
    df = pd.read_csv(raw_path, usecols=lambda c: c == 'timestamp' or c in KEEP_COLS,
                     parse_dates=['timestamp'], dayfirst=False,
                     on_bad_lines='skip', engine='python')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=False, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df[(df['timestamp'].dt.year == YEAR)].copy()
    print(f'  {YEAR} rows: {len(df):,}  range: {df["timestamp"].min()} -> {df["timestamp"].max()}')
    for c in KEEP_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.set_index('timestamp').sort_index()
    full_idx_5min = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min')
    df = df.reindex(full_idx_5min)
    df = df.interpolate(method='linear', limit=3).ffill().bfill()
    df_15min = df.resample('15min').mean()

    df_15min['Active_Power'] = df_15min['Active_Power'].clip(lower=0)
    df_15min['Global_Horizontal_Radiation'] = df_15min['Global_Horizontal_Radiation'].clip(lower=0)
    df_15min.loc[df_15min['Global_Horizontal_Radiation'] < 5, 'Global_Horizontal_Radiation'] = 0
    if 'Diffuse_Horizontal_Radiation' in df_15min.columns:
        df_15min['Diffuse_Horizontal_Radiation'] = df_15min['Diffuse_Horizontal_Radiation'].clip(lower=0)
    else:
        df_15min['Diffuse_Horizontal_Radiation'] = 0.0
    if 'Weather_Temperature_Celsius' in df_15min.columns:
        df_15min['Weather_Temperature_Celsius'] = df_15min['Weather_Temperature_Celsius'].clip(lower=-5, upper=50)
    if 'Weather_Relative_Humidity' in df_15min.columns:
        df_15min['Weather_Relative_Humidity'] = df_15min['Weather_Relative_Humidity'].clip(lower=0, upper=100)
    if 'Wind_Speed' in df_15min.columns:
        df_15min['Wind_Speed'] = df_15min['Wind_Speed'].fillna(0).clip(lower=0)
    if 'Wind_Direction' in df_15min.columns:
        wd = df_15min['Wind_Direction'].fillna(0)
        df_15min['Wind_Direction'] = np.where(np.abs(wd) > 360 * 100, 0, wd) % 360

    df_15min['GHI'] = df_15min['Global_Horizontal_Radiation']
    df_15min['DHI'] = df_15min['Diffuse_Horizontal_Radiation']
    df_15min['Temperature'] = df_15min['Weather_Temperature_Celsius'].fillna(0)
    if 'Weather_Relative_Humidity' in df_15min.columns:
        df_15min['Humidity'] = df_15min['Weather_Relative_Humidity'].fillna(0)
    else:
        df_15min['Humidity'] = 0.0
    if 'Air_Pressure' in df_15min.columns:
        df_15min['Air_Pressure'] = df_15min['Air_Pressure'].fillna(0)
    wd_rad = np.deg2rad(df_15min['Wind_Direction'].fillna(0))
    df_15min['WD_sin'] = np.sin(wd_rad)
    df_15min['WD_cos'] = np.cos(wd_rad)
    df_15min['has_wind'] = 1 if df_15min['Wind_Speed'].max() > 0.01 else 0

    df_compressed = compress_night(df_15min, gap)
    before = len(df_15min)
    after = len(df_compressed)
    print(f'  compressed: {before:,} -> {after:,} rows ({after/before*100:.0f}%), '
          f'wind_max={df_15min["Wind_Speed"].max():.2f}, day={(df_compressed["Active_Power"]>0.01).sum():,}')

    FINAL_COLS = ['Active_Power', 'GHI', 'DHI', 'Temperature', 'Humidity',
                  'Wind_Speed', 'WD_sin', 'WD_cos', 'has_wind', 'delta_t']
    if 'Air_Pressure' in df_15min.columns:
        FINAL_COLS.append('Air_Pressure')
    df_compressed['delta_t'] = df_compressed.index.to_series().diff().dt.total_seconds() / 60
    df_compressed.loc[df_compressed.index[0], 'delta_t'] = 15.0
    out = df_compressed[FINAL_COLS]
    out.index.name = 'timestamp'
    path = OUTPUT_DIR / f'site{site_id}_{YEAR}_15min.csv'
    out.to_csv(path)
    print(f'  saved: {path}')

print('\n  DONE: 79/92/213/11 preprocessed for 2016')
