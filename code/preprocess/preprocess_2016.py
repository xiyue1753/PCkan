"""Preprocess 5 DKASC sites for Phase 11 (2016 data).

Pipeline:
  1. Load raw CSV, filter 2016
  2. 15min resample (mean)
  3. Clean: clip negatives, clamp T/Hum, Wind_Direction % 360
  4. Minimal feature set (7 input cols + target): same 6 meteorological features as original paper
     - No derived features (hour_sin/cos, clear_sky_idx, PR, delta_t)
  5. Adaptive nighttime compression (time-gap based)
  6. delta_t = real minutes between kept rows
  7. Save to s_data/cleaned/

Note: Site 214 has Wind_Speed=0 for all years. Sites 56/63/64/73 have real wind data in 2016.
"""
import pandas as pd, numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))

SITES = {
    '214': 's_data/214-Site_DKA-M18_B-Phase_II.csv',
    '56':  's_data/56-Site_DKA-M20_A-Phase.csv',
    '63':  's_data/63-Site_DKA-M17_A-Phase.csv',
    '64':  's_data/64-Site_DKA-M17_B-Phase.csv',
    '73':  's_data/73-Site_DKA-M19_A-Phase.csv',
}
YEAR = 2016
POWER_THRESHOLD = 0.001
MIN_GAP_STEPS = 8
MIN_NIGHT_SEGMENT = 4
EDGE_KEEP = 2

KEEP_COLS = ['Active_Power', 'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation',
             'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Wind_Speed',
             'Wind_Direction', 'Performance_Ratio']

OUTPUT_DIR = Path('s_data/cleaned')
OUTPUT_DIR.mkdir(exist_ok=True)


for site_id, raw_path in SITES.items():
    print(f"\n{'='*60}")
    print(f"  Site {site_id}: {raw_path}")
    print(f"{'='*60}")

    print(f"  [1/6] Loading...")
    use_cols_fn = lambda c: c == 'timestamp' or c in KEEP_COLS
    df = pd.read_csv(raw_path, usecols=use_cols_fn,
                     parse_dates=['timestamp'], dayfirst=False,
                     on_bad_lines='skip', engine='python')
    print(f"    Raw: {len(df):,} rows")

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=False, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df[(df['timestamp'].dt.year == YEAR)].copy()
    print(f"    {YEAR} filtered: {len(df):,} rows")

    for c in KEEP_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    print(f"    Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    print(f"  [2/6] Resampling 5min -> 15min...")
    df = df.set_index('timestamp').sort_index()
    full_idx_5min = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min')
    df = df.reindex(full_idx_5min)
    df = df.interpolate(method='linear', limit=3).ffill().bfill()
    df_15min = df.resample('15min').mean()
    print(f"    After 15min resample: {len(df_15min):,} rows")

    print(f"  [3/6] Cleaning...")
    df_15min['Active_Power'] = df_15min['Active_Power'].clip(lower=0)
    df_15min['Global_Horizontal_Radiation'] = df_15min['Global_Horizontal_Radiation'].clip(lower=0)
    df_15min.loc[df_15min['Global_Horizontal_Radiation'] < 5, 'Global_Horizontal_Radiation'] = 0
    df_15min['Diffuse_Horizontal_Radiation'] = df_15min['Diffuse_Horizontal_Radiation'].clip(lower=0)
    if 'Weather_Temperature_Celsius' in df_15min.columns:
        df_15min['Weather_Temperature_Celsius'] = df_15min['Weather_Temperature_Celsius'].clip(lower=-5, upper=50)
    if 'Weather_Relative_Humidity' in df_15min.columns:
        df_15min['Weather_Relative_Humidity'] = df_15min['Weather_Relative_Humidity'].clip(lower=0, upper=100)
    if 'Wind_Speed' in df_15min.columns:
        df_15min['Wind_Speed'] = df_15min['Wind_Speed'].fillna(0).clip(lower=0)
    if 'Wind_Direction' in df_15min.columns:
        wd = df_15min['Wind_Direction'].fillna(0)
        wd = np.where(np.abs(wd) > 360 * 100, 0, wd)
        df_15min['Wind_Direction'] = wd % 360
    if 'Performance_Ratio' in df_15min.columns:
        df_15min['Performance_Ratio'] = df_15min['Performance_Ratio'].fillna(0).clip(lower=0)

    print(f"  [4/6] Feature engineering (6-met + target only)...")
    df_15min['GHI'] = df_15min['Global_Horizontal_Radiation']
    df_15min['DHI'] = df_15min['Diffuse_Horizontal_Radiation']
    df_15min['Temperature'] = df_15min['Weather_Temperature_Celsius'].fillna(0)
    df_15min['Humidity'] = df_15min['Weather_Relative_Humidity'].fillna(0)

    wd_rad = np.deg2rad(df_15min['Wind_Direction'].fillna(0))
    df_15min['WD_sin'] = np.sin(wd_rad)
    df_15min['WD_cos'] = np.cos(wd_rad)

    wind_max = df_15min['Wind_Speed'].max()
    df_15min['has_wind'] = 0 if wind_max > 0.01 else 1
    print(f"    Wind_Speed max={wind_max:.2f} -> has_wind={df_15min['has_wind'].iloc[0]}")

    print(f"  [5/6] Adaptive nighttime compression...")
    is_night = df_15min['Active_Power'] < POWER_THRESHOLD
    keep = np.ones(len(df_15min), dtype=bool)
    start = None
    for i in range(len(df_15min)):
        if is_night.iloc[i] and start is None:
            start = i
        elif (not is_night.iloc[i]) and start is not None:
            seg_len = i - start
            if seg_len > MIN_NIGHT_SEGMENT:
                interior_start = start + EDGE_KEEP
                interior_end = i - EDGE_KEEP
                if interior_end > interior_start:
                    for j in range(interior_start, interior_end):
                        if (j - interior_start) % MIN_GAP_STEPS != 0:
                            keep[j] = False
            start = None
    if start is not None:
        seg_len = len(df_15min) - start
        if seg_len > MIN_NIGHT_SEGMENT:
            interior_start = start + EDGE_KEEP
            interior_end = len(df_15min) - EDGE_KEEP
            if interior_end > interior_start:
                for j in range(interior_start, interior_end):
                    if (j - interior_start) % MIN_GAP_STEPS != 0:
                        keep[j] = False

    df_compressed = df_15min.iloc[keep].copy()
    before = len(df_15min)
    after = len(df_compressed)
    print(f"    Compressed: {before:,} -> {after:,} rows ({after/before*100:.0f}%)")

    df_compressed['delta_t'] = df_compressed.index.to_series().diff().dt.total_seconds() / 60
    df_compressed.loc[df_compressed.index[0], 'delta_t'] = 15.0

    print(f"  [6/6] Saving...")
    FINAL_COLS = ['Active_Power', 'GHI', 'DHI', 'Temperature', 'Humidity',
                  'Wind_Speed', 'WD_sin', 'WD_cos', 'has_wind', 'delta_t']
    df_out = df_compressed[FINAL_COLS]
    output_path = OUTPUT_DIR / f'site{site_id}_2016_15min.csv'
    df_out.to_csv(output_path)
    day_samples = (df_out['Active_Power'] > 0.01).sum()
    print(f"    Saved: {output_path}")
    print(f"    Shape: {df_out.shape}, nulls: {df_out.isnull().sum().sum()}")
    print(f"    Day samples: {day_samples}/{len(df_out)} ({day_samples/len(df_out)*100:.0f}%)")
    print(f"    delta_t: mean={df_out['delta_t'].mean():.1f} min, max={df_out['delta_t'].max():.0f} min")

print(f"\n{'='*60}")
print(f"  PREPROCESSING DONE: 5 sites x 2016 x 15min x adaptive compression")
print(f"  Output: s_data/cleaned/site*_2016_15min.csv")
print(f"{'='*60}")
