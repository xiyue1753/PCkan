"""Preprocess HKUST PV + weather data for KAN experiments.

  - sites: ALL available PV stations (40+)
  - 1 year: 2021-06-01 ~ 2022-05-31
  - Weather: 1min -> 15min mean resample, merge with PV (15min)
  - Features: GHI(Irradiance), Temperature, Humidity, Wind_Speed, WD_sin, WD_cos (6 cols, no DHI)
  - Nighttime compression: same logic as DKASC (keep edges + every 8th interior point)
"""
import pandas as pd, numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))

BASE = Path('D:/PycharmProjects/cnn_lstm_kan/Dataset/Time series dataset')
OUT = PROJECT_ROOT / 's_data/cleaned/hkust'
OUT.mkdir(parents=True, exist_ok=True)

SITES = {
    'LSKN': 'LSK North.csv',
    'SQ1':  'SQ1.csv',
    'UG7':  'UG Hall7.csv',
}
# full site inventory (original HKUST dataset)
ALL_SITES = {
    'LSKN': 'LSK North.csv', 'LSKS': 'LSK South.csv',
    'SQ1': 'SQ1.csv', 'SQ2': 'SQ2.csv', 'SQ3': 'SQ3.csv', 'SQ4': 'SQ4.csv',
    'SQ567': 'SQ567.csv', 'SQ_P': 'SQ Block P.csv', 'SQ_R': 'SQ Block R.csv',
    'SQ_S': 'SQ Block S.csv',
    'SQ_A1_12': 'SQ Apartment1-12.csv', 'SQ_A13_24': 'SQ Apartment13-24.csv',
    'SQ_A25_36': 'SQ Apartment25-36.csv', 'SQ_A37_48': 'SQ Apartment37-48.csv',
    'UG2_2F': 'UG Hall2 2F.csv', 'UG2_RF': 'UG Hall2 RF.csv',
    'UG3': 'UG Hall3.csv', 'UG4': 'UG Hall4.csv', 'UG6': 'UG Hall6.csv',
    'UG7': 'UG Hall7.csv', 'UG8': 'UG Hall8.csv', 'UG9': 'UG Hall9.csv',
    'Indoor': 'Indoor Sports Centre.csv', 'Library': 'Library.csv',
    'SHHo': 'S H Ho Sports Hall.csv', 'Shaw': 'Shaw Auditorium.csv',
    'WongCS': 'Wong Check She Research Centre.csv',
    'Zone_A1': 'Zone A1.csv', 'Zone_A2': 'Zone A2.csv',
    'Zone_A4': 'Zone A4.csv', 'Zone_A5': 'Zone A5.csv',
    'Zone_A6': 'Zone A6.csv', 'Zone_A7': 'Zone A7.csv',
    'Zone_D': 'Zone D.csv', 'Zone_J1': 'Zone J1.csv',
    'Zone_J2': 'Zone J2.csv', 'Zone_L2': 'Zone L2.csv',
}
PV_DIR = BASE / 'PV generation dataset/PV stations with panel level optimizer/Site level dataset'
MET_DIR = BASE / 'Meteorological dataset'

YEAR_START = '2021-06-01'
YEAR_END = '2022-05-31'
POWER_THRESHOLD = 1.0   # W (HKUST small systems)
MIN_GAP_STEPS = 8
MIN_NIGHT_SEGMENT = 4
EDGE_KEEP = 2


def load_met(var_name, col_map, year):
    """Load a meteorological variable for a year, resample to 15min mean."""
    path = MET_DIR / f'{var_name}/{var_name}_{year}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time').sort_index()
    df = df.rename(columns=col_map)
    df = df.resample('15min').mean()
    return df


for site_key, site_file in ALL_SITES.items():
    print(f"\n{'='*60}")
    print(f"  HKUST site: {site_key} ({site_file})")
    print(f"{'='*60}")
    out_path = OUT / f'hkust_{site_key}.csv'
    if out_path.exists():
        print(f"  EXISTS, skip: {out_path}")
        continue
    try:
        # ─── 1. Load PV ───
        pv = pd.read_csv(PV_DIR / site_file)
        pv['Time'] = pd.to_datetime(pv['Time'])
        pv = pv.set_index('Time').sort_index()
        pv = pv.loc[YEAR_START:YEAR_END]
        if len(pv) < 5000:
            print(f"  SKIP {site_key}: only {len(pv)} rows in window "
                  f"({pv.index.min()}..{pv.index.max()})")
            continue
    except Exception as e:
        print(f"  SKIP {site_key}: load error {e}")
        continue

    # ─── 2. Load weather (need 2021 + 2022 for the year spanning Jun-May) ───
    met_frames = []
    for year in [2021, 2022]:
        ghi = load_met('Irradiance', {'Irradiance (W/m2)': 'GHI'}, year)
        temp = load_met('Temperature', {'Temp (Degree Celsius)': 'Temperature'}, year)
        rh = load_met('Relative Humidity', {'RH (%)': 'Humidity'}, year)
        wind = load_met('Wind', {}, year)
        if ghi is None or temp is None or rh is None or wind is None:
            print(f"  SKIP {year}: missing met data")
            continue
        met = pd.concat([ghi, temp, rh, wind], axis=1)
        met_frames.append(met)
    met = pd.concat(met_frames).sort_index()
    met = met.loc[YEAR_START:YEAR_END]
    # dedupe index (consecutive years may overlap at boundary)
    met = met[~met.index.duplicated(keep='first')]
    # Wind direction -> sin/cos
    met['WD_sin'] = np.sin(np.deg2rad(met['Wind Direction (degree)'].fillna(0)))
    met['WD_cos'] = np.cos(np.deg2rad(met['Wind Direction (degree)'].fillna(0)))
    met = met.rename(columns={'Wind Speed (m/s)': 'Wind_Speed'})
    print(f"  MET: {len(met)} rows")

    # ─── 3. Merge ───
    df = pv.join(met, how='left')
    df['Active_Power'] = df['power(W)'].clip(lower=0)
    print(f"  MERGED: {len(df)} rows, nulls: {df.isnull().sum().sum()}")

    # ─── 4. Clean ───
    df['GHI'] = df['GHI'].clip(lower=0)
    df['Temperature'] = df['Temperature'].clip(lower=-5, upper=50)
    df['Humidity'] = df['Humidity'].clip(lower=0, upper=100)
    df['Wind_Speed'] = df['Wind_Speed'].fillna(0).clip(lower=0)
    df['WD_sin'] = df['WD_sin'].fillna(0)
    df['WD_cos'] = df['WD_cos'].fillna(0)
    # fill remaining gaps
    df = df.interpolate(limit=3).ffill().bfill()

    # ─── 5. Nighttime compression (same as DKASC) ───
    is_night = df['Active_Power'] < POWER_THRESHOLD
    keep = np.ones(len(df), dtype=bool)
    start = None
    for i in range(len(df)):
        if is_night.iloc[i] and start is None:
            start = i
        elif (not is_night.iloc[i]) and start is not None:
            seg_len = i - start
            if seg_len > MIN_NIGHT_SEGMENT:
                i0 = start + EDGE_KEEP
                i1 = i - EDGE_KEEP
                if i1 > i0:
                    for j in range(i0, i1):
                        if (j - i0) % MIN_GAP_STEPS != 0:
                            keep[j] = False
            start = None
    if start is not None:
        seg_len = len(df) - start
        if seg_len > MIN_NIGHT_SEGMENT:
            i0 = start + EDGE_KEEP
            i1 = len(df) - EDGE_KEEP
            if i1 > i0:
                for j in range(i0, i1):
                    if (j - i0) % MIN_GAP_STEPS != 0:
                        keep[j] = False
    df_c = df.iloc[keep].copy()
    print(f"  Compressed: {len(df)} -> {len(df_c)} rows ({len(df_c)/len(df)*100:.0f}%)")

    # delta_t
    df_c['delta_t'] = df_c.index.to_series().diff().dt.total_seconds() / 60
    df_c.loc[df_c.index[0], 'delta_t'] = 15.0

    # ─── 6. Save ───
    COLS = ['Active_Power', 'GHI', 'Temperature', 'Humidity', 'Wind_Speed',
            'WD_sin', 'WD_cos', 'delta_t']
    df_out = df_c[COLS]
    path = OUT / f'hkust_{site_key}.csv'
    df_out.to_csv(path)
    day = (df_out['Active_Power'] > POWER_THRESHOLD).sum()
    print(f"  Saved: {path}")
    print(f"  Shape: {df_out.shape}, day%: {day/len(df_out)*100:.0f}%")
    print(f"  GHI max={df_out.GHI.max():.0f}, Power max={df_out.Active_Power.max():.0f}")

print(f"\n{'='*60}")
print(f"  HKUST PREPROCESSING DONE: 3 sites, 1 year (2021-06~2022-05)")
print(f"{'='*60}")
