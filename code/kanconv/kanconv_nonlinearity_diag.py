"""Nonlinearity diagnostic across candidate datasets (phenomenon analysis).

Rationale: KANConv's per-channel B-spline calibration should pay off where
the GHI -> power mapping is strongly nonlinear (mid/high GHI saturation).
On datasets with an almost-linear GHI->power relation the spline has nothing
to learn (SQ1/UG7 showed ~0% gain). This diagnostic quantifies, per dataset,
how nonlinear the irradiance->power relationship is:

  1. GHI->power curve nonlinearity (fit linear vs quadratic vs cubic on the
     normalized relationship; report the R^2 improvement of the cubic over
     the linear fit - the "nonlinear information content").
  2. Local curvature: mean absolute second derivative of the smoothed
     GHI->power curve (data-driven nonlinearity).
  3. Saturation index: share of samples in the high-GHI regime where power
     growth flattens (GHI above the 90th percentile vs power below the 75th).

All metrics are computed on the normalized (MinMax -1..1) scale, same as the
model input. Output: s_data/cleaned/kanconv_nonlinearity_diag.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

RAW7 = ['GHI', 'DHI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']
SITE6_LD = ['GHI', 'Temperature', 'Wind_Speed', 'WD_sin', 'WD_cos', 'Air_Pressure']
SITE6_YU = ['GHI', 'Temperature', 'Wind_Speed', 'WD_sin', 'WD_cos', 'Air_Pressure']
ASF7 = ['GHI', 'DHI', 'DNI', 'Temperature', 'Humidity', 'Wind_Speed', 'WD_sin', 'WD_cos']

CANDIDATES = {
    '56':   ('s_data/cleaned/site56_2016_15min.csv', RAW7),
    '63':   ('s_data/cleaned/site63_2016_15min.csv', RAW7),
    '64':   ('s_data/cleaned/site64_2016_15min.csv', RAW7),
    '64_14':('s_data/cleaned/site64_2014_15min.csv', RAW7),
    '67':   ('s_data/cleaned/site67_2016_15min.csv', RAW7),
    '73':   ('s_data/cleaned/site73_2016_15min.csv', RAW7),
    '78':   ('s_data/cleaned/site78_2016_15min.csv', RAW7),
    '79':   ('s_data/cleaned/site79_2016_15min.csv', RAW7),
    '85':   ('s_data/cleaned/site85_2016_15min.csv', RAW7),
    '92':   ('s_data/cleaned/site92_2016_15min.csv', RAW7),
    '11':   ('s_data/cleaned/site11_2016_15min.csv', SITE6_LD),
    '212':  ('s_data/cleaned/site212_2016_15min.csv', RAW7),
    '213':  ('s_data/cleaned/site213_2016_15min.csv', RAW7),
    'yulara17': ('s_data/cleaned/yulara_LDPV1_2017_15min.csv', SITE6_YU),
    'yulara21A': ('s_data/cleaned/yulara_sub8_5_2021_15min.csv', SITE6_YU),
    'yulara21B': ('s_data/cleaned/yulara_sub9_3B_2021_15min.csv', SITE6_YU),
    'asf_cocoa': ('s_data/cleaned/asf_cocoa_2019_hourly.csv', ASF7),
    'asf_golden': ('s_data/cleaned/asf_golden_2020_hourly.csv', ASF7),
}

# all preprocessed HKUST stations
for _f in sorted(os.listdir('s_data/cleaned/hkust')):
    if _f.startswith('hkust_') and _f.endswith('.csv'):
        _k = _f[len('hkust_'):-len('.csv')]
        CANDIDATES[f'HK_{_k}'] = (os.path.join('s_data/cleaned/hkust', _f), SITE6_LD)


def ghi_power_nonlinearity(g, p, rng):
    """Metrics on the normalized GHI->power relationship."""
    g = np.clip(g, rng[0], rng[1])
    p = np.clip(p, rng[0], rng[1])
    m = (g > -0.9) & (p > -0.9)          # ignore night/dead samples
    if m.sum() < 500:
        return None
    g, p = g[m], p[m]
    # 1. polynomial fit improvement: cubic vs linear R^2 gain
    def r2(y, yhat):
        ss = np.sum((y - y.mean()) ** 2)
        return 1 - np.sum((y - yhat) ** 2) / (ss + 1e-12)
    c1 = np.polyfit(g, p, 1)
    c3 = np.polyfit(g, p, 3)
    r2_lin = r2(p, np.polyval(c1, g))
    r2_cub = r2(p, np.polyval(c3, g))
    nonlin_poly = (r2_cub - r2_lin) / max(r2_lin, 1e-6)
    # 2. local curvature of the smoothed curve (second derivative magnitude)
    edges = np.linspace(g.min(), g.max(), 41)
    idx = np.clip(np.digitize(g, edges) - 1, 0, len(edges) - 2)
    mean_p = np.array([p[idx == i].mean() for i in range(len(edges) - 1)])
    mean_g = np.array([g[idx == i].mean() for i in range(len(edges) - 1)])
    ok = np.isfinite(mean_p) & (mean_p > -0.9)
    if ok.sum() < 5:
        return None
    mg, mp = mean_g[ok], mean_p[ok]
    d2 = np.abs(np.gradient(np.gradient(mp, mg), mg))
    curvature = float(np.nanmean(d2))
    # 3. saturation index: share of high-GHI samples with flattened power
    hi = g > np.quantile(g, 0.9)
    sat = float((p[hi] < np.quantile(p, 0.75)).mean())
    return nonlin_poly, curvature, sat, float(r2_lin), int(m.sum())


def main():
    rows = []
    for name, (path, feats) in CANDIDATES.items():
        if not os.path.exists(path):
            print(f'  {name:10s} MISSING {path}')
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if 'GHI' not in df.columns or 'Active_Power' not in df.columns:
            print(f'  {name:10s} missing GHI/Active_Power columns')
            continue
        g = df['GHI'].values.astype(np.float64)
        p = df['Active_Power'].values.astype(np.float64)
        # normalize to -1..1 like the model input
        gmin, gmax = g.min(), g.max()
        pmin, pmax = p.min(), p.max()
        gn = 2 * (g - gmin) / (gmax - gmin) - 1
        pn = 2 * (p - pmin) / (pmax - pmin) - 1
        res = ghi_power_nonlinearity(gn, pn, (-1.0, 1.0))
        if res is None:
            print(f'  {name:10s} insufficient day samples')
            continue
        nonlin_poly, curv, sat, r2_lin, n = res
        rows.append({'site': name, 'n_samples': int(n),
                     'nonlin_poly_gain': nonlin_poly,
                     'curvature': curv, 'saturation_idx': sat,
                     'r2_linear': r2_lin})
        print(f'  {name:10s} n={n:7d} poly_gain={nonlin_poly:+.3f} '
              f'curv={curv:.4f} sat={sat:.2f} r2_lin={r2_lin:.3f}')
    df = pd.DataFrame(rows)
    df.to_csv('s_data/cleaned/kanconv_nonlinearity_diag.csv', index=False)
    print('\n===== ranked by nonlinearity (poly gain) =====')
    print(df.sort_values('nonlin_poly_gain', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
