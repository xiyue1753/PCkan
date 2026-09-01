"""Physics-consistent multi-channel augmentation for the joint HPO study.

Applies to normalized windows x: (B, T, C) in [-1, 1]; the target y is either
untouched (label-preserving strategies) or mixed consistently with x (mixup).

v3: per-strategy composition. The config carries one switch (use_*) and one
sigma per strategy; ANY subset of the four strategies may be enabled at once
(single strategy, several strategies mixed, or none). Enabled strategies are
applied in sequence per copy: mixup-family interpolation first (labels mixed
consistently), then jitter on the irradiance channels (labels untouched).

Why this family (replacing the generic noise family v1):
  - generic all-channel jitter/scaling breaks the GHI->power supervision the
    spline calibration is learning (noisy x with unchanged y = contradictory
    pairs) and corrupts angular features (WD_sin/WD_cos unit-circle).
  - the four strategies are physically consistent:
      * mixup          mixes (x, y) pairs together - label consistency is
                       guaranteed by construction; new samples lie on the
                       power manifold (standard for regression).
      * mixup_phys     same but pairs are restricted to the same GHI-state
                       bin (terciles of window-mean GHI): mixed samples keep
                       the GHI->power relationship physically plausible.
      * mixup_temporal same but pairs are restricted to the same hour-of-day
                       bin (peak-position proxy): keeps intraday phase.
      * jitter_ghi     perturbs ONLY the irradiance channels (GHI, channel 0;
                       +DHI when C==7), the drivers of PV power; leaves
                       angular/time features intact.

sigma semantics per strategy:
  - sigma_mixup / sigma_phys / sigma_temporal: Beta(a, a) shape (mixing
    strength; a=0.01 near-pure samples, a=0.5 smoother mixing)
  - sigma_jitter: Gaussian noise std on irradiance channels

Hard identity guarantees:
  - all sigma = 0  ->  augmented == input exactly
  - factor = 1     ->  dataset size unchanged
"""
import numpy as np
import torch

# Bump when the augmentation implementation changes so cached trials from an
# older implementation are automatically rejected (registry meta check).
AUG_VERSION = 3

VALID_STRATEGIES = ('none', 'mixup', 'mixup_phys', 'mixup_temporal',
                    'jitter_ghi', 'masking', 'mixup+jitter_ghi')

# GHI is always channel 0 in every site feature table; DHI exists only for
# the 7-feature DKASC protocol (RAW7). jitter_ghi perturbs these only.
_GHI_CH = 0


def _irradiance_channels(C):
    return (0, 1) if C == 7 else (0,)


def augment_batch(x, strategy, sigma, rng):
    """Label-preserving per-window strategies: x (B, T, C) -> (B, T, C).

    rng: torch.Generator for deterministic draws.
    """
    B, T, C = x.shape
    device = x.device

    if strategy in ('none',):
        return x.clone()
    if strategy == 'jitter_ghi':
        ch = _irradiance_channels(C)
        noise = torch.randn(B, T, len(ch), generator=rng, device=device) * sigma
        out = x.clone()
        out[:, :, ch] = (out[:, :, ch] + noise).clamp(-1.0, 1.0)
        return out
    raise ValueError(f'unknown label-preserving strategy: {strategy}')


def _same_bin_partner(x, rng, mode='ghi'):
    """Physics/time-consistent pairing: partners restricted to the same
    GHI-state bin (terciles of window-mean GHI, mode='ghi') or the same
    hour-of-day bin (mode='hour'). Mixing within a bin keeps the
    GHI -> power relationship / intraday phase physically plausible
    (random pairing creates impossible interpolations)."""
    B = x.shape[0]
    if mode == 'hour':
        # hour-of-day needs the absolute window start; approximate with a
        # feature: we do not have timestamps here, so bin by the GHI curve
        # shape proxy: time of peak = argmax of GHI within the window.
        g = x[:, :, 0]
        peak = g.argmax(dim=1).float()                     # (B,)
        edges = torch.linspace(0, x.shape[1], 7, device=x.device)  # 6 bins
        bins = torch.bucketize(peak, edges[1:-1])
    else:
        g = x[:, :, 0].mean(dim=1)                         # window-mean GHI
        q1, q2 = torch.quantile(g, torch.tensor([1 / 3.0, 2 / 3.0], device=x.device))
        bins = torch.bucketize(g, torch.tensor([q1, q2], device=x.device))
    partner = torch.empty(B, dtype=torch.long, device=x.device)
    n_bins = int(bins.max().item()) + 1
    for bv in range(n_bins):
        idx = torch.nonzero(bins == bv).flatten()
        if len(idx) > 1:
            perm = torch.randperm(len(idx), generator=rng)
            partner[idx] = idx[perm.roll(1)]               # rotate: no self-pairs
        else:
            partner[idx] = idx                             # singleton: self-copy ok
    return partner


def _mix_pair(x, y, sigma, rng, mode=None):
    """One label-consistent convex-combination round of (x, y) -> (xm, ym).

    w ~ Beta(a, a) drawn from the LOCAL generator (Beta.sample only uses
    the global RNG, so we sample two Gamma variates via the generator-aware
    _standard_gamma: w = X/(X+Y), X,Y ~ Gamma(a, a)).
    """
    B = x.shape[0]
    a = max(float(sigma), 1e-3)
    partner = _same_bin_partner(x, rng, mode) if mode else \
        torch.randperm(B, generator=rng)
    gx = torch._standard_gamma(torch.full((B,), a), generator=rng)
    gy = torch._standard_gamma(torch.full((B,), a), generator=rng)
    w = torch.nan_to_num(gx / (gx + gy), 0.5)     # (B,)
    w = w.unsqueeze(1).unsqueeze(2)               # (B, 1, 1)
    xm = (w * x + (1 - w) * x[partner]).clamp(-1.0, 1.0)
    ym = w[:, :, 0] * y + (1 - w[:, :, 0]) * y[partner]
    return xm, ym


def _enabled_strategies(c):
    """Ordered list of (strategy, sigma) from the config's use_* switches."""
    out = []
    if c.get('use_mixup'):
        out.append(('mixup', float(c.get('sigma_mixup', 0.2))))
    if c.get('use_mixup_phys'):
        out.append(('mixup_phys', float(c.get('sigma_phys', 0.2))))
    if c.get('use_mixup_temporal'):
        out.append(('mixup_temporal', float(c.get('sigma_temporal', 0.2))))
    if c.get('use_jitter'):
        out.append(('jitter_ghi', float(c.get('sigma_jitter', 0.05))))
    return out


def _apply_one(x, y, strategy, sigma, rng):
    if strategy == 'jitter_ghi':
        return augment_batch(x, 'jitter_ghi', sigma, rng), y.clone()
    mode = {'mixup': None, 'mixup_phys': 'ghi', 'mixup_temporal': 'hour'}[strategy]
    return _mix_pair(x, y, sigma, rng, mode)


def augment_dataset(x_train, y_train, aug_config, seed=0, aug_mode='append'):
    """Apply the config's augmentation to the whole train split.

    Returns (x_aug, y_aug). Each enabled strategy contributes to every
    augmented copy (strategies compose in sequence); label-preserving
    strategies replicate y. Returns the input unchanged when no strategy is
    enabled or factor rounds to 1.
    """
    c = aug_config or {}
    strategies = _enabled_strategies(c)
    if not strategies:
        return x_train, y_train

    x = x_train if torch.is_tensor(x_train) else torch.as_tensor(x_train)
    y = y_train if torch.is_tensor(y_train) else torch.as_tensor(y_train)
    is_cuda = x.is_cuda
    x_c = x.detach().cpu()
    y_c = y.detach().cpu()
    factor = float(c.get('augmentation_factor', 1.0))
    copies = int(round(factor)) - 1
    if copies <= 0 and aug_mode == 'append':
        return x_train, y_train

    B, T, C = x_c.shape
    rng = torch.Generator().manual_seed(seed)

    if aug_mode == 'replace':
        # dilution-free: augmented samples REPLACE originals, dataset size B
        # unchanged. factor controls the replaced fraction: kept = 1/factor.
        n_new = max(0, B - int(B / factor))
        xm, ym = x_c, y_c
        for strat, sig in strategies:
            xm, ym = _apply_one(xm, ym, strat, sig, rng)
        xs = [x_c[:B - n_new], xm[:n_new]]
        ys = [y_c[:B - n_new], ym[:n_new]]
    else:
        xs, ys = [x_c], [y_c]
        for _ in range(copies):
            xm, ym = x_c, y_c
            for strat, sig in strategies:
                xm, ym = _apply_one(xm, ym, strat, sig, rng)
            xs.append(xm)
            ys.append(ym)

    out_x = torch.cat(xs, dim=0)
    out_y = torch.cat(ys, dim=0)
    if is_cuda:
        out_x = out_x.cuda()
        out_y = out_y.cuda()
    return out_x, out_y
