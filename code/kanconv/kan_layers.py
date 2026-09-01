"""Custom KAN layers: pure B-spline, no SiLU base, identity init, fixed grid.

  - SplineAct: per-channel spline activation (diagonal, for CNN/LSTM use)
  - KANLayer_custom: full edge-function KAN layer (for output head use)
"""
import torch
import torch.nn as nn


def _compute_B_basis(x, grid, k):
    """Vectorized B-spline basis evaluation via Cox-de Boor recursion.

    Args:
        x: (batch, in_dim)
        grid: (in_dim, n_full) where n_full = G + 2k + 1
        k: spline order
    Returns:
        B: (batch, in_dim, G+k)  — B-spline basis values
    """
    batch, in_dim = x.shape
    x = x.unsqueeze(-1)                # (batch, in_dim, 1)
    g = grid.unsqueeze(0)              # (1, in_dim, n_full)
    n_full = g.shape[-1]

    # k=0: piecewise constant, n_full-1 basis functions
    B = (x >= g[:, :, :-1]) & (x < g[:, :, 1:])
    B = B.float()

    right_mask = (x == g[:, :, -1:])
    if right_mask.any():
        last = torch.zeros_like(B)
        last[:, :, -1:] = right_mask.float()
        B = B + last

    for deg in range(1, k + 1):
        left = B[:, :, :-1]
        right = B[:, :, 1:]

        t_i = g[:, :, :n_full - deg - 1]
        t_j = g[:, :, deg:n_full - 1]          # t_{i+deg}
        t_j1 = g[:, :, deg + 1:n_full]         # t_{i+deg+1}
        t_i1 = g[:, :, 1:n_full - deg]         # t_{i+1}

        w_left = (x - t_i) / (t_j - t_i + 1e-10)
        w_right = (t_j1 - x) / (t_j1 - t_i1 + 1e-10)

        B = w_left * left + w_right * right
        B = torch.nan_to_num(B, 0.0)

    return B   # (batch, in_dim, G+k)


class PerFeatureMLP(nn.Module):
    """M1 control: per-feature MLP replacing per-channel B-spline.

    out_i = x_i + MLP_i(x_i), MLP zero-init -> identity at start (matches
    SplineAct identity init). Same diagonal structure (no crosstalk),
    same input range [-1,1]. Tests whether the KAN gain comes from
    "learnable per-feature univariate transform" (then MLP ~= spline) or
    from B-spline's local/adaptive property (then MLP < spline).

    Params per feature: hidden*2 + hidden + 1. hidden=8 -> 25/feature.
    """
    def __init__(self, dim, hidden=8):
        super().__init__()
        self.dim = dim
        self.w1 = nn.Parameter(torch.randn(dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(dim, hidden))
        self.w2 = nn.Parameter(torch.zeros(dim, hidden))
        self.b2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if x.dim() == 3:
            B, C, T = x.shape
            x = x.permute(0, 2, 1).reshape(B * T, C)
            y = self._mlp(x)
            return y.reshape(B, T, C).permute(0, 2, 1)
        return self._mlp(x)

    def _mlp(self, x):
        h = torch.relu(x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0))  # (B, dim, hidden)
        out = (h * self.w2.unsqueeze(0)).sum(-1) + self.b2    # (B, dim)
        return x + out


class SplineAct(nn.Module):
    """Per-channel learnable B-spline activation (K-A idea as activation).

    y_i = spline_i(x_i) for each channel i. No crosstalk between channels.
    Identity-init: at init, spline_i(x) ≈ x (stable, linear start).

    Args:
        dim: number of channels (in_dim == out_dim, diagonal only)
        grid_size: G = number of grid intervals. Default 5.
        k: spline order. Default 3 (cubic).
        shared: True = ONE spline shared across all channels (SV-KAN style),
                False = per-channel spline (default).
        knots: optional (dim, G+1) or (G+1,) knot values in [-1,1] for a
               data-adaptive grid (quantile grid); None = uniform linspace.
        gate_init: if not None, learnable gate gamma on the spline deviation:
                   y = x + gamma * (spline(x) - x). Architecture-side
                   self-regularization: gamma can shrink toward 0 on
                   datasets where the identity calibration is sufficient.
    """
    def __init__(self, dim, grid_size=5, k=3, shared=False, knots=None, gate_init=None):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.k = k
        self.shared = shared

        h = 2.0 / grid_size
        n_full = grid_size + 1 + 2 * k
        if knots is None:
            grid_base = torch.linspace(-1, 1, grid_size + 1).unsqueeze(0).expand(dim, -1)
        elif knots.dim() == 1:
            grid_base = knots.float().unsqueeze(0).expand(dim, -1)
        else:
            grid_base = knots.float()                     # (dim, G+1)
        grid = torch.zeros(dim, n_full)
        grid[:, k:-k] = grid_base
        for i in range(k):
            grid[:, i] = grid_base[:, 0] - (k - i) * h
            grid[:, -(i + 1)] = grid_base[:, -1] + (k - i) * h
        self.register_buffer('grid', grid)

        n_basis = grid_size + k
        # identity init: solve B @ coef = x for grid interior points
        with torch.no_grad():
            if knots is None:
                n_pts = min(2 * grid_size, grid_size + 4)
                fit_pts = torch.linspace(-0.98, 0.98, n_pts)
            else:
                # fit points = base knots + interval midpoints (data-adapted)
                gp = grid_base[0]                          # channel-0 knots
                mids = (gp[:-1] + gp[1:]) / 2
                fit_pts = torch.sort(torch.cat([gp, mids]))[0]
            n_ch = 1 if shared else dim
            coef = torch.zeros(n_ch, n_basis)
            for ch in range(n_ch):
                x_pts = fit_pts.unsqueeze(1).expand(-1, 1)
                B_pts = _compute_B_basis(x_pts, grid[ch:ch + 1], k)   # (n_pts, 1, n_basis)
                B_ch = B_pts[:, 0, :]
                try:
                    sol = torch.linalg.lstsq(B_ch, fit_pts.unsqueeze(1)).solution.squeeze(1)
                except Exception:
                    sol = torch.zeros(n_basis)
                coef[ch, :] = sol
            if not shared:
                coef = coef.expand(dim, -1).clone()
        coef = coef + torch.randn_like(coef) * 1e-4         # symmetry breaking
        self.coef = nn.Parameter(coef)
        if gate_init is not None:
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.gate = None

    def forward(self, x):
        # Handle both (B, dim) and (B, dim, T) shape from Conv1d output
        if x.dim() == 3:
            B, C, T = x.shape
            x = x.permute(0, 2, 1).reshape(B * T, C)
            out = self._forward_2d(x)
            out = out.reshape(B, T, C).permute(0, 2, 1)     # (B, dim, T)
            return out
        return self._forward_2d(x)

    def _forward_2d(self, x):
        """2D forward with channel-blocked basis computation (memory-safe
        for wide-channel inputs, e.g. electricity 321ch / traffic 862ch).
        Same math as the full-basis version; per-block peak memory is
        O(B * block * n_basis)."""
        C = x.shape[1]
        block = getattr(self, '_block', 64)
        out = torch.empty_like(x)
        for c0 in range(0, C, block):
            c1 = min(c0 + block, C)
            xb = x[:, c0:c1]
            B_ = _compute_B_basis(xb, self.grid[c0:c1], self.k)   # (B, blk, n_basis)
            # shared=True -> one coef row broadcast over all blocks (SV-KAN style)
            coef_b = self.coef[:1] if self.shared else self.coef[c0:c1]
            yb = (B_ * coef_b.unsqueeze(0)).sum(dim=-1)
            if self.gate is not None:
                yb = xb + self.gate * (yb - xb)
            out[:, c0:c1] = yb
        return out


class KANLayer_custom(nn.Module):
    """Full edge-function KAN layer: y_j = Σ_i (spline_ij(x_i) + W_ij * x_i).

    Pure B-spline on edges, with linear shortcut for stable initialization.
    Zero spline init → behaves like Linear(x) at start. Learns nonlinear corrections.
    
    Args:
        in_dim: input dimension
        out_dim: output dimension
        grid_size: G. Default 5.
        k: spline order. Default 3.
    """
    def __init__(self, in_dim, out_dim, grid_size=5, k=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.k = k

        h = 2.0 / grid_size
        grid_base = torch.linspace(-1, 1, grid_size + 1)
        n_full = grid_size + 1 + 2 * k
        grid = torch.zeros(in_dim, n_full)
        grid[:, k:-k] = grid_base.unsqueeze(0).expand(in_dim, -1)
        for i in range(k):
            grid[:, i] = grid_base[0] - (k - i) * h
            grid[:, -(i + 1)] = grid_base[-1] + (k - i) * h
        self.register_buffer('grid', grid)

        n_basis = grid_size + k
        self.coef = nn.Parameter(torch.zeros(in_dim, out_dim, n_basis))
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        B = _compute_B_basis(x, self.grid, self.k)              # (B, in_dim, n_basis)
        y_spline = torch.einsum('bip,ijp->bj', B, self.coef)   # (B, out_dim)
        return y_spline + self.linear(x)


class KANConv1d(nn.Module):
    """KAN-based Conv1d: per-input-channel spline pre-transform + linear convolution.

    Applies learnable per-channel B-spline to inputs BEFORE the linear conv kernel,
    making the conv operation sensitive to input value distribution (KAN idea),
    while keeping ReLU sparsity intact.

    y_j(t) = Σ_i Σ_k w_{j,i,k} · spline_i(x_i(t+k))
    where spline_i is identity-init (starts ≈ linear, learns shape).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 grid_size=3, k=3, shared=False, knots=None, gate_init=None):
        super().__init__()
        self.spline = SplineAct(in_channels, grid_size=grid_size, k=k, shared=shared,
                                knots=knots, gate_init=gate_init)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        x_t = x.permute(0, 2, 1).reshape(B * T, C)
        x_t = self.spline(x_t).reshape(B, T, C).permute(0, 2, 1)
        return self.conv(x_t)


class MLPCalibConv1d(nn.Module):
    """R1#6 control: per-channel MLP calibration replacing the value-side B-spline.

    Same position/structure as KANConv1d (calibration before the linear conv),
    but the per-channel transform is a 1-hidden-layer MLP (hidden=8) instead
    of a B-spline. Zero-init residual (out = x + MLP(x)) -> exact identity at
    start, matching SplineAct's identity init. Same diagonal structure
    (no crosstalk), same input range [-1, 1].

    Tests "KAN is just swapping a layer": if a per-feature learnable
    univariate transform is enough, MLP ~= spline; if the gain requires
    B-spline properties (local support / C2 smoothness / identity fit),
    spline > MLP. 25 params/feature (hidden=8).

    y_j(t) = sum_i sum_k w_{j,i,k} * (x_i(t+k) + MLP_i(x_i(t+k)))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 hidden=8):
        super().__init__()
        self.calib = PerFeatureMLP(in_channels, hidden=hidden)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        x_t = x.permute(0, 2, 1).reshape(B * T, C)
        x_t = self.calib(x_t).reshape(B, T, C).permute(0, 2, 1)
        return self.conv(x_t)


class KernelSideKANConv1d(nn.Module):
    """Kernel-side KAN conv (ConvKAN family): per-edge splines on the kernel.

    y_j(t) = sum_{i,k} w_{j,i,k}(x_i(t+k))   with  w_{j,i,k}(z) = sum_p coef[j,i,k,p] * B_p(z)

    The B-spline basis of the input VALUE is computed pointwise, then a linear
    convolution over (C * n_basis) input channels implements the per-edge
    spline-weighted sum. Identity-init: coef[j,i,k,p] = w_lin[j,i,k] * coef_id[i,p],
    so at start y == the linear conv of the raw input (standard kernel-side
    KANs are random-init; ours is identity-init to match the fair protocol).

    Parameters: out * in * (G+k) * kernel = 48*7*6*3 = 6048 vs plain conv 1008.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 grid_size=3, k=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k = k
        self.n_basis = grid_size + k
        self.spline = SplineAct(in_channels, grid_size=grid_size, k=k)
        # identity target for the linear part: use the identity-fit spline coefs
        with torch.no_grad():
            conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
            coef = torch.zeros(out_channels, in_channels, self.n_basis, kernel_size)
            for i in range(in_channels):
                for kk in range(kernel_size):
                    coef[:, i, :, kk] = torch.outer(conv.weight[:, i, kk],
                                                    self.spline.coef[i])  # (out,) x (n_basis,)
            del conv
        self.coef = nn.Parameter(coef)

    def forward(self, x):
        B, C, T = x.shape
        xt = x.permute(0, 2, 1).reshape(B * T, C)
        Basis = _compute_B_basis(xt, self.spline.grid, self.k)          # (B*T, C, n_basis)
        Basis = Basis.reshape(B, T, C * self.n_basis).permute(0, 2, 1)  # (B, C*n_basis, T)
        w = self.coef.reshape(self.out_channels, C * self.n_basis, self.kernel_size)
        y = torch.nn.functional.conv1d(Basis, w, stride=self.stride, padding=self.padding)
        return y


class TemporalKANConv1d(nn.Module):
    """Temporal-axis KAN: spline on lagged differences Δx(t) = x(t) - x(t-lag).

    Per-channel B-spline applied to TEMPORAL DIFFERENCES (time evolution),
    orthogonal to ResKAN (output-axis residual). Differences capture
    time-scale-specific trends (cloud evolution, ramp rates) that raw
    feature values and fixed LSTM gates do not explicitly model.

    Design: for each input channel and each lag, compute Δx_lag(t),
    apply spline per (channel, lag) pair, concat with original channels,
    then linear conv.

    y_j(t) = Σ_i Σ_k w_{j,i,k} · [x_i(t+k) | φ_i,lag(x_i(t+k)-x_i(t+k-lag))]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 lags=(1, 3, 6, 12), grid_size=3, k=3):
        super().__init__()
        self.lags = tuple(lags)
        # one SplineAct per (channel, lag)
        self.splines = nn.ModuleList([
            SplineAct(in_channels, grid_size=grid_size, k=k) for _ in self.lags
        ])
        # conv input = in_channels (raw) + in_channels * n_lags (differences)
        conv_in = in_channels * (1 + len(self.lags))
        self.conv = nn.Conv1d(conv_in, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        feats = [x]
        for lag, spl in zip(self.lags, self.splines):
            if T > lag:
                x_now = x[:, :, lag:]          # (B, C, T-lag)
                x_prev = x[:, :, :-lag]        # (B, C, T-lag)
                diff = (x_now - x_prev).permute(0, 2, 1).reshape(-1, C)  # (B*(T-lag), C)
                diff = spl(diff).reshape(B, T - lag, C).permute(0, 2, 1)  # (B, C, T-lag)
                feats.append(diff)
            else:
                # pad with zeros to keep alignment
                feats.append(torch.zeros_like(x))
        # concat along channels — all have same T only if lags handled;
        # simpler: pad diff to T by zero-padding left
        feats_align = []
        for f in feats:
            if f.shape[2] == T:
                feats_align.append(f)
            else:
                pad = T - f.shape[2]
                feats_align.append(torch.cat([torch.zeros(B, C, pad, device=x.device), f], dim=2))
        cat = torch.cat(feats_align, dim=1)  # (B, C*(1+n_lags), T)
        return self.conv(cat)


class LinearCalibConv1d(nn.Module):
    """Linear calibration control: per-channel learnable scale + conv.

    Same structure as KANConv1d but replaces the B-spline with a pure
    per-channel linear scale (diag). Isolation test: if performance
    matches KANConv1d, the gain comes from input calibration, not from
    spline nonlinearity. If KANConv1d is better, nonlinearity matters.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 grid_size=3, k=3):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_channels))
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x * self.scale.view(1, -1, 1))


class DualAxisKANConv1d(nn.Module):
    """Dual-axis KAN conv: feature-value spline + temporal-difference spline.

    Combines KANConv (per-channel spline on raw feature VALUES, feature axis)
    and TemporalKAN (per-channel spline on lagged DIFFERENCES, temporal axis)
    in one input layer, so the convolution sees nonlinearly calibrated
    values AND nonlinearly calibrated change rates simultaneously.

    Design: for each input channel:
      - value spline φ_i(x_i(t)) on raw values
      - for each lag, difference spline ψ_{i,lag}(x_i(t) - x_i(t-lag))
    Concat [φ(x) | ψ_1(Δx) ... ψ_L(Δx)] -> linear conv.

    Both splines are identity-init -> starts as pure linear conv with
    duplicated inputs; learns nonlinear calibration per axis.

    y_j(t) = Σ_i Σ_k w_{j,i,k} · [φ_i(x_i(t+k)) | ψ_{i,lag}(x_i(t+k)-x_i(t+k-lag))]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 lags=(1, 3, 6, 12), grid_size=3, k=3):
        super().__init__()
        self.lags = tuple(lags)
        # feature-axis: one spline per channel (identity init)
        self.val_spline = SplineAct(in_channels, grid_size=grid_size, k=k)
        # temporal-axis: one spline per (channel, lag)
        self.diff_splines = nn.ModuleList([
            SplineAct(in_channels, grid_size=grid_size, k=k) for _ in self.lags
        ])
        conv_in = in_channels * (1 + len(self.lags))
        self.conv = nn.Conv1d(conv_in, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        feats = []
        # feature-axis: spline on raw values
        x_v = x.permute(0, 2, 1).reshape(B * T, C)
        x_v = self.val_spline(x_v).reshape(B, T, C).permute(0, 2, 1)
        feats.append(x_v)
        # temporal-axis: spline on differences
        for lag, spl in zip(self.lags, self.diff_splines):
            if T > lag:
                x_now = x[:, :, lag:]
                x_prev = x[:, :, :-lag]
                diff = (x_now - x_prev).permute(0, 2, 1).reshape(-1, C)
                diff = spl(diff).reshape(B, T - lag, C).permute(0, 2, 1)
                feats.append(diff)
            else:
                feats.append(torch.zeros_like(x))
        feats_align = []
        for f in feats:
            if f.shape[2] == T:
                feats_align.append(f)
            else:
                pad = T - f.shape[2]
                feats_align.append(torch.cat([torch.zeros(B, C, pad, device=x.device), f], dim=2))
        cat = torch.cat(feats_align, dim=1)  # (B, C*(1+n_lags), T)
        return self.conv(cat)


class AxisFusedKANConv1d(nn.Module):
    """Axis-fused KAN conv: value-spline + diff-spline ADDED in channel space.

    DualAxis failed because concat blew up channels (7 -> 35), diluting the
    conv weights. Here the two axes' nonlinear calibrations are SUMMED:
        z_i(t) = φ_i(x_i(t)) + Σ_lag ψ_{i,lag}(x_i(t) - x_i(t-lag))
    keeping in_channels == 7. Both splines identity-init, so the sum starts
    as ~2x identity; conv learns to combine the axes. Input-dependent
    calibration from both value domain and change-rate domain, no channel
    explosion, no parameter blow-up.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 lags=(1, 3, 6, 12), grid_size=3, k=3):
        super().__init__()
        self.lags = tuple(lags)
        self.val_spline = SplineAct(in_channels, grid_size=grid_size, k=k)
        self.diff_splines = nn.ModuleList([
            SplineAct(in_channels, grid_size=grid_size, k=k) for _ in self.lags
        ])
        # per-lag learnable gate: how much each temporal scale contributes
        self.lag_gate = nn.Parameter(torch.ones(len(self.lags)) / len(self.lags))
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.forward_with_diff(x)[0]

    def forward_with_diff(self, x):
        """Return (conv_out, diff_sum) — diff_sum is the temporal-axis calibration
        (B, C, T), usable as a parallel signal for the LSTM branch."""
        B, C, T = x.shape
        x_v = x.permute(0, 2, 1).reshape(B * T, C)
        z = self.val_spline(x_v).reshape(B, T, C).permute(0, 2, 1)  # (B,C,T)
        dsum = torch.zeros_like(z)
        for lag, spl, g in zip(self.lags, self.diff_splines, self.lag_gate):
            if T > lag:
                diff = (x[:, :, lag:] - x[:, :, :-lag]).permute(0, 2, 1).reshape(-1, C)
                ds = spl(diff).reshape(B, T - lag, C).permute(0, 2, 1)
                ds = torch.cat([torch.zeros(B, C, lag, device=x.device), ds], dim=2)
                dsum = dsum + torch.sigmoid(g) * ds
        return self.conv(z), dsum


class SplineLSTMCell(nn.Module):
    """LSTM cell with learnable B-spline candidate activation (KAN idea).

    Standard i/f/o sigmoid gates kept for stability.
    Candidate gate g: LN → SplineAct(W_c·[x,h]) instead of tanh(W_c·[x,h]).
    LayerNorm aligns pre-activation distribution to fixed spline grid [-1,1].
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ifo = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        self.spline_c = SplineAct(hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat([x_t, h_prev], dim=1)
        ifo = self.W_ifo(combined)
        i, f, o = torch.chunk(ifo, 3, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c_cand = self.spline_c(self.ln_c(self.W_c(combined)))
        c = f * c_prev + i * c_cand
        h = o * torch.tanh(c)
        return h, c


class SplineLSTM(nn.Module):
    """Multi-layer LSTM with spline candidate gate. Drop-in replacement for nn.LSTM.

    Args:
        input_size, hidden_size, num_layers — same as nn.LSTM
        batch_first=True always
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            sz = input_size if l == 0 else hidden_size
            self.cells.append(SplineLSTMCell(sz, hidden_size))

    def forward(self, x, h0_c0=None):
        B, T, _ = x.shape
        device = x.device
        if h0_c0 is None:
            h = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
            c = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        else:
            h0, c0 = h0_c0
            h = list(h0)
            c = list(c0)

        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l](inp, h[l], c[l])
                inp = h[l]
            outputs.append(h[-1].unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        return out, (h_n, c_n)


class TKANCell(nn.Module):
    """T-KAN cell (Temporal-KAN, Makinde et al. 2026, faithful full version).

    Paper idea: "replace the fixed, linear weights of standard LSTMs with
    learnable B-spline activation functions" — the four gates i/f/o/c all use
    a KAN (B-spline + linear shortcut, our KANLayer_custom) instead of a
    single Linear layer + fixed activation. Learning the SHAPE of signals,
    not just magnitude.

    Gates keep sigmoid outputs; cell candidate c is the KAN output directly
    (paper replaces the weight-activation pair). Linear shortcut zero-init
    -> starts ~ standard LSTM behavior (sigmoid(0)=0.5), stable training.

    Adaptation (user-approved): input bottleneck — concat[x_t, h_prev] (112d)
    is linearly projected to hidden_size (64d) before the four KAN gates,
    so the KAN inputs stay 64d (4GB GPU constraint; 4-5x speedup, still
    faithful: all four gates fully KAN-ized).
    """
    def __init__(self, input_size, hidden_size, grid_size=3, k=3):
        super().__init__()
        self.proj = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.kan_i = KANLayer_custom(hidden_size, hidden_size, grid_size=grid_size, k=k)
        self.kan_f = KANLayer_custom(hidden_size, hidden_size, grid_size=grid_size, k=k)
        self.kan_o = KANLayer_custom(hidden_size, hidden_size, grid_size=grid_size, k=k)
        self.kan_c = KANLayer_custom(hidden_size, hidden_size, grid_size=grid_size, k=k)
        # identity-ish start: zero spline + near-zero linear -> gates ≈ sigmoid(0)
        with torch.no_grad():
            for kan in (self.kan_i, self.kan_f, self.kan_o, self.kan_c):
                kan.linear.weight.mul_(0.01)

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat([x_t, h_prev], dim=1)
        g_in = self.proj(combined)
        i = torch.sigmoid(self.kan_i(g_in))
        f = torch.sigmoid(self.kan_f(g_in))
        o = torch.sigmoid(self.kan_o(g_in))
        c_cand = self.kan_c(g_in)
        c = f * c_prev + i * c_cand
        h = o * torch.tanh(c)
        return h, c


class TKAN(nn.Module):
    """Multi-layer T-KAN (full gate KAN-ization). Drop-in for nn.LSTM.

    batch_first=True always.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, grid_size=3, k=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            sz = input_size if l == 0 else hidden_size
            self.cells.append(TKANCell(sz, hidden_size, grid_size=grid_size, k=k))

    def forward(self, x, h0_c0=None):
        B, T, _ = x.shape
        device = x.device
        if h0_c0 is None:
            h = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
            c = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        else:
            h0, c0 = h0_c0
            h = list(h0)
            c = list(c0)
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l](inp, h[l], c[l])
                inp = h[l]
            outputs.append(h[-1].unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        return out, (h_n, c_n)
