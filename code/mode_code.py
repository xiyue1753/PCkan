"""CNN-LSTM-KAN core model and training pipeline.

Current valid mechanisms (Phase 11+):
  - CNN_LSTM_Head: backbone + configurable head
    Heads: FC, MLP_mid, KAN_mid, KAN_custom
    KAN enhancements: KANConv (feature-axis), TemporalKAN (temporal-axis),
                      ResKAN_v2 (output-axis residual)
  - FusedCNN_LSTM_KAN + GAOptimizer: GA initialization (retained)
  - Trainer: mini-batch, early stopping (patience/min_delta)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time, json, re, random, shutil, warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.interpolate import CubicSpline
from scipy import stats

warnings.filterwarnings('ignore')
from config import *
from kanconv.kan_layers import KANLayer_custom, SplineAct, KANConv1d, TemporalKANConv1d, DualAxisKANConv1d, LinearCalibConv1d, AxisFusedKANConv1d, SplineLSTM, PerFeatureMLP, TKAN, KernelSideKANConv1d, MLPCalibConv1d


# ============================================================
# FusedCNN_LSTM_KAN (retained for GA; pykan replaced by KANLayer_custom)
# ============================================================

class FusedCNN_LSTM_KAN(nn.Module):

    def __init__(self, in_features=6, hidden_size=64, num_layers=2, out_channels=48,
                 horizon=12, use_dilated=False, use_highway=False,
                 use_lw_seq=False, use_wide_kan=False, kan_seed=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_dilated = use_dilated
        self.use_highway = use_highway
        self.use_lw_seq = use_lw_seq
        self.use_wide_kan = use_wide_kan
        self.horizon = horizon

        if use_dilated:
            self.conv = nn.Sequential(
                nn.Conv1d(in_features, out_channels, kernel_size=3, dilation=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, dilation=4, padding=4),
                nn.ReLU(),
                nn.MaxPool1d(2))
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_features, out_channels, kernel_size=3, padding=1),
                nn.ReLU(), nn.MaxPool1d(2))

        lstm_input_dim = out_channels
        if use_highway:
            self.raw_proj = nn.Linear(in_features, out_channels)
            self.highway_alpha = nn.Parameter(torch.tensor(0.5))

        self.lstm = nn.LSTM(lstm_input_dim, hidden_size, num_layers, batch_first=True)

        if use_lw_seq:
            self.seq_weights = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softmax(dim=1))

        if use_wide_kan:
            self.kan = nn.Sequential(
                KANLayer_custom(hidden_size, hidden_size, grid_size=3, k=3),
                KANLayer_custom(hidden_size, horizon, grid_size=3, k=3))
        else:
            self.kan = KANLayer_custom(hidden_size, horizon, grid_size=3, k=3)

    def set_cnn_weights(self, chromosome):
        start_idx = 0
        with torch.no_grad():
            for p in self.conv.parameters():
                if p.requires_grad:
                    n = p.data.numel()
                    if start_idx + n > len(chromosome):
                        raise ValueError("Chromosome too short")
                    p.data = chromosome[start_idx:start_idx + n].reshape(p.data.shape).clone()
                    start_idx += n

    def forward(self, x):
        B, T, C = x.shape
        x_c = x.permute(0, 2, 1)
        cnn_out = self.conv(x_c).permute(0, 2, 1)  # (B, T', out_c)

        if self.use_highway:
            raw_proj = self.raw_proj(x.mean(dim=1)).unsqueeze(1).expand(-1, cnn_out.shape[1], -1)
            a = torch.sigmoid(self.highway_alpha)
            lstm_input = a * cnn_out + (1 - a) * raw_proj
        else:
            lstm_input = cnn_out

        lstm_out, _ = self.lstm(lstm_input)  # (B, T', hidden)

        if self.use_lw_seq:
            w = self.seq_weights(lstm_out)   # (B, T', 1)
            lstm_agg = (w * lstm_out).sum(dim=1)
        else:
            lstm_agg = lstm_out[:, -1, :]

        out = self.kan(lstm_agg)
        return out


# ============================================================
# CNN_LSTM_Head — current main model
# ============================================================

class CNN_LSTM_Head(nn.Module):
    """CNN-LSTM backbone with configurable head + KAN enhancements.

    Heads: FC, MLP_mid, KAN_mid, KAN_custom
    KAN enhancements:
      - use_kanconv: KANConv1d (per-channel spline calibration before conv, feature axis)
      - use_tempkan: TemporalKANConv1d (spline on temporal differences, temporal axis)
      - use_reskan_lstm: ResKAN_v2, out = FC(h_last) + tanh(alpha)*KAN(concat[h_last,h_mean])
      - use_reskan_only: skip FC branch (ResKAN is the only output)
    """

    def __init__(self, in_features=6, hidden_size=64, num_layers=2, out_channels=48,
                 horizon=12, head_type='FC', kan_seed=0, device='cpu',
                 use_kanconv=False, use_reskan_lstm=False,
                 use_reskan_only=False, use_tempkan=False, use_dualaxis=False,
                 use_lin_calib=False, use_axis_fused=False, use_bridge_kan=False,
                 use_spline_lstm=False, use_split_kan=False, use_dual_inject=False,
                 use_mlp_inject=False, use_tkan=False,
                 use_shared_kanconv=False, use_kernel_kanconv=False, freeze_spline=False,
                 use_mlp_calib=False,
                 kan_grid=3, kan_k=3, no_bn=False,
                 kan_knots=None, kan_gate_init=None, pool_mode='max', kan_kernel=3,
                 reskan_alpha_mode='tanh', reskan_pool='last_mean', dropout=0.2):
        super().__init__()
        self.head_type = head_type
        self.kan_seed = kan_seed
        self._device = device
        self.use_kanconv = use_kanconv
        self.use_shared_kanconv = use_shared_kanconv
        self.use_kernel_kanconv = use_kernel_kanconv
        self.use_mlp_calib = use_mlp_calib
        self.freeze_spline = freeze_spline
        self.use_reskan_lstm = use_reskan_lstm
        self.use_reskan_only = use_reskan_only
        self.use_tempkan = use_tempkan
        self.use_dualaxis = use_dualaxis
        self.use_lin_calib = use_lin_calib
        self.use_axis_fused = use_axis_fused
        self.use_bridge_kan = use_bridge_kan
        self.use_spline_lstm = use_spline_lstm
        self.use_split_kan = use_split_kan
        self.use_dual_inject = use_dual_inject
        self.use_mlp_inject = use_mlp_inject
        self.use_tkan = use_tkan
        self.kan_grid = kan_grid
        self.kan_k = kan_k
        self.no_bn = no_bn
        self.reskan_alpha_mode = reskan_alpha_mode
        self.reskan_pool = reskan_pool
        self.pool_mode = pool_mode
        if pool_mode == 'avg':
            self._pool_layer = nn.AvgPool1d(2)
        elif pool_mode == 'none':
            self._pool_layer = nn.Identity()
        else:
            self._pool_layer = nn.MaxPool1d(2)
        self.kan_kernel = kan_kernel
        kan_kernel = getattr(self, 'kan_kernel', 3)
        kan_pad = kan_kernel // 2

        conv_in = in_features
        if use_mlp_inject:
            # M1 control: same DualInject structure but per-feature MLP
            # replaces both value and diff B-splines. Tests whether the
            # KAN gain survives without spline machinery.
            self.mlp_val = PerFeatureMLP(conv_in, hidden=8)
            self.mlp_diffs = nn.ModuleList([PerFeatureMLP(conv_in, hidden=8) for _ in (1, 3, 6, 12)])
            self.lag_gate = nn.Parameter(torch.ones(4) / 4)
            self.conv = nn.Sequential(
                nn.Conv1d(conv_in, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
            self.diff_norm = nn.LayerNorm(conv_in)
            self.lstm_in = out_channels + conv_in
        elif use_dual_inject:
            # DualInject: ONE KAN system (AxisFused) serves BOTH components:
            #   - z = φ(x) + Σψ(Δx) -> CNN (feature+change-rate calibration)
            #   - Σψ(Δx)            -> LSTM (temporal-axis calibration signal)
            # Shared ψ parameters: single coherent KAN, no independent-KAN
            # gradient competition. Diff path LN-stabilized for LSTM.
            self.kanconv1 = AxisFusedKANConv1d(conv_in, out_channels, 3, padding=1,
                                               lags=(1, 3, 6, 12), grid_size=3, k=3)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
            self.diff_norm = nn.LayerNorm(conv_in)
            self.lstm_in = out_channels + conv_in
        elif use_split_kan:
            # SplitKAN: value spline -> CNN path; diff spline -> LSTM input path.
            # CNN gets feature-axis calibration, LSTM gets temporal-axis diff
            # calibration, on SEPARATE paths (no overlap, no channel explosion).
            self.val_spline = SplineAct(conv_in, grid_size=3, k=3)
            self.conv = nn.Sequential(
                nn.Conv1d(conv_in, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
            self.diff_splines = nn.ModuleList([
                SplineAct(conv_in, grid_size=3, k=3) for _ in (1, 3, 6, 12)
            ])
            self.lag_gate = nn.Parameter(torch.ones(4) / 4)
            self.diff_pool = self._pool_layer
            self.diff_norm = nn.LayerNorm(conv_in)
            self.lstm_in = out_channels + conv_in
        elif use_axis_fused:
            self.kanconv1 = AxisFusedKANConv1d(conv_in, out_channels, 3, padding=1,
                                               lags=(1, 3, 6, 12), grid_size=3, k=3)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
        elif use_dualaxis:
            self.kanconv1 = DualAxisKANConv1d(conv_in, out_channels, 3, padding=1,
                                              lags=(1, 3, 6, 12), grid_size=3, k=3)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
        elif use_lin_calib:
            self.kanconv1 = LinearCalibConv1d(conv_in, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
        elif use_tempkan:
            self.kanconv1 = TemporalKANConv1d(conv_in, out_channels, 3, padding=1,
                                              lags=(1, 3, 6, 12), grid_size=3, k=3)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
        elif use_kernel_kanconv:
            # kernel-side KAN conv (ConvKAN family): per-edge spline kernel
            self.kanconv1 = KernelSideKANConv1d(conv_in, out_channels, 3, padding=1,
                                                grid_size=kan_grid, k=kan_k)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
        elif use_shared_kanconv:
            # shared value-side spline (SV-KAN style): one spline for all channels
            self.kanconv1 = KANConv1d(conv_in, out_channels, 3, padding=1,
                                      grid_size=kan_grid, k=kan_k, shared=True,
                                      knots=kan_knots, gate_init=kan_gate_init)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
        elif use_kanconv:
            self.kanconv1 = KANConv1d(conv_in, out_channels, kan_kernel, padding=kan_pad,
                                      grid_size=kan_grid, k=kan_k,
                                      knots=kan_knots, gate_init=kan_gate_init)
            if no_bn:
                # No-BN variant: GA sigma tuning fully effective (no BN to
                # absorb amplitude). ReLU directly after kanconv1.
                self.bn1 = nn.Identity()
                self.conv2_block = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv1d(out_channels, out_channels, kan_kernel, padding=kan_pad),
                    nn.ReLU(),
                    self._pool_layer)
            else:
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.conv2_block = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv1d(out_channels, out_channels, kan_kernel, padding=kan_pad),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    self._pool_layer)
        elif use_mlp_calib:
            # R1#6 control: per-channel MLP calibration (identity-init residual)
            # in the same position as the KANConv value-side spline.
            self.kanconv1 = MLPCalibConv1d(conv_in, out_channels, kan_kernel, padding=kan_pad)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kan_kernel, padding=kan_pad),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(conv_in, out_channels, kan_kernel, padding=kan_pad),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kan_kernel, padding=kan_pad),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                self._pool_layer)
        self.dropout_c = nn.Dropout(dropout)
        if use_bridge_kan:
            # Bridge-KAN: per-channel spline calibration on CNN output features
            # (B, out_channels, T') before LSTM. Identity-init, +out_c*(G+k) params.
            self.bridge_kan = SplineAct(out_channels, grid_size=3, k=3)
        if use_spline_lstm:
            # SplineLSTM: KAN candidate gate inside LSTM (per-cell spline).
            self.lstm = SplineLSTM(out_channels, hidden_size, num_layers)
        elif use_tkan:
            # T-KAN full version: all four gates KAN-ized (paper-faithful).
            self.lstm = TKAN(out_channels, hidden_size, num_layers)
        elif use_split_kan or use_dual_inject or use_mlp_inject:
            self.lstm = nn.LSTM(self.lstm_in, hidden_size, num_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True)
        self.dropout_l = nn.Dropout(dropout)

        if head_type == 'FC':
            self.head = nn.Linear(hidden_size, horizon)
        elif head_type == 'MLP_mid':
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32), nn.ReLU(),
                nn.Linear(32, horizon))
        elif head_type == 'KAN_mid':
            self.head = nn.Sequential(
                KANLayer_custom(hidden_size, 16, grid_size=3, k=3),
                KANLayer_custom(16, horizon, grid_size=3, k=3))
        elif head_type == 'KAN_custom':
            self.head = nn.Sequential(
                KANLayer_custom(hidden_size, 16, grid_size=3, k=3),
                KANLayer_custom(16, horizon, grid_size=3, k=3))
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        if use_reskan_lstm:
            # KAN input dimension depends on pooling variant
            if reskan_pool == 'last_mean':
                pool_dim = hidden_size * 2
            elif reskan_pool == 'last_mean_std':
                pool_dim = hidden_size * 3
            elif reskan_pool == 'last3_mean':
                pool_dim = hidden_size * 4
            else:
                raise ValueError(f'Unknown reskan_pool: {reskan_pool}')
            self.reskan_lstm = KANLayer_custom(pool_dim, horizon, grid_size=3, k=3)
            # alpha init: 0.1 both modes (linear mode is NOT tanh-clamped)
            self.reskan_alpha = nn.Parameter(torch.tensor(0.1))

        if freeze_spline:
            # frozen-control: spline locked at identity -> KANConv degenerates
            # to a plain linear conv (attribution: structure vs learning)
            for m in self.modules():
                if isinstance(m, SplineAct):
                    m.coef.requires_grad_(False)

    def forward(self, x):
        if self.use_mlp_inject:
            x3 = x.permute(0, 2, 1)                            # (B, C, T)
            B, C, T = x3.shape
            xt = x3.permute(0, 2, 1).reshape(B * T, C)
            z = self.mlp_val(xt).reshape(B, T, C).permute(0, 2, 1)  # (B, C, T)
            dsum = torch.zeros_like(z)
            for lag, mlp, g in zip((1, 3, 6, 12), self.mlp_diffs, self.lag_gate):
                if T > lag:
                    diff = (x3[:, :, lag:] - x3[:, :, :-lag]).permute(0, 2, 1).reshape(-1, C)
                    ds = mlp(diff).reshape(B, T - lag, C).permute(0, 2, 1)
                    ds = torch.cat([torch.zeros(B, C, lag, device=x.device), ds], dim=2)
                    dsum = dsum + torch.sigmoid(g) * ds
            c = self.conv(z).permute(0, 2, 1)                  # (B, T', Co)
            dsum = torch.nn.functional.max_pool1d(dsum, kernel_size=2, stride=2).permute(0, 2, 1)
            dsum = self.diff_norm(dsum)
            c = torch.cat([c, dsum], dim=2)
        elif self.use_split_kan:
            # value spline calibrates features BEFORE CNN
            xv = x.permute(0, 2, 1)                      # (B, C, T)
            B, C, T = xv.shape
            xv_t = xv.permute(0, 2, 1).reshape(B * T, C)
            xv = self.val_spline(xv_t).reshape(B, T, C).permute(0, 2, 1)
            c = self.conv(xv).permute(0, 2, 1)
        elif self.use_dual_inject:
            c_raw, dsum = self.kanconv1.forward_with_diff(x.permute(0, 2, 1))
            c_raw = self.bn1(c_raw)
            c = self.conv2_block(c_raw).permute(0, 2, 1)      # (B, T', Co)
            # diff signal (B, C, T): MaxPool along T to T', LN, inject into LSTM
            dsum = torch.nn.functional.max_pool1d(dsum, kernel_size=2, stride=2)  # (B, C, T')
            dsum = dsum.permute(0, 2, 1)                       # (B, T', C)
            dsum = self.diff_norm(dsum)                        # (B, T', C)
            c = torch.cat([c, dsum], dim=2)                    # (B, T', Co+C)
        elif self.use_kanconv or self.use_tempkan or self.use_dualaxis or self.use_lin_calib or self.use_axis_fused or self.use_shared_kanconv or self.use_kernel_kanconv or self.use_mlp_calib:
            c = self.kanconv1(x.permute(0, 2, 1))
            c = self.bn1(c)
            c = self.conv2_block(c)
            c = c.permute(0, 2, 1)
        else:
            c = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        c = self.dropout_c(c)
        if self.use_split_kan:
            # SplitKAN path: CNN already got value-spline features above; now
            # add temporal-axis diff features to LSTM input.
            B, Tp, Co = c.shape
            x3 = x.permute(0, 2, 1)                      # (B, C, T)
            diffs = []
            for lag, spl in zip((1, 3, 6, 12), self.diff_splines):
                if x3.shape[2] > lag:
                    diff = (x3[:, :, lag:] - x3[:, :, :-lag]).permute(0, 2, 1).reshape(-1, x3.shape[1])
                    ds = spl(diff).reshape(B, x3.shape[2] - lag, x3.shape[1]).permute(0, 2, 1)
                    ds = torch.cat([torch.zeros(B, x3.shape[1], lag, device=x.device), ds], dim=2)
                else:
                    ds = torch.zeros_like(x3)
                diffs.append(ds)
            dsum = sum(torch.sigmoid(g) * d for g, d in zip(self.lag_gate, diffs))
            dsum = self.diff_pool(dsum).permute(0, 2, 1)  # (B, T', C)
            dsum = self.diff_norm(dsum)                   # stabilize scale for LSTM
            c = torch.cat([c, dsum], dim=2)              # (B, T', Co+C)
        elif self.use_bridge_kan:
            # Bridge-KAN: calibrate CNN feature map per-channel before LSTM.
            # c is (B, T', out_c); SplineAct 3D path expects (B, C, T).
            c = c.permute(0, 2, 1)
            c = self.bridge_kan(c)
            c = c.permute(0, 2, 1)
        out_all, _ = self.lstm(c)
        h = self.dropout_l(out_all[:, -1, :])

        out = None
        if not self.use_reskan_only:
            out = self.head(h)
        if self.use_reskan_lstm:
            h_last = out_all[:, -1, :]          # (B, 64) richest step
            h_mean = out_all.mean(dim=1)         # (B, 64) global context
            if self.reskan_pool == 'last_mean':
                h_concat = torch.cat([h_last, h_mean], dim=1)      # (B, 128)
            elif self.reskan_pool == 'last_mean_std':
                h_std = out_all.std(dim=1)                        # (B, 64)
                h_concat = torch.cat([h_last, h_mean, h_std], dim=1)  # (B, 192)
            elif self.reskan_pool == 'last3_mean':
                h3 = out_all[:, -3:, :].reshape(out_all.shape[0], -1)  # (B, 192)
                h_concat = torch.cat([h3, h_mean], dim=1)             # (B, 256)
            if self.reskan_alpha_mode == 'linear':
                scale = self.reskan_alpha.clamp(0, 1)   # no tanh gate
            else:
                scale = torch.tanh(self.reskan_alpha)
            rk = scale * self.reskan_lstm(h_concat)
            out = out + rk if out is not None else rk
        return out

    def get_alpha(self):
        if self.use_reskan_lstm:
            if self.reskan_alpha_mode == 'linear':
                return self.reskan_alpha.clamp(0, 1).item()
            return torch.tanh(self.reskan_alpha).item()
        return None


def build_model_variant(variant='V0', in_features=6, hidden_size=64, num_layers=2,
                        out_channels=48, horizon=12, kan_seed=0):
    cfg = ARCH_VARIANTS[variant]
    return FusedCNN_LSTM_KAN(
        in_features=in_features, hidden_size=hidden_size, num_layers=num_layers,
        out_channels=out_channels, horizon=horizon, kan_seed=kan_seed,
        use_dilated=cfg['dilated'], use_highway=cfg['highway'],
        use_lw_seq=cfg['lw_seq'], use_wide_kan=cfg['wide_kan'])


# ============================================================
# GA with F1-F5 fitness
# ============================================================

class GAOptimizer:
    def __init__(self, model, pop_size=GA_SMALL_POP, mutation_rate=0.25,
                 crossover_rate=0.8, generations=GA_SMALL_GEN, device='cpu'):
        self.model = model
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.fitness_history = []
        self.best_per_generation = []
        self.device = device

    def _count_cnn_params(self):
        return sum(p.numel() for p in self.model.conv.parameters() if p.requires_grad)

    def initialize_population(self):
        n_params = self._count_cnn_params()
        population = []
        for _ in range(self.population_size):
            chromosome = torch.randn(n_params) * 0.1
            population.append(chromosome.to(self.device))
        return population

    def _set_weights(self, chromosome):
        self.model.set_cnn_weights(chromosome)

    def _cnn_params(self):
        for p in self.model.conv.parameters():
            if p.requires_grad:
                yield p

    def _compute_f1(self, x, y, crit):
        """train_MSE + gradient_norm"""
        self.model.train()
        y_pred = self.model(x)
        loss = crit(y_pred, y)
        self.model.zero_grad()
        loss.backward()
        gn = 0.0
        for param in self._cnn_params():
            if param.grad is not None:
                gn += (param.grad ** 2).sum().item()
        gn = np.sqrt(gn)
        self.model.zero_grad()
        return loss.item() + GA_LAMBDA_GRAD * gn

    def _compute_f2(self, x, y, crit, k=GA_FITNESS_K):
        """MC-mean of train_MSE + gradient_norm over k LSTM/KAN seeds"""
        orig_lstm = {k_: v.clone() for k_, v in self.model.lstm.state_dict().items()}
        orig_kan = {k_: v.clone() for k_, v in self.model.kan.state_dict().items()}
        total = 0.0
        for _ in range(k):
            self.model.lstm.reset_parameters()
            try:
                self.model.kan.reset_parameters()
            except (AttributeError, RuntimeError):
                pass
            total += self._compute_f1(x, y, crit)
        self.model.lstm.load_state_dict(orig_lstm)
        try:
            self.model.kan.load_state_dict(orig_kan)
        except (AttributeError, RuntimeError):
            pass
        return total / k

    def _compute_f3(self, x, y, crit):
        """SNIP: -sum|w*grad| + gradient_norm"""
        self.model.train()
        y_pred = self.model(x)
        loss = crit(y_pred, y)
        self.model.zero_grad()
        loss.backward()
        sensitivity = 0.0
        gn = 0.0
        for param in self._cnn_params():
            if param.grad is not None:
                sensitivity += (param.data * param.grad).abs().sum().item()
                gn += (param.grad ** 2).sum().item()
        gn = np.sqrt(gn)
        self.model.zero_grad()
        return -sensitivity + GA_LAMBDA_GRAD * gn

    def _compute_f4(self, x):
        """NASWOT: kernel norm / frobenius norm of activation matrix"""
        self.model.eval()
        with torch.no_grad():
            cnn_out = self.model.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            feat = cnn_out.reshape(-1, cnn_out.shape[-1])
            K = feat @ feat.T
            nuc = torch.linalg.matrix_norm(K, ord='nuc').item()
            fro = torch.linalg.matrix_norm(K, ord='fro').item()
            if fro > 0:
                return nuc / fro
            return 0.0

    def _compute_f5(self, x, y, crit):
        """F2 + F3 mixed"""
        return 0.5 * self._compute_f2(x, y, crit) + 0.5 * self._compute_f3(x, y, crit)

    def _evaluate(self, chromosome, x, y, crit, fitness_type):
        self._set_weights(chromosome)
        if fitness_type == 'F1':
            return self._compute_f1(x, y, crit)
        elif fitness_type == 'F2':
            return self._compute_f2(x, y, crit)
        elif fitness_type == 'F3':
            return self._compute_f3(x, y, crit)
        elif fitness_type == 'F4':
            return -self._compute_f4(x)
        elif fitness_type == 'F5':
            return self._compute_f5(x, y, crit)
        else:
            return self._compute_f1(x, y, crit)

    def optimize(self, x_train, y_train, criterion, fitness_type='F1', verbose=True):
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        population = self.initialize_population()
        best_fitness = float('inf')
        best_chromosome = None
        self.best_per_generation = []

        for gen in range(self.generations):
            fitness_scores = []
            for chromosome in population:
                fit = self._evaluate(chromosome, x_train, y_train, criterion, fitness_type)
                fitness_scores.append(fit)
                if fit < best_fitness:
                    best_fitness = fit
                    best_chromosome = chromosome.clone()
            self.fitness_history.append(best_fitness)
            self.best_per_generation.append(best_chromosome.clone())

            selected = []
            for _ in range(self.population_size):
                idx = np.random.choice(len(population), size=3, replace=False)
                best_idx = idx[np.argmin([fitness_scores[i] for i in idx])]
                selected.append(population[best_idx].clone())

            new_pop = [best_chromosome.clone()]
            for i in range(1, self.population_size):
                p1 = selected[np.random.randint(len(selected))]
                p2 = selected[np.random.randint(len(selected))]
                if np.random.rand() < self.crossover_rate and len(p1) > 1:
                    cp = np.random.randint(1, len(p1))
                    child = torch.cat([p1[:cp], p2[cp:]])
                else:
                    child = p1.clone()
                mask = torch.rand(len(child), device=self.device) < self.mutation_rate
                child[mask] += torch.randn(mask.sum().item(), device=self.device) * 0.1
                new_pop.append(child)
            population = new_pop[:self.population_size]

        return best_chromosome, best_fitness


# ============================================================
# Training + Evaluation
# ============================================================

class Trainer:
    def __init__(self, model, lr=0.01, device='cpu'):
        self.model = model.to(device)
        self.lr = lr
        self.device = device

    def train(self, x_train, y_train, x_val, y_val, epochs=30, verbose=False, batch_size=1024,
              patience=15, min_delta=1e-5, fixed_epochs=0, grad_accum=1):
        """fixed_epochs>0: train exactly that many epochs, return last state
        (no early stopping) — used for stable FC-vs-KAN comparison when
        early-stop variance dominates.
        grad_accum>1: split each batch into grad_accum micro-batches and
        accumulate gradients before one optimizer step — mathematically
        equivalent to the full batch (same gradient sum, same Adam update),
        used when the model doesn't fit the GPU at full batch size."""
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=False)
        crit = nn.MSELoss()
        train_losses, val_losses = [], []
        n_train = x_train.shape[0]

        best_val = float('inf')
        best_state = None
        best_epoch = 0
        no_improve = 0

        for e in range(epochs):
            self.model.train()
            perm = torch.randperm(n_train, device=self.device)
            epoch_loss = 0.0
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                # gradient accumulation: split into grad_accum micro-batches
                n_micro = min(grad_accum, len(idx))
                micro = max(1, len(idx) // n_micro)
                for j in range(0, len(idx), micro):
                    jdx = idx[j:j + micro]
                    pred = self.model(x_train[jdx])
                    loss = crit(pred, y_train[jdx]) / n_micro
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    # NOT pure logging: in train mode this forward updates the
                    # BatchNorm running stats - skipping it changes val/test
                    # (verified: test 0.0904 -> 0.1048 on site 212 seed 42).
                    full_pred = self.model(x_train[idx])
                    epoch_loss += crit(full_pred, y_train[idx]).item() * len(idx)
            epoch_loss /= n_train

            self.model.eval()
            with torch.no_grad():
                vp = self.model(x_val)
                vloss = crit(vp, y_val).item()
            train_losses.append(epoch_loss)
            val_losses.append(vloss)

            scheduler.step(vloss)

            if vloss < best_val - min_delta:
                best_val = vloss
                if fixed_epochs <= 0:
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                best_epoch = e
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience and fixed_epochs <= 0:
                break

        if fixed_epochs > 0:
            best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            best_epoch = fixed_epochs - 1

        if best_state is not None:
            self.model.load_state_dict(best_state)

        last_train_loss = train_losses[-1] if train_losses else 0.0
        return last_train_loss, best_val, best_epoch, train_losses, val_losses


def evaluate_full(model, x_test, y_test, scaler_y=None):
    """y_test should be in ORIGINAL scale; prediction is inverse-scaled if scaler_y provided"""
    model.eval()
    with torch.no_grad():
        pred = model(x_test).cpu().numpy()
        true = y_test.cpu().numpy()
    if scaler_y is not None:
        pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
    mse = mean_squared_error(true.ravel(), pred.ravel())
    mae = mean_absolute_error(true.ravel(), pred.ravel())
    r2 = r2_score(true.ravel(), pred.ravel())
    return {'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE': np.sqrt(mse),
            'pred': pred, 'true': true}


def prepare_multistep_data(df, feature_cols=FEATURE_COLS_V2, target_col=TARGET_COL,
                           seq_len=SEQLEN, max_horizon=max(HORIZONS)):
    values = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)

    X_list, y_list = [], []
    for i in range(len(values) - seq_len - max_horizon):
        X_list.append(values[i:i + seq_len])
        y_list.append(target[i + seq_len:i + seq_len + max_horizon])
    return np.array(X_list), np.array(y_list)


def small_scale_run(site_file, variant='V0', horizon=12, fitness_type='baseline',
                    epochs=30, seeds=[42, 123, 456], verbose=True):
    from utils import set_seed

    df = pd.read_csv(site_file, index_col=0, parse_dates=True)
    X, Y = prepare_multistep_data(df)
    Y = Y[:, :horizon]

    results = {}
    for seed in seeds:
        set_seed(seed)
        shutil.rmtree('./model', ignore_errors=True)

        n = len(X)
        tr_end = int(n * 0.64)
        vl_end = int(n * 0.80)
        x_train, y_train = torch.FloatTensor(X[:tr_end]), torch.FloatTensor(Y[:tr_end])
        x_val, y_val = torch.FloatTensor(X[tr_end:vl_end]), torch.FloatTensor(Y[tr_end:vl_end])
        x_test, y_test = torch.FloatTensor(X[vl_end:]), torch.FloatTensor(Y[vl_end:])

        scaler_x = MinMaxScaler((-1, 1)).fit(x_train.reshape(-1, X.shape[2]))
        scaler_y = MinMaxScaler((-1, 1)).fit(y_train.reshape(-1, 1))

        def scale_x(t, s):
            return torch.FloatTensor(s.transform(t.reshape(-1, t.shape[-1])).reshape(t.shape))

        x_train_s = scale_x(x_train, scaler_x)
        x_val_s = scale_x(x_val, scaler_x)
        x_test_s = scale_x(x_test, scaler_x)
        y_train_s = torch.FloatTensor(scaler_y.transform(y_train.reshape(-1, 1)).reshape(y_train.shape))
        y_val_s = torch.FloatTensor(scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape))

        model = build_model_variant(variant, in_features=X.shape[2], horizon=horizon, kan_seed=seed)

        if fitness_type != 'baseline':
            ga = GAOptimizer(model, pop_size=GA_SMALL_POP, generations=GA_SMALL_GEN, device=DEVICE)
            best_c, best_f = ga.optimize(x_train_s, y_train_s, nn.MSELoss(), fitness_type, verbose=False)
            model.set_cnn_weights(best_c)
            if verbose:
                print(f'  [{variant}][{fitness_type}] seed={seed}: GA best_fit={best_f:.4f}')

        trainer = Trainer(model, lr=0.01, device=DEVICE)
        train_loss, val_loss, _, _, _ = trainer.train(x_train_s, y_train_s, x_val_s, y_val_s,
                                                      epochs=epochs, verbose=False)
        metrics = evaluate_full(model, x_test_s, y_test, scaler_y=scaler_y)

        persistence_pred = x_test[:, -1, 0:1].repeat(1, horizon).numpy()
        persistence_mse = mean_squared_error(Y[vl_end:].ravel(), persistence_pred.ravel())
        metrics['Skill'] = 1 - metrics['MSE'] / persistence_mse if persistence_mse > 0 else 0

        results[seed] = metrics
        if verbose:
            print(f'  MSE={metrics["MSE"]:.6f}, R2={metrics["R2"]:.4f}, Skill={metrics["Skill"]:.4f}')

    mse_list = [r['MSE'] for r in results.values()]
    return {'variant': variant, 'horizon': horizon, 'fitness': fitness_type,
            'MSE_mean': np.mean(mse_list), 'MSE_std': np.std(mse_list),
            **{k: np.mean([r[k] for r in results.values()]) for k in ['MAE', 'R2', 'RMSE', 'Skill']}}
