import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.interpolate import CubicSpline
from config import *


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(file_path, target_col=TARGET_COL):
    df = pd.read_csv(file_path)
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    data = df[[target_col]].values.astype(np.float32)
    return data


def prepare_sequences(data, lookback=LOOKBACK):
    sequences = []
    for i in range(len(data) - lookback):
        sequences.append(data[i:i + lookback])
    sequences = np.array(sequences)
    return sequences[:, :-1, :], sequences[:, -1, :]


def split_data(data, lookback=LOOKBACK, val_ratio=SPLIT_RATIOS[1], test_ratio=SPLIT_RATIOS[2]):
    data_raw = data if isinstance(data, np.ndarray) else data.to_numpy()
    sequences = []
    for index in range(len(data_raw) - lookback):
        sequences.append(data_raw[index: index + lookback])
    sequences = np.array(sequences)
    n_samples = sequences.shape[0]

    test_size = int(np.round(test_ratio * n_samples))
    val_size = int(np.round(val_ratio * n_samples))
    train_size = n_samples - val_size - test_size

    x_train = sequences[:train_size, :-1, :]
    y_train = sequences[:train_size, -1, :]
    x_val = sequences[train_size:train_size + val_size, :-1, :]
    y_val = sequences[train_size:train_size + val_size, -1, :]
    x_test = sequences[train_size + val_size:, :-1, :]
    y_test = sequences[train_size + val_size:, -1, :]

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_val, y_val, x_test, y_test


def scale_data(x_train, y_train, x_val, y_val, x_test, y_test):
    n_features = x_train.shape[2]
    x_train_2d = x_train.reshape(-1, n_features)
    y_train_2d = y_train.reshape(-1, 1)
    x_val_2d = x_val.reshape(-1, n_features)
    y_val_2d = y_val.reshape(-1, 1)
    x_test_2d = x_test.reshape(-1, n_features)
    y_test_2d = y_test.reshape(-1, 1)

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    x_train_scaled = scaler_x.fit_transform(x_train_2d).reshape(x_train.shape)
    y_train_scaled = scaler_y.fit_transform(y_train_2d)
    x_val_scaled = scaler_x.transform(x_val_2d).reshape(x_val.shape)
    y_val_scaled = scaler_y.transform(y_val_2d)
    x_test_scaled = scaler_x.transform(x_test_2d).reshape(x_test.shape)
    y_test_scaled = scaler_y.transform(y_test_2d)

    return (x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled,
            x_test_scaled, y_test_scaled, scaler_y)


def to_tensor(*arrays, device=DEVICE):
    tensors = []
    for a in arrays:
        t = torch.from_numpy(a).float()
        if device != 'cpu':
            t = t.to(device)
        tensors.append(t)
    return tuple(tensors)


def compute_profit(active_power, hour, price_schedule=None, consumption_schedule=None,
                   energy_conversion=ENERGY_CONVERSION, i_base=I_BASE):
    if price_schedule is None:
        price_schedule = PRICE_SCHEDULE
    if consumption_schedule is None:
        consumption_schedule = CONSUMPTION_SCHEDULE

    price = price_schedule.get(int(hour), 0.15)
    consumption = 5
    for (h_start, h_end), val in consumption_schedule:
        if h_start <= int(hour) < h_end:
            consumption = val
            break

    energy = active_power * energy_conversion
    profit = (energy - consumption) * price + i_base
    return profit


def evaluate_metrics(y_true, y_pred, scaler_y=None):
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    else:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


def paired_ttest(results_a, results_b, metric='MSE'):
    a = np.array(results_a[metric])
    b = np.array(results_b[metric])
    t_stat, p_value = stats.ttest_rel(a, b)
    return {'t_statistic': t_stat, 'p_value': p_value, 'mean_diff': a.mean() - b.mean()}


def compute_inverse_metrics(model, x_data, y_data, scaler_y):
    model.eval()
    with torch.no_grad():
        predictions = model(x_data).cpu().numpy()
        actuals = y_data.cpu().numpy()
    return evaluate_metrics(actuals, predictions, scaler_y)


def plot_error_distribution(errors, title='Error Distribution', save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'N({mu:.4f}, {sigma:.4f})')
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].scatter(range(len(errors)), sorted(errors), s=1, alpha=0.5)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Sample Index (sorted)')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Sorted Error Profile')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_hourly_error_boxplot(timestamps, errors, title='Hourly Error Distribution', save_path=None):
    hours = [t.hour + t.minute / 60 for t in timestamps]
    df = pd.DataFrame({'hour_bin': [int(h) for h in hours], 'error': np.abs(errors)})

    fig, ax = plt.subplots(figsize=(14, 5))
    df.boxplot(column='error', by='hour_bin', ax=ax, showfliers=False)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Absolute Error')
    ax.set_title(title)
    ax.get_figure().suptitle('')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_daily_zoom(timestamps, y_true, y_pred, day_idx, title=None, save_path=None):
    day_df = pd.DataFrame({
        'timestamp': timestamps,
        'true': y_true.ravel(),
        'pred': y_pred.ravel()
    })
    day_df['date'] = day_df['timestamp'].dt.date
    unique_dates = sorted(day_df['date'].unique())

    if day_idx >= len(unique_dates):
        day_idx = len(unique_dates) - 1

    target_date = unique_dates[day_idx]
    subset = day_df[day_df['date'] == target_date]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(subset)), subset['true'], 'b-', lw=1.5, label='Actual')
    ax.plot(range(len(subset)), subset['pred'], 'r--', lw=1.5, label='Predicted')
    ax.fill_between(range(len(subset)), subset['true'], subset['pred'], alpha=0.2, color='gray')
    ax.set_xlabel('5-min Timestep')
    ax.set_ylabel('Profit')
    ax.set_title(title or f'Daily Zoom: {target_date}')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def get_high_error_days(timestamps, y_true, y_pred, n_days=2):
    df = pd.DataFrame({
        'date': [t.date() for t in timestamps],
        'error': np.abs(y_true.ravel() - y_pred.ravel())
    })
    daily_mae = df.groupby('date')['error'].mean().sort_values(ascending=False)
    return daily_mae.head(n_days).index.tolist()


def plot_training_curves(train_losses, val_losses, title='Training Curve', save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, 'b-', lw=1, label='Train Loss')
    ax.plot(val_losses, 'r-', lw=1, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_ga_fitness(fitness_history, title='GA Fitness History', save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fitness_history, 'g-', lw=1, marker='o', markersize=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
