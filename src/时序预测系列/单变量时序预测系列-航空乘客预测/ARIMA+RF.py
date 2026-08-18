import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(path='data.csv'):
    df = pd.read_csv(path)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df.columns = ['y']
    return df


def time_split(df, train_end, val_end):
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def fit_arima_and_residual(train_series, order=(2, 1, 2)):
    model = ARIMA(train_series, order=order).fit()
    fitted = model.fittedvalues
    residual = train_series - fitted
    return model, fitted, residual


def make_time_features(df, lags=24):
    X = pd.DataFrame(index=df.index)
    for lag in range(1, lags + 1):
        X[f'lag_{lag}'] = df['y'].shift(lag)
    X['roll_mean_12'] = df['y'].rolling(window=12, min_periods=1).mean().shift(1)
    X['roll_std_12'] = df['y'].rolling(window=12, min_periods=1).std().shift(1).fillna(0)
    X['roll_mean_3'] = df['y'].rolling(window=3, min_periods=1).mean().shift(1)
    X['month'] = df.index.month
    X['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    X['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    X['year_frac'] = (df.index - df.index[0]).days / 365.25
    return X


def prepare_direct_dataset(full_X, full_y, start_i, end_i, H, lags):
    X_list = {h: [] for h in range(1, H + 1)}
    y_list = {h: [] for h in range(1, H + 1)}
    idx_list = {h: [] for h in range(1, H + 1)}
    for t_i in range(start_i + lags, end_i - H):
        X_t = full_X.iloc[t_i].values
        for h in range(1, H + 1):
            target = full_y.iloc[t_i + h]
            X_list[h].append(X_t)
            y_list[h].append(target)
            idx_list[h].append(full_X.index[t_i])
    for h in range(1, H + 1):
        if len(X_list[h]) > 0:
            X_list[h] = np.vstack(X_list[h])
            y_list[h] = np.array(y_list[h])
        else:
            X_list[h] = np.empty((0, full_X.shape[1]))
            y_list[h] = np.array([])
    return X_list, y_list, idx_list


def train_rf_residuals(full_X, df, train_end, H, lags, arima_fitted_train):
    residual_series = pd.Series(index=df.index, dtype=float)
    residual_series.iloc[:train_end] = (df['y'].iloc[:train_end] - arima_fitted_train).values

    X_train_list, y_train_list, idx_train_list = prepare_direct_dataset(
        full_X, df['y'], 0, train_end, H, lags
    )

    y_resid_train_list = {}
    for h in range(1, H + 1):
        idxs = idx_train_list[h]
        y_resid = []
        valid_rows = []
        for i, idx in enumerate(idxs):
            t_idx = df.index.get_loc(idx) + h
            if t_idx < train_end:
                res_val = residual_series.iloc[t_idx]
                if not np.isnan(res_val):
                    y_resid.append(res_val)
                    valid_rows.append(i)
        if len(valid_rows) == 0:
            X_valid = np.empty((0, full_X.shape[1]))
        else:
            X_valid = X_train_list[h][valid_rows]
        y_resid_train_list[h] = (X_valid, np.array(y_resid))

    rf_models = {}
    rf_params = {'n_estimators': 100, 'max_depth': 8, 'random_state': 42, 'n_jobs': -1}
    for h in range(1, H + 1):
        X_h, y_h = y_resid_train_list[h]
        if X_h.shape[0] < 8:
            rf_models[h] = None
            continue
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_h, y_h)
        rf_models[h] = rf
    return rf_models


def train_combiner_on_validation(arima_model_train, rf_models, full_X, df,
                                  train_end, val_end, H, lags):
    horizon_pairs = {h: [] for h in range(1, H + 1)}
    for t_i in range(max(train_end, lags), val_end - H):
        X_t = full_X.iloc[t_i].values.reshape(1, -1)
        for h in range(1, H + 1):
            target_idx = t_i + h
            if target_idx >= val_end:
                continue
            y_true = df['y'].iloc[target_idx]
            rf_h = float(rf_models[h].predict(X_t)[0]) if rf_models.get(h) is not None else 0.0
            arima_h_pred = float(arima_model_train.forecast(steps=h).iloc[-1])
            horizon_pairs[h].append((arima_h_pred, rf_h, y_true))

    weights = {}
    for h in range(1, H + 1):
        if len(horizon_pairs[h]) < 2:
            weights[h] = None
            continue
        arr = np.array(horizon_pairs[h])
        arima_preds = arr[:, 0]
        rf_preds = arr[:, 1]
        y_trues = arr[:, 2]
        mse_arima = np.mean((arima_preds - y_trues) ** 2)
        mse_sum = np.mean(((arima_preds + rf_preds) - y_trues) ** 2)
        best_w, best_mse = 0.5, float('inf')
        for w in np.arange(0.0, 1.05, 0.05):
            combined = w * arima_preds + (1 - w) * rf_preds
            mse = np.mean((combined - y_trues) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w
        if best_mse >= min(mse_arima, mse_sum) * 0.98:
            weights[h] = None
        else:
            weights[h] = best_w
    return weights, None, None


def retrain_and_forecast(df, train_end, val_end, H, lags,
                          arima_order=(2, 1, 2), combiner=None,
                          comb_mean=None, comb_std=None):
    trainval = df.iloc[:val_end].copy()
    arima_tv = ARIMA(trainval['y'], order=arima_order).fit()
    fitted_tv = arima_tv.fittedvalues
    trainval['resid'] = trainval['y'] - fitted_tv

    full_X = make_time_features(df, lags=lags)
    X_tv_list, y_tv_list, idx_tv_list = prepare_direct_dataset(
        full_X, df['y'], 0, val_end, H, lags
    )

    y_resid_tv_list = {}
    for h in range(1, H + 1):
        idxs = idx_tv_list[h]
        y_resid = []
        valid_rows = []
        for i, idx in enumerate(idxs):
            t_idx = df.index.get_loc(idx) + h
            if t_idx < val_end:
                res_val = trainval['resid'].iloc[t_idx]
                if not np.isnan(res_val):
                    y_resid.append(res_val)
                    valid_rows.append(i)
        if len(valid_rows) == 0:
            X_valid = np.empty((0, full_X.shape[1]))
        else:
            X_valid = X_tv_list[h][valid_rows]
        y_resid_tv_list[h] = (X_valid, np.array(y_resid))

    rf_models_tv = {}
    rf_params = {'n_estimators': 100, 'max_depth': 8, 'random_state': 42, 'n_jobs': -1}
    for h in range(1, H + 1):
        X_h, y_h = y_resid_tv_list[h]
        if X_h.shape[0] < 8:
            rf_models_tv[h] = None
            continue
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_h, y_h)
        rf_models_tv[h] = rf

    origin_idx = val_end - 1
    full_X = make_time_features(df, lags=lags)
    X_origin = full_X.iloc[origin_idx].values.reshape(1, -1)
    arima_fore = arima_tv.forecast(steps=H)

    preds = {'h': [], 'arima': [], 'rf_resid': [], 'sum': [], 'combiner': [], 'true': []}
    for h in range(1, H + 1):
        arima_h = float(arima_fore.iloc[h - 1])
        rf_h = float(rf_models_tv[h].predict(X_origin)[0]) if rf_models_tv.get(h) is not None else 0.0
        sum_pred = arima_h + rf_h
        if combiner is not None and combiner.get(h) is not None:
            w = combiner[h]
            comb_pred = w * arima_h + (1 - w) * rf_h
        else:
            comb_pred = sum_pred
        true_idx = origin_idx + h
        true_val = df['y'].iloc[true_idx] if true_idx < len(df) else np.nan
        preds['h'].append(h)
        preds['arima'].append(arima_h)
        preds['rf_resid'].append(rf_h)
        preds['sum'].append(sum_pred)
        preds['combiner'].append(comb_pred)
        preds['true'].append(true_val)

    results = pd.DataFrame(preds)
    mask = ~results['true'].isna()

    def metrics(y_true, y_pred):
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

    metrics_arima = metrics(results.loc[mask, 'true'], results.loc[mask, 'arima'])
    metrics_sum = metrics(results.loc[mask, 'true'], results.loc[mask, 'sum'])
    metrics_comb = metrics(results.loc[mask, 'true'], results.loc[mask, 'combiner'])
    return results, (metrics_arima, metrics_sum, metrics_comb), arima_tv, rf_models_tv


def plot_results(df, train_end, val_end, train, arima_fitted,
                 results, rf_models_tv, full_X, H):
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.rcParams['figure.figsize'] = (12, 6)

    fig1, ax1 = plt.subplots()
    ax1.plot(df.index, df['y'], label='Observed (y)', color='crimson', linewidth=1.1)
    ax1.plot(train.index, arima_fitted, label='ARIMA fitted (train)', color='navy', linewidth=1.0)
    ax1.axvspan(df.index[0], df.index[train_end - 1], alpha=0.12, color='gold', label='Train')
    ax1.axvspan(df.index[train_end], df.index[val_end - 1], alpha=0.10, color='orchid', label='Validation')
    ax1.axvspan(df.index[val_end], df.index[-1], alpha=0.08, color='lightgreen', label='Test')
    ax1.set_title("Complete Time Series & ARIMA Fitting (Train)")
    ax1.set_ylabel("Passengers")
    ax1.legend()
    ax1.grid(alpha=0.2)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.plot(results['h'], results['true'], label='True', marker='o', color='black')
    ax2.plot(results['h'], results['arima'], label='ARIMA', marker='s', color='navy')
    ax2.plot(results['h'], results['sum'], label='ARIMA+RF (sum)', marker='D', color='crimson')
    ax2.plot(results['h'], results['combiner'], label='Combiner', marker='^', color='limegreen')
    ax2.set_xlabel('h (forecast horizon)')
    ax2.set_ylabel('Passengers')
    ax2.set_title("Multi-step Predictions: True / ARIMA / ARIMA+RF / Combiner")
    ax2.legend()
    ax2.grid(alpha=0.2)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots()
    ax3.plot(results['h'], results['true'] - results['arima'], label='Error ARIMA', marker='o', color='navy')
    ax3.plot(results['h'], results['true'] - results['sum'], label='Error ARIMA+RF(sum)', marker='D', color='crimson')
    ax3.plot(results['h'], results['true'] - results['combiner'], label='Error Combiner', marker='^', color='limegreen')
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_xlabel('h')
    ax3.set_ylabel('error (true - pred)')
    ax3.set_title("Residuals by Forecast Horizon")
    ax3.legend()
    ax3.grid(alpha=0.2)
    fig3.tight_layout()

    chosen_h = min(6, H)
    rf_chosen = rf_models_tv.get(chosen_h, None)
    fig4, ax4 = plt.subplots()
    if rf_chosen is not None:
        importances = rf_chosen.feature_importances_
        feat_names = full_X.columns.tolist()
        idxs = np.argsort(importances)[-12:][::-1]
        ax4.barh([feat_names[i] for i in idxs][::-1], importances[idxs][::-1], color='magenta')
        ax4.set_title(f"RF Feature Importance (h={chosen_h}, top 12)")
        ax4.set_xlabel("importance")
    else:
        ax4.text(0.1, 0.5, f"No RF trained for h={chosen_h}", fontsize=12)
    ax4.grid(alpha=0.15, axis='x')
    fig4.tight_layout()

    fig5, ax5 = plt.subplots()
    mask = ~results['true'].isna()
    ax5.scatter(results.loc[mask, 'true'], results.loc[mask, 'combiner'],
                s=90, c='crimson', edgecolor='k', alpha=0.9)
    mn = min(results.loc[mask, 'true'].min(), results.loc[mask, 'combiner'].min())
    mx = max(results.loc[mask, 'true'].max(), results.loc[mask, 'combiner'].max())
    ax5.plot([mn, mx], [mn, mx], linestyle='--', color='navy')
    ax5.set_xlabel('True')
    ax5.set_ylabel('Combiner Pred')
    ax5.set_title("Combiner Predictions vs True (Scatter)")
    ax5.grid(alpha=0.2)
    fig5.tight_layout()

    fig6, ax6 = plt.subplots()
    mask = ~results['true'].isna()
    ax6.plot(results.loc[mask, 'true'].values, label='True', marker='o', color='black', linewidth=1.5)
    ax6.plot(results.loc[mask, 'sum'].values, label='ARIMA+RF (sum)', marker='D', color='crimson', linewidth=1.5)
    ax6.fill_between(range(mask.sum()),
                        results.loc[mask, 'sum'].values * 0.95,
                        results.loc[mask, 'sum'].values * 1.05,
                        alpha=0.15, color='crimson', label='+-5% band')
    ax6.set_xlabel('Test Index')
    ax6.set_ylabel('Passengers')
    ax6.set_title("Test Set: True vs ARIMA+RF Prediction")
    ax6.legend()
    ax6.grid(alpha=0.2)
    fig6.tight_layout()

    fig1.savefig('plot_1_timeseries_arima.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2.savefig('plot_2_multistep_predictions.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3.savefig('plot_3_residuals.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)

    fig4.savefig('plot_4_rf_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)

    fig5.savefig('plot_5_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)

    fig6.savefig('plot_6_test_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig6)

    print("\n6 plots saved as plot_1~6_*.png")


def main():
    df = load_data('data.csv')
    print(f"Data shape: {df.shape}")
    print(df.head())

    n = len(df)
    train_end = int(n * 0.65)
    val_end = int(n * 0.85)
    H = 12
    lags = 24
    arima_order = (2, 1, 2)

    train, val, test = time_split(df, train_end, val_end)
    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    arima_model, arima_fitted, residual = fit_arima_and_residual(train['y'], order=arima_order)
    print(f"\nARIMA{arima_order} fitted on train. AIC={arima_model.aic:.2f}")

    full_X = make_time_features(df, lags=lags)

    rf_models = train_rf_residuals(full_X, df, train_end, H, lags, arima_fitted)
    print(f"RF models trained for {sum(1 for v in rf_models.values() if v is not None)}/{H} horizons")

    combiner, comb_mean, comb_std = train_combiner_on_validation(
        arima_model, rf_models, full_X, df, train_end, val_end, H, lags
    )
    print(f"Combiner trained: {'Per-horizon weighted' if combiner is not None else 'None (fallback to sum)'}")

    results, metrics, arima_tv, rf_models_tv = retrain_and_forecast(
        df, train_end, val_end, H, lags, arima_order=arima_order,
        combiner=combiner, comb_mean=comb_mean, comb_std=comb_std
    )

    print("\n" + "=" * 60)
    print("Evaluation Results (origin = last point of train+val)")
    print("=" * 60)
    print(f"  ARIMA-only:          RMSE={metrics[0]['RMSE']:.2f}, MAE={metrics[0]['MAE']:.2f}, MAPE={metrics[0]['MAPE']:.2f}%")
    print(f"  ARIMA+RF (sum):      RMSE={metrics[1]['RMSE']:.2f}, MAE={metrics[1]['MAE']:.2f}, MAPE={metrics[1]['MAPE']:.2f}%")
    print(f"  ARIMA+RF (combiner): RMSE={metrics[2]['RMSE']:.2f}, MAE={metrics[2]['MAE']:.2f}, MAPE={metrics[2]['MAPE']:.2f}%")

    display_df = results.copy()
    display_df['abs_err_arima'] = np.abs(display_df['true'] - display_df['arima'])
    display_df['abs_err_sum'] = np.abs(display_df['true'] - display_df['sum'])
    display_df['abs_err_comb'] = np.abs(display_df['true'] - display_df['combiner'])
    print("\nPer-horizon comparison:")
    print(display_df[['h', 'arima', 'rf_resid', 'sum', 'combiner', 'true',
                       'abs_err_arima', 'abs_err_sum', 'abs_err_comb']].to_string(index=False))

    plot_results(df, train_end, val_end, train, arima_fitted,
                 results, rf_models_tv, full_X, H)


if __name__ == "__main__":
    main()
