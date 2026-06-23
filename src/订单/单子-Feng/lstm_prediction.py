import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
# plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
# plt.rcParams["axes.unicode_minus"] = False
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)



DATA_PATH = "./data/数据集.xlsx"
OUTPUT_DIR = "./output/lstm_prediction"
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def load_and_merge_data():
    df_features = pd.read_excel(DATA_PATH, sheet_name="数据集")
    df_target = pd.read_excel(DATA_PATH, sheet_name="预测集")

    df_features["日期"] = pd.to_datetime(df_features["日期"])
    df_target["日期"] = pd.to_datetime(df_target["日期"])

    df_merged = pd.merge(df_target, df_features, on="日期", how="inner")
    df_merged = df_merged.sort_values("日期").reset_index(drop=True)

    df_merged = df_merged.ffill()
    df_merged = df_merged.bfill()

    return df_merged


def create_sequences(data, target_col_idx, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_col_idx])
    return np.array(X), np.array(y)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    early_stop_patience=10,
):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses


def predict_all(model, X, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        preds = model(X_t).squeeze().cpu().numpy()
    return preds


def inverse_transform(scaler, y_scaled, target_col_idx):
    dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
    dummy[:, target_col_idx] = y_scaled
    return scaler.inverse_transform(dummy)[:, target_col_idx]


def plot_results(
    y_train_true,
    y_train_pred,
    y_val_true,
    y_val_pred,
    y_test_true,
    y_test_pred,
    dates_train,
    dates_val,
    dates_test,
    output_dir,
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    ax1.plot(dates_train, y_train_true, label="Train Actual", alpha=0.7)
    ax1.plot(dates_train, y_train_pred, label="Train Pred", alpha=0.7, linestyle="--")
    ax1.set_title("Training Set: Actual vs Predicted")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[0, 1]
    ax2.plot(dates_val, y_val_true, label="Val Actual")
    ax2.plot(dates_val, y_val_pred, label="Val Pred", linestyle="--")
    ax2.set_title("Validation Set: Actual vs Predicted")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)

    ax3 = axes[1, 0]
    ax3.plot(dates_test, y_test_true, label="Test Actual")
    ax3.plot(dates_test, y_test_pred, label="Test Pred", linestyle="--")
    ax3.set_title("Test Set: Actual vs Predicted")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.legend()
    ax3.tick_params(axis="x", rotation=45)

    ax4 = axes[1, 1]
    ax4.scatter(y_test_true, y_test_pred, alpha=0.5)
    min_val = min(y_test_true.min(), y_test_pred.min())
    max_val = max(y_test_true.max(), y_test_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
    ax4.set_title("Scatter Plot: Actual vs Predicted")
    ax4.set_xlabel("Actual Price")
    ax4.set_ylabel("Predicted Price")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction results saved to {output_dir}/prediction_results.png")


def plot_loss_curves(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved to {output_dir}/loss_curves.png")


def plot_full_comparison(
    y_train_true,
    y_train_pred,
    y_val_true,
    y_val_pred,
    y_test_true,
    y_test_pred,
    dates_all,
    train_size,
    val_size,
    output_dir,
):
    y_all_true = np.concatenate([y_train_true, y_val_true, y_test_true])
    y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
    test_start = train_size + val_size

    plt.figure(figsize=(14, 6))

    plt.plot(
        dates_all[:test_start],
        y_all_true[:test_start],
        label="Train+Val Actual",
        alpha=0.7,
    )
    plt.plot(
        dates_all[:test_start],
        y_all_pred[:test_start],
        label="Train+Val Pred",
        alpha=0.7,
        linestyle="--",
    )
    plt.plot(
        dates_all[test_start:],
        y_all_true[test_start:],
        label="Test Actual",
        color="green",
    )
    plt.plot(
        dates_all[test_start:],
        y_all_pred[test_start:],
        label="Test Pred",
        color="orange",
        linestyle="--",
    )

    plt.axvline(
        x=dates_all[test_start], color="red", linestyle=":", label="Train/Test Split"
    )
    plt.title("Full Dataset: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/full_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Full comparison saved to {output_dir}/full_comparison.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== Loading and preprocessing data ===")
    df = load_and_merge_data()
    print(f"Merged dataset: {df}")
    print(f"Merged dataset shape: {df.shape}")
    print(f"Date range: {df['日期'].min()} to {df['日期'].max()}")

    # feature_cols = [col for col in df.columns if col not in ["日期", "价格"]]
    feature_cols = [col for col in df.columns if col not in ["日期"]]
    print(feature_cols)
    target_col = "价格"

    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target column: {target_col}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols + [target_col]].values)

    target_col_idx = -1

    X, y = create_sequences(scaled_data, target_col_idx, SEQ_LEN)
    print(f"Sequence data shape: X={X.shape}, y={y.shape}")

    train_size = int(len(X) * TRAIN_RATIO)
    val_size = int(len(X) * VAL_RATIO)
    test_size = len(X) - train_size - val_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = (
        X[train_size : train_size + val_size],
        y[train_size : train_size + val_size],
    )
    X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

    print(
        f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}"
    )

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = X_train.shape[2]
    output_size = 1

    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size).to(device)
    print(f"\nModel: LSTM({input_size}->{HIDDEN_SIZE}) x{NUM_LAYERS}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n=== Training model ===")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
    )

    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth"))

    y_train_pred = predict_all(model, X_train, device)
    y_val_pred = predict_all(model, X_val, device)
    y_test_pred = predict_all(model, X_test, device)

    y_train_true = y_train
    y_val_true = y_val
    y_test_true = y_test

    y_train_true_orig = inverse_transform(scaler, y_train, -1)
    y_train_pred_orig = inverse_transform(scaler, y_train_pred, -1)
    y_val_true_orig = inverse_transform(scaler, y_val, -1)
    y_val_pred_orig = inverse_transform(scaler, y_val_pred, -1)
    y_test_true_orig = inverse_transform(scaler, y_test, -1)
    y_test_pred_orig = inverse_transform(scaler, y_test_pred, -1)

    train_mae = mean_absolute_error(y_train_true_orig, y_train_pred_orig)
    train_rmse = np.sqrt(mean_squared_error(y_train_true_orig, y_train_pred_orig))
    train_mape = (
        np.mean(np.abs((y_train_true_orig - y_train_pred_orig) / y_train_true_orig))
        * 100
    )
    train_r2 = r2_score(y_train_true_orig, y_train_pred_orig)

    val_mae = mean_absolute_error(y_val_true_orig, y_val_pred_orig)
    val_rmse = np.sqrt(mean_squared_error(y_val_true_orig, y_val_pred_orig))
    val_mape = (
        np.mean(np.abs((y_val_true_orig - y_val_pred_orig) / y_val_true_orig)) * 100
    )
    val_r2 = r2_score(y_val_true_orig, y_val_pred_orig)

    test_mae = mean_absolute_error(y_test_true_orig, y_test_pred_orig)
    test_rmse = np.sqrt(mean_squared_error(y_test_true_orig, y_test_pred_orig))
    test_mape = (
        np.mean(np.abs((y_test_true_orig - y_test_pred_orig) / y_test_true_orig)) * 100
    )
    test_r2 = r2_score(y_test_true_orig, y_test_pred_orig)

    print(f"\n=== Training Set Metrics ===")
    print(f"MAE:  {train_mae:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAPE: {train_mape:.2f}%")
    print(f"R²:   {train_r2:.4f}")

    print(f"\n=== Validation Set Metrics ===")
    print(f"MAE:  {val_mae:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"MAPE: {val_mape:.2f}%")
    print(f"R²:   {val_r2:.4f}")

    print(f"\n=== Test Set Metrics ===")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    print(f"R²:   {test_r2:.4f}")

    with open(f"{OUTPUT_DIR}/metrics.txt", "w") as f:
        f.write(f"=== Training Set ===\n")
        f.write(f"MAE: {train_mae:.4f}\n")
        f.write(f"RMSE: {train_rmse:.4f}\n")
        f.write(f"MAPE: {train_mape:.2f}%\n")
        f.write(f"R2: {train_r2:.4f}\n\n")
        f.write(f"=== Validation Set ===\n")
        f.write(f"MAE: {val_mae:.4f}\n")
        f.write(f"RMSE: {val_rmse:.4f}\n")
        f.write(f"MAPE: {val_mape:.2f}%\n")
        f.write(f"R2: {val_r2:.4f}\n\n")
        f.write(f"=== Test Set ===\n")
        f.write(f"MAE: {test_mae:.4f}\n")
        f.write(f"RMSE: {test_rmse:.4f}\n")
        f.write(f"MAPE: {test_mape:.2f}%\n")
        f.write(f"R2: {test_r2:.4f}\n")
    print(f"Metrics saved to {OUTPUT_DIR}/metrics.txt")

    dates_all = df["日期"].values[SEQ_LEN:]
    dates_train = dates_all[:train_size]
    dates_val = dates_all[train_size : train_size + val_size]
    dates_test = dates_all[train_size + val_size :]

    plot_results(
        y_train_true_orig,
        y_train_pred_orig,
        y_val_true_orig,
        y_val_pred_orig,
        y_test_true_orig,
        y_test_pred_orig,
        dates_train,
        dates_val,
        dates_test,
        OUTPUT_DIR,
    )
    plot_loss_curves(train_losses, val_losses, OUTPUT_DIR)
    plot_full_comparison(
        y_train_true_orig,
        y_train_pred_orig,
        y_val_true_orig,
        y_val_pred_orig,
        y_test_true_orig,
        y_test_pred_orig,
        dates_all,
        train_size,
        val_size,
        OUTPUT_DIR,
    )

    print("\n=== Done! ===")
    print(f"All outputs saved to {OUTPUT_DIR}/")
