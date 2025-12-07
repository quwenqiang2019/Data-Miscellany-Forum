import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 全局设置
device = torch.device("cuda"if torch.cuda.is_available() else"cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
vivid_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#17becf','#d62728']

# 2. 读取多变量时间序列数据
def split_series(series, n_past, n_future):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, 1]
        X.append(past)
        y.append(future)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def generate_series(T_in, H):
    # 读取数据
    df = pd.read_csv("data.csv", parse_dates=["Date"], index_col=[0])
    # df_for_scaled = df.values
    scaler = MinMaxScaler(feature_range=(0,1))
    df_for_scaled = scaler.fit_transform(df)
    # 假设给定过去 10 天的观察结果，我们需要预测接下来的 3 天观察结果
    n_past = T_in
    n_future = H
    # # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    X, Y = split_series(df_for_scaled, n_past, n_future)

    return X, Y

class TSForecastDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  # (N, T_in, d_in)
        self.Y = torch.from_numpy(Y)  # (N, H)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 生成数据集
T_in, H = 10, 3
X, Y = generate_series(T_in, H)
N = 5191
d_in = 5

# 划分训练/验证/测试（时间顺序）
n_train = int(N*0.7)
n_val = int(N*0.15)
X_train, Y_train = X[:n_train], Y[:n_train]
X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]

train_ds = TSForecastDataset(X_train, Y_train)
val_ds   = TSForecastDataset(X_val, Y_val)
test_ds  = TSForecastDataset(X_test, Y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

# 3. 模型定义
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div)
        pe[:, 1::2] = torch.cos(position*div)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model) for (T,B,E)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (T,B,E)
        T = x.size(0)
        x = x + self.pe[:T]
        return self.dropout(x)

class TransformerEncoderLayerWithAttn(nn.Module):
    """
    自定义 Transformer Encoder Layer，可返回注意力权重，便于可视化。
    """
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._last_attn = None# 缓存最近一次的注意力

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=False):
        # src: (T,B,E)
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False
        )
        if need_weights:
            self._last_attn = attn_weights  # shape: (B, nhead, T, T) in PyTorch>=2
        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)
        ff = self.linear2(self.dropout(self.act(self.linear1(src))))
        src2 = self.dropout2(ff)
        src = self.norm2(src + src2)
        return src

    def get_last_attn(self):
        return self._last_attn

class AttentionPooling(nn.Module):
    """
    对时序特征 H (B,T,D) 进行可学习注意力池化，输出 (B,D)。
    """
    def __init__(self, d_in, d_hidden=None):
        super().__init__()
        d_hidden = d_hidden or d_in
        self.proj = nn.Linear(d_in, d_hidden)
        self.tanh = nn.Tanh()
        self.context = nn.Parameter(torch.randn(d_hidden))  # learnable query

    def forward(self, H):
        # H: (B,T,D)
        score = torch.matmul(self.tanh(self.proj(H)), self.context)  # (B,T)
        alpha = torch.softmax(score, dim=1)  # (B,T)
        z = torch.sum(H * alpha.unsqueeze(-1), dim=1)  # (B,D)
        return z, alpha

class FusionLSTM_BiLSTM_Transformer(nn.Module):
    def __init__(self, d_in, d_model=64, lstm_hidden=64, bilstm_hidden=48,
                 nhead=4, num_trans_layers=2, ff_dim=128, dropout=0.1, horizon=10):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(dropout)

        # 分支 A: LSTM
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True, bidirectional=False,
                            num_layers=1, dropout=dropout)

        # 分支 B: BiLSTM
        self.bilstm = nn.LSTM(d_model, bilstm_hidden, batch_first=True, bidirectional=True,
                              num_layers=1, dropout=dropout)

        # 分支 C: Transformer Encoder
        self.posenc = PositionalEncoding(d_model, dropout=dropout)
        self.trans_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model, nhead, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_trans_layers)
        ])

        # 注意力池化
        self.pool_lstm = AttentionPooling(lstm_hidden)
        self.pool_bilstm = AttentionPooling(bilstm_hidden*2)
        self.pool_trans = AttentionPooling(d_model)

        # 融合头
        fused_dim = lstm_hidden + bilstm_hidden*2 + d_model
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, horizon)
        )

    def forward(self, x, need_attn=False):
        # x: (B,T_in,d_in)
        B,T,_ = x.shape
        z = self.dropout(torch.relu(self.embed(x)))  # (B,T,d_model)

        # A: LSTM
        out_lstm, _ = self.lstm(z)  # (B,T,h_lstm)
        z_lstm, alpha_lstm = self.pool_lstm(out_lstm)

        # B: BiLSTM
        out_bilstm, _ = self.bilstm(z)  # (B,T,2*h_bilstm)
        z_bilstm, alpha_bilstm = self.pool_bilstm(out_bilstm)

        # C: Transformer
        # 转为 (T,B,E)
        z_trans = z.transpose(0,1)  # (T,B,E)
        z_trans = self.posenc(z_trans)
        attn_maps = None
        for i, layer in enumerate(self.trans_layers):
            z_trans = layer(z_trans, need_weights=need_attn and (i==len(self.trans_layers)-1))
            if need_attn and (i==len(self.trans_layers)-1):
                attn_maps = layer.get_last_attn()  # (B, nhead, T, T)
        out_trans = z_trans.transpose(0,1)  # (B,T,E)
        z_trans_ctx, alpha_trans = self.pool_trans(out_trans)

        # 融合
        z_fuse = torch.cat([z_lstm, z_bilstm, z_trans_ctx], dim=-1)  # (B, fused_dim)
        yhat = self.head(z_fuse)  # (B,H)

        aux = {
            "alpha_lstm": alpha_lstm,         # (B,T)
            "alpha_bilstm": alpha_bilstm,     # (B,T)
            "alpha_trans": alpha_trans,       # (B,T)
            "attn_maps": attn_maps            # (B, nhead, T, T) or None
        }
        return yhat, aux

# 4. 训练与评估工具
def smape_loss(y_true, y_pred, eps=1e-6):
    return (2.0 * torch.abs(y_pred - y_true) / (torch.abs(y_true) + torch.abs(y_pred) + eps)).mean()

def combined_loss(y_true, y_pred, alpha=0.5, beta=0.4, gamma=0.1):
    mse = ((y_true - y_pred)**2).mean()
    mae = (y_true - y_pred).abs().mean()
    smape = smape_loss(y_true, y_pred)
    return alpha*mse + beta*mae + gamma*smape, mse.item(), mae.item(), smape.item()

def evaluate_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    eps = 1e-8
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse + eps)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/(np.abs(y_true)+eps)))
    smape = np.mean(2*np.abs(y_true - y_pred)/(np.abs(y_true)+np.abs(y_pred)+eps))
    # R2
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + eps
    r2 = 1 - ss_res/ss_tot
    # Pearson
    corr = np.corrcoef(y_true, y_pred)[0,1]
    return dict(MSE=mse, RMSE=rmse, MAE=mae, MAPE=mape, sMAPE=smape, R2=r2, Corr=corr)

# 5. 训练主循环
model = FusionLSTM_BiLSTM_Transformer(
    d_in=d_in, d_model=64, lstm_hidden=64, bilstm_hidden=48,
    nhead=4, num_trans_layers=2, ff_dim=128, dropout=0.1, horizon=H
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
max_epochs = 10
grad_clip = 1.0
best_val = float('inf')
ckpt_path = "fusion_ts_best.pt"

history = {"train_loss": [], "val_loss": []}

for epoch in range(1, max_epochs+1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        yhat, _ = model(xb, need_attn=False)
        loss, mse, mae, sm = combined_loss(yb, yhat, alpha=0.5, beta=0.4, gamma=0.1)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # 验证
    model.eval()
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat, _ = model(xb, need_attn=False)
            loss, _, _, _ = combined_loss(yb, yhat)
            val_total += loss.item() * xb.size(0)
    val_loss = val_total / len(val_loader.dataset)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), ckpt_path)

    print(f"Epoch [{epoch}/{max_epochs}] TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} LR={scheduler.get_last_lr()[0]:.6f}")

# 加载最佳模型
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# 6. 测试与可视化
# 收集测试预测
y_true_all, y_pred_all = [], []
attn_example = None
alpha_trans_example = None
x_example = None
with torch.no_grad():
    for i, (xb, yb) in enumerate(test_loader):
        xb = xb.to(device)
        yb = yb.to(device)
        need_attn = (i == 0)  # 仅在第一个 batch 记录注意力
        yhat, aux = model(xb, need_attn=need_attn)
        y_true_all.append(yb.cpu().numpy())
        y_pred_all.append(yhat.cpu().numpy())
        if need_attn:
            if aux["attn_maps"] is not None:
                attn_example = aux["attn_maps"].cpu().numpy()  # (B, nhead, T, T)
            alpha_trans_example = aux["alpha_trans"].cpu().numpy()
            x_example = xb.cpu().numpy()

y_true_all = np.concatenate(y_true_all, axis=0)  # (N_test, H)
y_pred_all = np.concatenate(y_pred_all, axis=0)  # (N_test, H)

metrics = evaluate_metrics(y_true_all, y_pred_all)
print("Test metrics:", metrics)

# 图1：训练/验证损失曲线
plt.figure(figsize=(10,6))
plt.plot(history["train_loss"], label="Train Loss", linewidth=3)
# plt.plot(history["val_loss"], label="Val Loss", linewidth=3)
plt.title("损失曲线（融合模型）")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# 图2：测试集单样本预测 vs 真值（多步）
idx = 0
plt.figure(figsize=(10,6))
plt.plot(range(1, H+1), y_true_all[idx], marker='o', linewidth=3, label="True", color=vivid_colors[1])
plt.plot(range(1, H+1), y_pred_all[idx], marker='s', linewidth=3, label="Pred", color=vivid_colors[0])
plt.fill_between(range(1,H+1), y_pred_all[idx]-0.05, y_pred_all[idx]+0.05, color=vivid_colors[0], alpha=0.15, label="±0.05 band")
plt.title("测试样本多步预测对比")
plt.xlabel("预测步 (horizon)")
plt.ylabel("数值")
plt.legend()
plt.tight_layout()
plt.show()

# 图3：残差分布与 KDE
resid = (y_pred_all - y_true_all).reshape(-1)
plt.figure(figsize=(10,6))
sns.histplot(resid, bins=40, stat="density", color=vivid_colors[3], kde=True, alpha=0.7)
plt.axvline(np.mean(resid), color=vivid_colors[0], linestyle='--', linewidth=2, label=f"Mean={np.mean(resid):.3f}")
plt.title("残差分布与 KDE")
plt.xlabel("残差")
plt.ylabel("密度")
plt.legend()
plt.tight_layout()
plt.show()

# 图4：Rolling MAPE（窗口=50 个样本）
win = 50
mape_series = np.abs((y_pred_all - y_true_all)/(np.abs(y_true_all)+1e-8)).mean(axis=1)
rolling = np.array([mape_series[i:i+win].mean() for i in range(0, len(mape_series)-win+1)])
plt.figure(figsize=(12,5))
plt.plot(rolling, linewidth=3, color=vivid_colors[6])
plt.title(f"Rolling MAPE (窗口={win})")
plt.xlabel("窗口起始样本索引")
plt.ylabel("MAPE")
plt.tight_layout()
plt.show()

# 图5：Transformer 自注意力热力图（取第一批第一个样本，取平均头）
if attn_example is not None:
    # attn_example: (B, nhead, T, T)
    attn_sample = attn_example[0]  # (nhead, T, T)
    attn_avg = attn_sample.mean(axis=0)  # (T, T)
    plt.figure(figsize=(8,7))
    sns.heatmap(attn_avg, cmap="magma", square=True, cbar=True)
    plt.title("Transformer 自注意力热力图（各头平均）")
    plt.xlabel("Key/Source 时间步")
    plt.ylabel("Query 时间步")
    plt.tight_layout()
    plt.show()

# 额外：置换重要性
def permutation_importance(model, X_base, Y_base, metric_fn, repeats=1):
    """
    简易置换重要性：打乱每个特征，测试 MAPE 恶化程度。
    X_base: (N, T, d)
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_base).to(device)
        Y_t = torch.from_numpy(Y_base).to(device)
        yhat, _ = model(X_t, need_attn=False)
        base = evaluate_metrics(Y_t.cpu().numpy(), yhat.cpu().numpy())["MAPE"]

    d = X_base.shape[-1]
    imp = []
    for j in range(d):
        scores = []
        for _ in range(repeats):
            X_perm = X_base.copy()
            # 对第 j 个特征沿 samples 维打乱（保持时间结构）
            perm_idx = np.random.permutation(X_perm.shape[0])
            X_perm[:, :, j] = X_perm[perm_idx, :, j]
            with torch.no_grad():
                yhat_p, _ = model(torch.from_numpy(X_perm).to(device), need_attn=False)
                m = evaluate_metrics(Y_base, yhat_p.cpu().numpy())["MAPE"]
            scores.append(m)
        imp.append(np.mean(scores) - base)
    return imp, base

imp, base_mape = permutation_importance(model, X_test, Y_test, evaluate_metrics, repeats=1)
plt.figure(figsize=(10,6))
feat_names = [f"feat{i+1}"for i in range(d_in)]
sns.barplot(x=imp, y=feat_names, palette="viridis")
plt.axvline(0, color="k", lw=2)
plt.title(f"置换重要性（越大越重要） | 基线MAPE={base_mape:.4f}")
plt.xlabel("MAPE 恶化量")
plt.ylabel("特征")
plt.tight_layout()
plt.show()


# 图6：输入序列 + 预测序列 + 真实序列完整趋势
# 选取测试集第一个样本
idx = 5
x_input = x_example[idx]      # (T_in, d_in)
y_true = y_true_all[idx]      # (H,)
y_pred = y_pred_all[idx]      # (H,)

plt.figure(figsize=(14,6), facecolor="#1e1e2f")
ax = plt.gca()
ax.set_facecolor("#1e1e2f") 
ax.grid(True, color='#44475a', linestyle='--', linewidth=0.8)

# 输入序列（主通道）
plt.plot(range(1, T_in+1), x_input[:,0], marker='o', linestyle='-', linewidth=2, label="Input (历史)", color="#ff6e6e")
# 真实未来
plt.plot(range(T_in+1, T_in+H+1), y_true, marker='s', linestyle='-', linewidth=2, label="True (未来)", color="#50fa7b")
# 预测未来
plt.plot(range(T_in+1, T_in+H+1), y_pred, marker='^', linestyle='--', linewidth=2, label="Predicted (未来)", color="#8be9fd")
# 预测误差区间 ±0.05
plt.fill_between(range(T_in+1, T_in+H+1), y_pred-0.05, y_pred+0.05, color="#8be9fd", alpha=0.2)

plt.title("时间序列预测示例（历史输入 + 未来预测）", fontsize=16, fontweight='bold', color='white')
plt.xlabel("时间步", fontsize=12, color='white')
plt.ylabel("数值", fontsize=12, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.legend(facecolor="#2e2e3e", edgecolor='white')
plt.tight_layout()
plt.show()