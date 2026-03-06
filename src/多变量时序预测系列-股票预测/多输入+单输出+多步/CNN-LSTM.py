import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

# 1. 构造时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=30, pred_len=5):
        super(TimeSeriesDataset, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(df) - seq_len - pred_len + 1
        self.data, self.targets = self.generate_data(df)

    def generate_data(self, df):
        window_size = self.seq_len

        data, targets = list(), list()
        for window_start in range(len(df)):
            past_end = window_start + window_size
            future_end = past_end + self.pred_len
            if future_end > len(df):
                break
            # slicing the past and future parts of the window
            past, future = df[window_start:past_end, :], df[past_end:future_end, 0:1]
            data.append(past)
            targets.append(future)

        data = np.array(data, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        print(data.shape, targets.shape)

        return data, targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


# 2. 定义 CNN-LSTM 模型（去掉 Transformer）
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim=5, cnn_channels=64, lstm_hidden=128, 
                 lstm_layers=2, pred_len=5, dropout=0.2):
        super().__init__()
        
        # CNN 特征提取层
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout/2),
            
            # 第二层 CNN（可选，增强特征提取能力）
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels*2),
            nn.Dropout(dropout/2)
        )
        
        # LSTM 时序建模层
        self.lstm = nn.LSTM(
            input_size=cnn_channels*2, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False  # 可以设为 True 使用双向LSTM
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden//2, pred_len)
        )
        
        self.pred_len = pred_len
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # CNN 特征提取: [B, T, C] -> [B, C, T] -> [B, C', T]
        x = x.transpose(1, 2)  # [B, input_dim, T]
        cnn_out = self.cnn(x)  # [B, cnn_channels*2, T]
        cnn_out = cnn_out.transpose(1, 2)  # [B, T, cnn_channels*2]
        
        # LSTM 时序建模: [B, T, features] -> [B, T, hidden]
        lstm_out, (hidden, cell) = self.lstm(cnn_out)
        
        # 取最后一个时间步的隐藏状态进行预测
        # 或者使用所有时间步的 attention（这里简化取最后一步）
        last_hidden = lstm_out[:, -1, :]  # [B, hidden]
        
        # 全连接层输出预测
        out = self.fc(last_hidden)  # [B, pred_len]
        
        return out.unsqueeze(-1)  # [B, pred_len, 1]



# 3. 训练、评估与可视化函数
def train_model(model, dataloader, num_epochs=50, learning_rate=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()

    loss_history = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_data, batch_targets in dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return loss_history

def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch_data, batch_targets in dataloader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_targets.cpu().numpy())

    # print(preds.shape)
    # print(trues.shape)
    preds = np.concatenate(preds, axis=0).squeeze()
    trues = np.concatenate(trues, axis=0).squeeze()
    print(preds.shape)
    print(trues.shape)
    return preds, trues

def visualize_results(loss_history, preds, trues):
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)

    # 图 1：训练损失曲线
    # 模型在训练过程中损失的下降情况，说明模型不断优化拟合数据。
    # plt.plot(loss_history, marker='o', color='dodgerblue', linestyle='-', linewidth=2)
    # plt.title("Training Loss Curve")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE Loss")
    # plt.tight_layout()
    # plt.savefig('output_image1.png', dpi=300, format='png')
    # plt.show()

    # 图 2：真实值与预测值对比曲线
    # 对比曲线直观展示模型预测趋势与真实数据的匹配情况，越接近表示模型效果越好。
    plt.plot(trues, label="True Values", color='limegreen')
    plt.plot(preds, label="Predicted Values", color='crimson')
    plt.title("True vs. Predicted Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Trend Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_image2.png', dpi=300, format='png')
    plt.show()

def visualize_results_v2(idx, preds, trues):
    # 图：测试集 单样本预测 vs 真值（多步）
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)

    vivid_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
                    '#ff7f00','#ffff33','#a65628','#f781bf',
                    '#17becf','#d62728']

    idx = idx
    plt.figure(figsize=(10,6))
    plt.plot(range(1, pred_len+1), trues[idx], marker='o', linewidth=3, label="True", color=vivid_colors[1])
    plt.plot(range(1, pred_len+1), preds[idx], marker='s', linewidth=3, label="Pred", color=vivid_colors[0])
    plt.fill_between(range(1,pred_len+1), preds[idx]-0.05, preds[idx]+0.05, color=vivid_colors[0], alpha=0.15, label="±0.05 band")
    plt.title("comparison of multi-step predictions on test samples")
    plt.xlabel("prediction step(horizon)")
    plt.ylabel("numerical value")
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_image3.png', dpi=300, format='png')
    plt.show()

    # 图：残差分布与 KDE
    resid = (preds - trues).reshape(-1)
    sns.histplot(resid, bins=40, stat="density", color=vivid_colors[3], kde=True, alpha=0.7)
    plt.axvline(np.mean(resid), color=vivid_colors[0], linestyle='--', linewidth=2, label=f"Mean={np.mean(resid):.3f}")
    plt.title("residual distribution and KDE")
    plt.xlabel("residual")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_image4.png', dpi=300, format='png')
    plt.show()

    # 图：Rolling MAPE（窗口=50 个样本）
    win = 50
    mape_series = np.abs((preds - trues)/(np.abs(trues)+1e-8)).mean(axis=1)
    rolling = np.array([mape_series[i:i+win].mean() for i in range(0, len(mape_series)-win+1)])
    plt.figure(figsize=(12,5))
    plt.plot(rolling, linewidth=3, color=vivid_colors[6])
    plt.title(f"Rolling MAPE (window={win})")
    plt.xlabel("start sample index of window")
    plt.ylabel("MAPE")
    plt.tight_layout()
    plt.savefig('output_image5.png', dpi=300, format='png')
    plt.show()


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

# 5. 主函数：数据加载、模型训练、评估与可视化
if __name__ == '__main__':
    # 设置随机种子，保证结果可重复
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数设置
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-3
    seq_len = 30
    pred_len = 5

    # 数据加载
    df = pd.read_csv('~/snap/wenqiang/Data-Miscellany-Forum/src/多变量时序预测系列-股票预测/多输入+单输出+多步/data.csv', parse_dates=["Date"], index_col=[0])
    df = pd.DataFrame(df)
    var_num = len(df.columns)
    print(f"Data shape: {df.shape}")
    print(df.head())

    # 数据划分
    test_split=round(len(df)*0.20)
    df_for_training=df[:-test_split]
    df_for_testing=df[-test_split:]
    print(f"Training samples: {len(df_for_training)}, Testing samples: {len(df_for_testing)}")

    # 数据归一化处理
    scaler = MinMaxScaler(feature_range=(0,1))
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    df_for_testing_scaled=scaler.transform(df_for_testing)

    # 构造时序数据集
    train_dataset = TimeSeriesDataset(df_for_training_scaled, seq_len=seq_len, pred_len=pred_len)
    test_dataset = TimeSeriesDataset(df_for_testing_scaled, seq_len=seq_len, pred_len=pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    model = CNN_LSTM(
        input_dim=var_num,
        cnn_channels=64,
        lstm_hidden=128,
        lstm_layers=2,
        pred_len=pred_len,
        dropout=0.2
    ).to(device)
    print(f"\nModel architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("开始训练模型...")
    loss_history = train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

    # 在测试集上进行评估
    preds, trues = evaluate_model(model, test_loader, device=device)
    y_true = trues.reshape(-1)
    y_pred = preds.reshape(-1)
    preds_copies_array = np.repeat(y_pred, var_num, axis=-1)
    preds_test=scaler.inverse_transform(np.reshape(preds_copies_array, (len(y_pred), int(var_num))))[:,0]
    preds_test_2d = preds_test.reshape(int(len(y_true))//int(var_num), int(var_num))

    trues_copies_array = np.repeat(y_true, var_num, axis=-1)
    trues_test=scaler.inverse_transform(np.reshape(trues_copies_array, (len(y_true), int(var_num))))[:,0]
    trues_test_2d = trues_test.reshape(int(len(y_true))//int(var_num), int(var_num))

    # 可视化结果
    visualize_results(loss_history, preds_test, trues_test)
    # visualize_results_v2(0, preds_test_2d, trues_test_2d)

    # 计算误差
    metrics = evaluate_metrics(trues_test, preds_test)

    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")