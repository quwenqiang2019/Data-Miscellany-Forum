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
    def __init__(self, df, seq_len=30, pred_len=1):
        super(TimeSeriesDataset, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(df) - seq_len - pred_len + 1
        self.data, self.targets = self.generate_data(df)

    def generate_data(self, df):
        window_size = self.seq_len
        data = []
        targets = []

        for i in range(len(df) - window_size):
            data.append(df[i:i + window_size, 0:df.shape[1]])
            targets.append(df[i + window_size, 0])

        data = np.array(data, dtype=np.float32)  # shape: (num_samples, seq_len, fea_num)
        targets = np.array(targets, dtype=np.float32).reshape(-1, 1) # shape: (num_samples, pred_len)
        targets = targets.reshape(-1, 1, 1) # shape: (num_samples, pred_len, fea_num)
        print(data.shape, targets.shape)

        return data, targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


# 2. 定义融合模型：CNN + LSTM + Transformer
class CNN_LSTM_Transformer(nn.Module):
    def __init__(self, input_dim=5, cnn_channels=16, lstm_hidden=32, transformer_dim=32,
                 transformer_heads=4, transformer_layers=1, pred_len=1):
        super().__init__()
        # CNN
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.cnn_relu = nn.ReLU()
        
        # LSTM
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, batch_first=True)
        
        # Transformer Encoder 
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Projection layers
        self.proj_lstm = nn.Linear(lstm_hidden, transformer_dim)
        self.pred_len = pred_len
        self.fc_out = nn.Linear(transformer_dim, pred_len)
    
    def forward(self, x):
        # x: [batch, seq_len, 1]
        batch_size, seq_len, _ = x.shape
        # CNN expects [batch, channels, seq_len]
        cnn_out = self.cnn_relu(self.cnn(x.transpose(1,2)))  # [B, C, T]
        cnn_out = cnn_out.transpose(1,2)  # [B, T, C]
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # [B, T, hidden]
        lstm_proj = self.proj_lstm(lstm_out)  # [B, T, transformer_dim]
        # Transformer
        trans_out = self.transformer(lstm_proj)  # [B, T, transformer_dim]
        # 取最后时间步输出预测
        out = self.fc_out(trans_out[:, -1, :])  # [B, pred_len]
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

    preds = np.concatenate(preds, axis=0).squeeze()
    trues = np.concatenate(trues, axis=0).squeeze()

    return preds, trues

def visualize_results(loss_history, preds, trues):
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)

    # 图 1：训练损失曲线
    # 模型在训练过程中损失的下降情况，说明模型不断优化拟合数据。
    plt.plot(loss_history, marker='o', color='dodgerblue', linestyle='-', linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig('output_image1.png', dpi=300, format='png')
    plt.show()

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


# 5. 主函数：数据加载、模型训练、评估与可视化
if __name__ == '__main__':
    # 设置随机种子，保证结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数设置
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3
    seq_len = 30
    pred_len = 1

    # 数据加载
    df = pd.read_csv('/workspaces/Data-Miscellany-Forum/src/多变量时序预测系列-股票预测/多输入+单输出+单步/data.csv', parse_dates=["Date"], index_col=[0])
    df = pd.DataFrame(df)
    print(df)

    # 数据划分
    test_split=round(len(df)*0.20)
    df_for_training=df[:-test_split]
    df_for_testing=df[-test_split:]

    # 数据归一化处理
    scaler = MinMaxScaler(feature_range=(0,1))
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    df_for_testing_scaled=scaler.transform(df_for_testing)

    # 构造时序数据集
    train_dataset = TimeSeriesDataset(df_for_training_scaled, seq_len=30, pred_len=1)
    test_dataset = TimeSeriesDataset(df_for_testing_scaled, seq_len=30, pred_len=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_LSTM_Transformer().to(device)
    print("开始训练模型...")
    loss_history = train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

    # 在测试集上进行评估
    preds, trues = evaluate_model(model, test_loader, device=device)

    preds_copies_array = np.repeat(preds, 5, axis=-1)
    preds_test=scaler.inverse_transform(np.reshape(preds_copies_array, (len(preds), 5)))[:,0]

    trues_copies_array = np.repeat(trues, 5, axis=-1)
    trues_test=scaler.inverse_transform(np.reshape(trues_copies_array, (len(trues), 5)))[:,0]

    # 可视化结果
    visualize_results(loss_history, preds_test, trues_test)

    # 计算误差
    testScore1 = math.sqrt(mean_squared_error(preds_test, trues_test))
    print('Test Score: %.2f RMSE' % (testScore1))
    testScore2 = mean_absolute_error(preds_test, trues_test)
    print('Test Score: %.2f MAE' % (testScore2))
    testScore3 = r2_score(preds_test, trues_test)
    print('Test Score: %.2f R2' % (testScore3))
    testScore4 = mean_absolute_percentage_error(preds_test, trues_test)
    print('Test Score: %.2f MAPE' % (testScore4))