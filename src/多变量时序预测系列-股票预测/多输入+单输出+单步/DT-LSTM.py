import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
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
        # targets = targets.reshape(-1, 1, 1) # shape: (num_samples, pred_len, fea_num)
        print(data.shape, targets.shape)

        return data, targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


# 2. 构建LSTM模型
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=1, output_size=1, dropout=0.2):
        """
        input_size: 输入特征维度（多变量个数）
        hidden_size: LSTM隐藏层维度
        num_layers: LSTM层数
        """
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size

        # LSTM层
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True)


        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        

    def forward(self, x):
        """
        x: (batch_size, seq_length, input_dim)
        return: (batch_size, output_dim)
        """
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch, seq, hidden)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)取最后一个时间步的隐藏状态（或可用 hidden[-1]）
        out = self.fc(last_hidden)  # (batch, output_dim)
        return out


# 3. 训练、评估与可视化函数
def train_model(model, dataloader, num_epochs=50, learning_rate=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()

    preds = []
    trues = []

    loss_history = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_data, batch_targets in dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_targets.detach().cpu().numpy())
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    preds = np.concatenate(preds, axis=0).squeeze()
    trues = np.concatenate(trues, axis=0).squeeze()
    return loss_history, preds, trues


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


def visualize_results(loss_history, preds, final_preds, trues):
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)

    # # 图 1：训练损失曲线
    # # 模型在训练过程中损失的下降情况，说明模型不断优化拟合数据。
    # plt.plot(loss_history, marker='o', color='dodgerblue', linestyle='-', linewidth=2)
    # plt.title("Training Loss Curve")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE Loss")
    # plt.tight_layout()
    # plt.savefig('output_image1.png', dpi=300, format='png')
    # plt.show()

    # # 图 2：真实值与预测值对比曲线(LSTM)
    # # 对比曲线直观展示模型预测趋势与真实数据的匹配情况，越接近表示模型效果越好。
    # plt.plot(trues, label="True Values", color='limegreen')
    # plt.plot(preds, label="Predicted Values", color='crimson')
    # plt.title("True vs. Predicted Values")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Trend Value")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('output_image2.png', dpi=300, format='png')
    # plt.show()


    # # 图 3：真实值与预测值对比曲线(融合模型)
    # # 对比曲线直观展示模型预测趋势与真实数据的匹配情况，越接近表示模型效果越好。
    # plt.plot(trues, label="True Values", color='limegreen')
    # plt.plot(final_preds, label="Predicted Values", color='crimson')
    # plt.title("True vs. Predicted Values")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Trend Value")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('output_image3.png', dpi=300, format='png')
    # plt.show()


    # # 图 4：残差分布图（增强前后）
    # sns.histplot(trues - preds, color='red', label='LSTM Residuals', kde=True)
    # sns.histplot(trues - final_preds, color='green', label='Hybrid Residuals', kde=True)
    # plt.title("Residual Distribution: LSTM vs Hybrid")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('output_image4.png', dpi=300, format='png')
    # plt.show()

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
    num_epochs = 20
    learning_rate = 1e-3
    seq_len = 30
    pred_len = 1
    hidden_size = 50
    num_layers = 1
    dropout = 0.2

    # 数据加载
    df = pd.read_csv('/workspaces/Data-Miscellany-Forum/src/多变量时序预测系列-股票预测/多输入+单输出+单步/data.csv', parse_dates=["Date"], index_col=[0])
    df = pd.DataFrame(df)
    var_num = len(df.columns)
    print(df)

    # 数据划分
    test_split=round(len(df)*0.20)
    df_for_training=df[:-test_split]
    df_for_testing=df[-test_split:]

    # 数据归一化处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    df_for_testing_scaled=scaler.transform(df_for_testing)

    # 构造时序数据集
    train_dataset = TimeSeriesDataset(df_for_training_scaled, seq_len=seq_len, pred_len=pred_len)
    test_dataset = TimeSeriesDataset(df_for_testing_scaled, seq_len=seq_len, pred_len=pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    model = LSTMRegressor(input_size=var_num, hidden_size=hidden_size, num_layers=num_layers, output_size=pred_len, dropout=dropout).to(device)
    print("开始训练模型...")
    loss_history, train_preds, train_trues = train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)
    # 决策树基于LSTM残差训练
    residuals = np.array(train_trues).reshape(-1, 1)- np.array(train_preds).reshape(-1, 1)
    X_tree_train = train_preds.reshape(-1, 1)
    model_tree = DecisionTreeRegressor(max_depth=3)
    model_tree.fit(X_tree_train, residuals)

    # 在测试集上进行评估
    preds, trues = evaluate_model(model, test_loader, device=device)
    X_tree_test = preds.reshape(-1, 1)
    residual_preds = model_tree.predict(X_tree_test)
    final_preds = preds + residual_preds

    preds_copies_array = np.repeat(preds, var_num, axis=-1)
    preds_test=scaler.inverse_transform(np.reshape(preds_copies_array, (len(preds), var_num)))[:,0]

    final_preds_copies_array = np.repeat(final_preds, var_num, axis=-1)
    final_preds_test=scaler.inverse_transform(np.reshape(final_preds_copies_array, (len(final_preds), var_num)))[:,0]

    trues_copies_array = np.repeat(trues, var_num, axis=-1)
    trues_test=scaler.inverse_transform(np.reshape(trues_copies_array, (len(trues), var_num)))[:,0]

    # 可视化结果
    # visualize_results(loss_history, preds_test, final_preds_test, trues_test)

    # 计算误差
    metrics = evaluate_metrics(final_preds_test, trues_test)
    print("Test metrics:", metrics)




