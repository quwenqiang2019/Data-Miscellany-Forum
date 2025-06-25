import math
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 1. 构造时间序列数据集
class VirtualTimeSeriesDataset(Dataset):
    def __init__(self, seq_len=12, num_samples=132):
        super(VirtualTimeSeriesDataset, self).__init__()
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data, self.targets = self.generate_data()


    def generate_data(self):
        window_size = self.seq_len

        data = []
        targets = []
        # 读取数据集
        # df = pd.read_csv('data.csv')
        df = pd.read_excel('_data.xlsx')
        # 将日期列转换为日期时间类型
        df['Month'] = pd.to_datetime(df['Month'])
        # 将日期列设置为索引
        df.set_index('Month', inplace=True)
        # scaler = MinMaxScaler()
        print("-----")
        print(df.values.reshape(-1, 1))
        df = scaler.fit_transform(df.values.reshape(-1, 1))
        print(df)

        for i in range(len(df) - window_size):
            data.append(df[i:i + window_size, 0:df.shape[1]])
            targets.append(df[i + window_size, 0])


        data = np.array(data, dtype=np.float32)  # shape: (num_samples, seq_len, 1)
        targets = np.array(targets, dtype=np.float32).reshape(-1, 1)
        print(data.shape, targets.shape)

        return data, targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


# 2. 位置编码模块（Transformer 必需）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 使用公式进行计算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维使用 cos
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


# 3. 融合 Transformer 与 CNN 的模型定义
class FusedCNNTransformer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, cnn_channels=64, kernel_size=3,
                 num_transformer_layers=2, nhead=4, dropout=0.1, fc_dim=32):
        super(FusedCNNTransformer, self).__init__()
        self.embed_dim = embed_dim
        # 线性嵌入层：将输入数据映射到 embed_dim 维度
        self.embedding = nn.Linear(input_dim, embed_dim)

        # CNN 分支：提取局部特征
        # 输入需变换为 (batch, channels, seq_len) 形式
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=cnn_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 压缩时间维度，输出 shape (batch, cnn_channels, 1)
        )

        # Transformer 分支：捕捉全局依赖
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)
        # 对 Transformer 输出进行池化：取最后时刻或均值池化
        self.transformer_pool = nn.AdaptiveAvgPool1d(1)  # 最终输出 (batch, embed_dim, 1)

        # 融合层（门控机制）
        # 将 CNN 与 Transformer 特征拼接后计算门控系数
        self.gate_fc = nn.Linear(cnn_channels + embed_dim, cnn_channels + embed_dim)

        # 最终预测层
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels + embed_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 1)  # 输出回归值，预测下一个价格/趋势
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        # 线性嵌入
        x_embed = self.embedding(x)  # [batch, seq_len, embed_dim]

        # Transformer 分支：添加位置编码并转置为 [seq_len, batch, embed_dim] 供 Transformer 使用
        x_pos = self.pos_encoder(x_embed)
        x_trans_input = x_pos.transpose(0, 1)  # [seq_len, batch, embed_dim]
        transformer_output = self.transformer_encoder(x_trans_input)  # [seq_len, batch, embed_dim]
        transformer_output = transformer_output.transpose(0, 1)  # [batch, seq_len, embed_dim]
        # 对 Transformer 输出进行池化，提取全局特征
        transformer_pool = self.transformer_pool(transformer_output.transpose(1, 2))  # [batch, embed_dim, 1]
        transformer_feature = transformer_pool.squeeze(-1)  # [batch, embed_dim]

        # CNN 分支：将嵌入后的数据转置为 [batch, embed_dim, seq_len] 进行一维卷积
        cnn_input = x_embed.transpose(1, 2)  # [batch, embed_dim, seq_len]
        cnn_feature = self.cnn(cnn_input)  # [batch, cnn_channels, 1]
        cnn_feature = cnn_feature.squeeze(-1)  # [batch, cnn_channels]

        # 融合：拼接两个分支的特征，然后计算门控系数
        fused = torch.cat([cnn_feature, transformer_feature], dim=1)  # [batch, cnn_channels+embed_dim]
        gate = torch.sigmoid(self.gate_fc(fused))  # 门控系数，形状同 fused
        fused_feature = gate * fused  # 简单使用门控系数对特征加权
        # 最终预测
        output = self.fc(fused_feature)  # [batch, 1]

        # 为了展示部分中间结果，这里返回 cnn_feature 与 transformer_feature（用于可视化）
        return output, cnn_feature, transformer_feature


# 4. 训练与可视化函数
def train_model(model, dataloader, num_epochs=50, learning_rate=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    loss_history = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_data, batch_targets in dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(batch_data)
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
    cnn_features = []
    trans_features = []
    with torch.no_grad():
        for batch_data, batch_targets in dataloader:
            batch_data = batch_data.to(device)
            outputs, cnn_f, trans_f = model(batch_data)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_targets.cpu().numpy())
            cnn_features.append(cnn_f.cpu().numpy())
            trans_features.append(trans_f.cpu().numpy())


    preds_reshaped = np.array(preds).reshape(-1, 1)
    trues_reshaped = np.array(trues).reshape(-1, 1)
    print(preds_reshaped, trues_reshaped)
    preds_original = scaler.inverse_transform(preds_reshaped)
    trues_original = scaler.inverse_transform(trues_reshaped)

    print(preds_original, trues_original)

    preds = np.concatenate(preds_original, axis=0).squeeze()
    trues = np.concatenate(trues_original, axis=0).squeeze()
    cnn_features = np.concatenate(cnn_features, axis=0)
    trans_features = np.concatenate(trans_features, axis=0)
    df = pd.DataFrame({"preds":preds, "trues":trues})
    print(df)
    return preds, trues, cnn_features, trans_features


def visualize_results(loss_history, preds, trues, cnn_features, trans_features):
    # 四个子图：训练损失曲线、真实值与预测值对比、CNN局部特征图、Transformer全局特征图
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)

    # 图 1：训练损失曲线
    # 模型在训练过程中损失的下降情况，说明模型不断优化拟合数据。
    plt.plot(loss_history, marker='o', linestyle='-', linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'figure1.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 图 2：真实值与预测值对比曲线
    # 对比曲线直观展示模型预测趋势与真实数据的匹配情况，越接近表示模型效果越好。
    plt.plot(trues, label="True Values",  marker='o')
    plt.plot(preds, label="Predicted Values", marker='x')
    plt.title("True vs. Predicted Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Trend Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'figure2.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 图 3：CNN 提取的局部特征展示（选取第一个样本的特征向量）
    # 展示 CNN 分支提取到的局部特征，反映了数据中局部模式信息的提取情况。
    plt.bar(np.arange(cnn_features.shape[1]), cnn_features[0])
    plt.title("CNN Local Features (Sample 1)")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'figure3.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 图 4：Transformer 提取的全局特征展示（选取第一个样本的特征向量）
    # 展示 Transformer 分支提取到的全局特征，体现了长距离依赖关系的建模效果。
    plt.plot(trans_features[0], marker='s', linestyle='-')
    plt.title("Transformer Global Features (Sample 1)")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'figure4.jpg'), bbox_inches='tight', dpi=600)
    plt.show()


# 5. 主函数：数据加载、模型训练、评估与可视化
if __name__ == '__main__':

    base_dir = './'
    # 设置随机种子，保证结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数设置
    seq_len = 200
    num_samples = 3000  # 总样本数
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3

    scaler = MinMaxScaler()

    # 构造数据集
    dataset = VirtualTimeSeriesDataset()
    # 划分训练集与测试集（80%/20%）
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型实例化
    model = FusedCNNTransformer(input_dim=1, embed_dim=64, cnn_channels=64, kernel_size=3,
                                num_transformer_layers=2, nhead=4, dropout=0.1, fc_dim=32)
    model.to(device)

    # 训练模型
    print("开始训练模型...")
    loss_history = train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

    # 在测试集上进行评估
    preds, trues, cnn_features, trans_features = evaluate_model(model, test_loader, device=device)

    # 可视化结果：绘制包含四个子图的综合图像
    visualize_results(loss_history, preds, trues, cnn_features, trans_features)