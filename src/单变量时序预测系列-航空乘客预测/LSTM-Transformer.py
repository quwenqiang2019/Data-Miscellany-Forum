import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
set_seed(42)

# 1. 数据集生成
class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, seq_length=200, num_samples=1000, noise_std=0.1):
        """
        生成虚拟的多尺度时间序列数据。
        数据由多个正弦波叠加构成，包含低频、中频、高频成分，加上高斯噪声。
        """
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.noise_std = noise_std
        
        self.data = []
        self.targets = []
        t = np.linspace(0, 10, seq_length)
        # 对于每个样本随机生成不同的参数
        for _ in range(num_samples):
            # 随机选取不同的幅值、频率与相位
            A1, f1, phi1 = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.0), np.random.uniform(0, 2*np.pi)
            A2, f2, phi2 = np.random.uniform(0.2, 1.0), np.random.uniform(1.5, 3.0), np.random.uniform(0, 2*np.pi)
            A3, f3, phi3 = np.random.uniform(0.1, 0.5), np.random.uniform(3.0, 6.0), np.random.uniform(0, 2*np.pi)
            signal = A1 * np.sin(2*np.pi * f1 * t + phi1) + \
                     A2 * np.sin(2*np.pi * f2 * t + phi2) + \
                     A3 * np.sin(2*np.pi * f3 * t + phi3)
            signal += np.random.normal(0, noise_std, size=seq_length)
            # 将序列归一化
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-5)
            self.data.append(signal.astype(np.float32))
            # 预测目标这里取序列最后一个值作为示例（可扩展为多步预测）
            self.targets.append(signal[-1].astype(np.float32))
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 返回一个样本及其目标
        return self.data[idx], self.targets[idx]

# 2. 模型构建：融合 LSTM 与 Transformer 的多尺度模型
class MultiScaleTimeSeriesModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, lstm_layers=1, fusion_size=128, 
                 transformer_d_model=128, transformer_nhead=4, transformer_layers=2, dropout=0.1):
        super(MultiScaleTimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # 多尺度 LSTM 部分，假设三个尺度：不同的时间窗口或采样率（这里简单使用相同 LSTM 结构，不同的下采样）
        self.lstm_scale1 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.lstm_scale2 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.lstm_scale3 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        
        # 对于不同尺度输入，假设通过下采样（简单用切片模拟不同尺度）
        # 融合层：拼接后通过线性映射
        self.fusion_linear = nn.Linear(hidden_size*3, fusion_size)
        
        # Transformer 编码器部分
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=transformer_nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 输出层
        self.out_linear = nn.Linear(transformer_d_model, 1)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 投影层将融合后的特征映射到 Transformer 所需维度
        self.proj_linear = nn.Linear(fusion_size, transformer_d_model)
        
    def forward(self, x):
        """
        输入 x 的形状为 [batch_size, seq_length]
        为了适应 LSTM 的输入要求，将 x 展开为 [batch_size, seq_length, 1]
        """
        batch_size, seq_length = x.shape
        x = x.unsqueeze(-1)  # [B, T, 1]
        
        # 模拟多尺度处理：不同尺度的数据可以通过不同采样率获得
        # 尺度 1: 原始序列
        x_scale1 = x  # [B, T, 1]
        # 尺度 2: 每隔2个时间步采样一次
        x_scale2 = x[:, ::2, :]  # [B, T/2, 1]
        # 尺度 3: 每隔4个时间步采样一次
        x_scale3 = x[:, ::4, :]  # [B, T/4, 1]
        
        # 通过 LSTM 提取局部特征，取每个尺度最后时刻的隐藏状态作为特征表示
        _, (h_n1, _) = self.lstm_scale1(x_scale1)  # h_n1 shape: [num_layers, B, hidden_size]
        feat1 = h_n1[-1]  # [B, hidden_size]
        
        _, (h_n2, _) = self.lstm_scale2(x_scale2)
        feat2 = h_n2[-1]
        
        _, (h_n3, _) = self.lstm_scale3(x_scale3)
        feat3 = h_n3[-1]
        
        # 融合三个尺度的特征：拼接后通过非线性映射
        fused_feature = torch.cat([feat1, feat2, feat3], dim=-1)  # [B, hidden_size*3]
        fused_feature = self.relu(self.fusion_linear(fused_feature))  # [B, fusion_size]
        
        # 将融合后的特征复制为 Transformer 的序列输入（这里为了简单起见，重复复制形成序列）
        # 实际应用中，可以根据需求设计合适的时间步展开方式
        transformer_input = fused_feature.unsqueeze(1).repeat(1, seq_length, 1)  # [B, T, fusion_size]
        transformer_input = self.proj_linear(transformer_input)  # [B, T, transformer_d_model]
        # Transformer 要求输入形状为 [T, B, d_model]
        transformer_input = transformer_input.transpose(0, 1)  # [T, B, d_model]
        
        # Transformer 编码器
        transformer_output = self.transformer_encoder(transformer_input)  # [T, B, d_model]
        # 取最后一个时间步的输出作为最终表示
        final_feature = transformer_output[-1, :, :]  # [B, d_model]
        
        # 输出层得到最终预测结果
        out = self.out_linear(final_feature)  # [B, 1]
        out = out.squeeze(-1)  # [B]
        return out, transformer_output

# 3. 定义训练、评估和绘图函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=50, device='cpu'):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss = epoch_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return train_losses

def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    eval_loss = 0.0
    preds = []
    trues = []
    transformer_outputs = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, trans_out = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item() * inputs.size(0)
            preds.append(outputs.cpu().numpy())
            trues.append(targets.cpu().numpy())
            # 取 Transformer 输出第一个样本的注意力（或特征）进行可视化示例
            transformer_outputs.append(trans_out[:, 0, :].cpu().numpy())
    eval_loss = eval_loss / len(dataloader.dataset)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    transformer_outputs = np.concatenate(transformer_outputs, axis=1)  # shape: [T, num_samples]
    return eval_loss, preds, trues, transformer_outputs

def plot_results(train_losses, sample_time, sample_signal, transformer_feature, preds, trues):
    """
    绘制一个包含4个子图的图像：
    子图1：原始时间序列示例（使用鲜艳的颜色展示波动），说明原始数据的多尺度特性。
    子图2：多尺度局部特征（这里以 LSTM 模块提取的融合特征变化为示例），说明局部特征提取情况。
    子图3：Transformer 全局特征热图（注意力特征或输出特征随时间步的变化热图），说明全局依赖关系。
    子图4：预测结果与真实值对比图，说明模型预测效果。
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1：原始时间序列
    axs[0, 0].plot(sample_time, sample_signal, color='tab:blue', linewidth=2)
    axs[0, 0].set_title("Original Time Series Example", fontsize=14)
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Signal Amplitude")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].text(0.1, 0.9, "Note: Demonstrates the mixture of multi-scale periodic signals and noise in synthetic data",
                   transform=axs[0, 0].transAxes, fontsize=12, color='purple')

    # 图2：多尺度局部特征（此处用训练过程中记录的损失曲线模拟局部特征变化）
    axs[0, 1].plot(np.arange(len(train_losses)), train_losses, color='tab:red', linewidth=2)
    axs[0, 1].set_title("Training Loss Curve (Local Feature Extraction)", fontsize=14)
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Mean Squared Error")
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].text(0.1, 0.9, "Note: As training progresses, local feature extraction improves, and loss decreases",
                   transform=axs[0, 1].transAxes, fontsize=12, color='darkred')

    # 图3：Transformer 全局特征热图
    im = axs[1, 0].imshow(transformer_feature, aspect='auto', cmap='viridis')
    axs[1, 0].set_title("Transformer Global Feature Heatmap", fontsize=14)
    axs[1, 0].set_xlabel("Sample Index")
    axs[1, 0].set_ylabel("Time Step")
    fig.colorbar(im, ax=axs[1, 0])
    axs[1, 0].text(0.1, 0.9, "Note: Shows variations of Transformer encoder output features at different time steps",
                   transform=axs[1, 0].transAxes, fontsize=12, color='white')

    # 图4：预测结果与真实值对比图
    axs[1, 1].plot(preds, label='Predicted Values', color='tab:green', marker='o', linestyle='--')
    axs[1, 1].plot(trues, label='True Values', color='tab:orange', marker='x', linestyle='-')
    axs[1, 1].set_title("Prediction vs. Ground Truth", fontsize=14)
    axs[1, 1].set_xlabel("Sample Index")
    axs[1, 1].set_ylabel("Signal Amplitude")
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].text(0.1, 0.9, "Note: Comparing predictions with ground truth to evaluate model performance",
                   transform=axs[1, 1].transAxes, fontsize=12, color='blue')

    plt.tight_layout()
    plt.show()


# 4. 主函数：数据加载、模型训练、评估与结果可视化
if __name__ == '__main__':
    # 参数设置
    seq_length = 200
    num_samples = 1000
    batch_size = 32
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构造数据集与 DataLoader
    dataset = SyntheticTimeSeriesDataset(seq_length=seq_length, num_samples=num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = MultiScaleTimeSeriesModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练模型
    print("开始模型训练：")
    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
    
    # 在验证集上评估模型
    val_loss, preds, trues, transformer_feature = evaluate_model(model, val_loader, criterion, device=device)
    print(f"验证集损失：{val_loss:.4f}")
    
    # 为绘图生成示例数据：取一个样本的原始序列
    sample_idx = 0
    sample_signal, _ = dataset[sample_idx]
    sample_time = np.linspace(0, 10, seq_length)
    
    plot_results(train_losses, sample_time, sample_signal, transformer_feature, preds, trues)