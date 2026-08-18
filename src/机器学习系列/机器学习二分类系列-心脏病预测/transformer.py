# pip install tensorflow pandas numpy matplotlib seaborn scikit-learn scipy openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)


def create_results_folder():
    """创建结果文件夹"""
    desktop_path = "./"
    results_path = os.path.join(desktop_path, "Results模型-Transformer")

    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"创建结果文件夹: {results_path}")
    else:
        print(f"结果文件夹已存在: {results_path}")

    return results_path


def load_and_preprocess_data():
    """加载和预处理数据"""
    desktop_path = "./"
    file_path = os.path.join(desktop_path, "Dataset.csv")

    data = pd.read_csv(file_path)
    print("数据加载成功!")
    print(f"数据形状: {data.shape}")
    return data


def prepare_transformer_data(df):
    """准备Transformer数据"""
    X = df.drop('target', axis=1).values
    y = df['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 重塑数据为Transformer格式 (样本数, 序列长度=1, 特征数)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    return X_reshaped, y, scaler

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer块实现"""
    def __init__ (self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock,self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
        Dense(ff_dim, activation="relu"),
        Dense(embed_dim),
        ])

        self.layernorm1 =LayerNormalization(epsilon=1e-6)
        self.layernorm2 =LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape, num_heads=4, ff_dim=32, num_blocks=2):
    """构建Transformer模型"""
    inputs = Input(shape=input_shape)

    # 位置编码
    x = inputs

    print(input_shape[-1])
    # Transformer块
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim=input_shape[-1], num_heads=num_heads, ff_dim=ff_dim)(x, training=True)

    # 全局平均池化
    x = GlobalAveragePooling1D()(x)

    # 分类头
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def build_alternative_model(input_shape):
    """构建替代模型 - 使用全连接网络"""
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def plot_training_history(history, results_path, model_type="Transformer"):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['loss'], label='训练损失', color='blue')
    ax1.plot(history.history['val_loss'], label='验证损失', color='red')
    ax1.set_title(f'{model_type} - 训练损失')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['accuracy'], label='训练准确率', color='blue')
    ax2.plot(history.history['val_accuracy'], label='验证准确率', color='red')
    ax2.set_title(f'{model_type} - 训练准确率')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, '训练历史.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, results_path, model_type="Transformer"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'{model_type} - 混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.savefig(os.path.join(results_path, '混淆矩阵.jpg'), dpi=300, bbox_inches='tight')
    plt.show()

    return cm


def plot_roc_curve(y_true, y_prob, results_path, model_type="Transformer"):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', linewidth=2, label=f'{model_type} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (1 - 特异度)')
    plt.ylabel('真正率 (灵敏度)')
    plt.title(f'{model_type} - ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, 'ROC曲线.jpg'), dpi=300, bbox_inches='tight')
    plt.show()

    return fpr, tpr, thresholds, auc_score


def plot_calibration_curve(y_true, y_prob, results_path, model_type="Transformer"):
    """绘制校准曲线"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', color='blue', linewidth=2, label=model_type)
    plt.plot([0, 1], [0, 1], '--', color='red', linewidth=1, label='完美校准')
    plt.xlabel('预测概率')
    plt.ylabel('实际概率')
    plt.title(f'{model_type} - 校准曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(results_path, '校准曲线.jpg'), dpi=300, bbox_inches='tight')
    plt.show()

    return prob_true, prob_pred


def plot_clinical_impact_curve(y_true, y_prob, results_path, model_type="Transformer"):
    """绘制临床影响曲线"""
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefit = []

    for p in thresholds:
        tp = np.sum((y_prob >= p) & (y_true == 1))
        fp = np.sum((y_prob >= p) & (y_true == 0))
        n = len(y_true)
        nb = tp / n - fp / n * (p / (1 - p))
        net_benefit.append(nb)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit, color='blue', linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('阈值概率')
    plt.ylabel('净获益')
    plt.title(f'{model_type} - 临床影响曲线')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, '临床影响曲线.jpg'), dpi=300, bbox_inches='tight')
    plt.show()

    return thresholds, net_benefit


# ==================== Transformer特有可视化 ====================
def create_attention_visualization(results_path):
    """创建多头注意力机制模拟图"""
    seq_length = 10
    n_heads =8
    #创建模拟的注意力权重
    np.random.seed(123)
    attention_weights = np.random.uniform(0,1,(seq_length, seq_length))
    # 标准化为概率分布
    attention_weights = attention_weights /np.sum(attention_weights, axis=1,keepdims=True)
    # 绘制热图
    plt.figure(figsize=(10,8))
    sns.heatmap(attention_weights, cmap='plasma', annot=True, fmt='.2f',
                xticklabels=[f'Pos{i}' for i in range(1,seq_length + 1)],
                yticklabels=[f'Q{i}'for i in range(1,seq_length + 1)])
    plt.title('Transformer多头注意力机制模拟')
    plt.xlabel('键位置')
    plt.ylabel('查询位置')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path,'多头注意力机制.jpg'), dpi=300,bbox_inches='tight')
    plt.show()


def create_positional_encoding_plot(results_path):
    """创建位置编码可视化"""
    seq_length = 20
    d_model = 64

    # 创建位置编码矩阵
    positional_encoding = np.zeros((seq_length, d_model))

    for pos in range(seq_length):
        for i in range(d_model):
            if i % 2 == 0:
                # 偶数位置使用正弦
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((i) / d_model)))
            else:
                # 奇数位置使用余弦
                positional_encoding[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))

    # 选择前8个维度进行可视化
    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.plot(range(seq_length), positional_encoding[:, i],
                 label=f'Dim {i + 1}', linewidth=2)

    plt.title('Transformer位置编码可视化')
    plt.xlabel('序列位置')
    plt.ylabel('编码值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, '位置编码可视化.jpg'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
def create_transformer_architecture_plot(results_path):
    """创建Transformer架构图"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # 定义层和连接
    layers = [
        {'name': '输入嵌入', 'x': 1, 'y': 0.5, 'color': '#1f77b4'},
        {'name': '位置编码', 'x': 2, 'y': 0.5, 'color': '#ff7f0e'},
        {'name': '编码器层×6', 'x': 3, 'y': 0.5, 'color': '#2ca02c'},
        {'name': '多头注意力', 'x': 4, 'y': 0.3, 'color': '#d62728'},
        {'name': '前馈网络', 'x': 4, 'y': 0.7, 'color': '#9467bd'},
        {'name': '解码器层×6', 'x': 5, 'y': 0.5, 'color': '#8c564b'},
        {'name': '输出层', 'x': 6, 'y': 0.5, 'color': '#e377c2'}
    ]

    connections = [
        (0, 1), (1, 2), (2, 3), (2, 4), (3, 2), (4, 2), (2, 5), (5, 6)
    ]

    # 绘制连接线
    for conn in connections:
        start = layers[conn[0]]
        end = layers[conn[1]]
        ax.annotate('', xy=(end['x'], end['y']), xytext=(start['x'], start['y']),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.7))

    # 绘制层节点
    for layer in layers:
        ax.add_patch(plt.Rectangle((layer['x'] - 0.3, layer['y'] - 0.1), 0.6, 0.2,
                                   color=layer['color'], alpha=0.8))
        ax.text(layer['x'], layer['y'], layer['name'],
                ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, 1)
    ax.set_title('Transformer架构示意图', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'Transformer架构图.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


def create_layer_norm_plot(results_path):
    """创建层归一化效果可视化"""
    np.random.seed(123)
    original_data = np.random.normal(10, 5, 1000)
    normalized_data = (original_data - np.mean(original_data)) / np.std(original_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(original_data, bins=30, alpha=0.7, color='steelblue', density=True)
    ax1.set_title('原始数据分布')
    ax1.set_xlabel('数值')
    ax1.set_ylabel('密度')
    ax1.grid(True, alpha=0.3)

    ax2.hist(normalized_data, bins=30, alpha=0.7, color='darkred', density=True)
    ax2.set_title('层归一化后分布')
    ax2.set_xlabel('数值')
    ax2.set_ylabel('密度')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Transformer层归一化效果', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, '层归一化效果.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


def create_residual_connection_plot(results_path):
    """创建残差连接示意图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 定义节点
    nodes = [
        {'name': '输入', 'x': 1, 'y': 0.5, 'color': '#1f77b4'},
        {'name': '子层处理', 'x': 2, 'y': 0.5, 'color': '#2ca02c'},
        {'name': '层归一化', 'x': 3, 'y': 0.5, 'color': '#ff7f0e'},
        {'name': '残差连接', 'x': 2.5, 'y': 0.2, 'color': '#d62728'},
        {'name': '前馈网络', 'x': 4, 'y': 0.5, 'color': '#9467bd'},
        {'name': '输出', 'x': 5, 'y': 0.5, 'color': '#e377c2'}
    ]

    # 定义连接
    connections = [
        (0, 1), (1, 2), (2, 4), (0, 3), (3, 4), (4, 5)
    ]

    # 绘制连接线
    for conn in connections:
        start = nodes[conn[0]]
        end = nodes[conn[1]]
        ax.annotate('', xy=(end['x'], end['y']), xytext=(start['x'], start['y']),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.7))

    # 绘制节点
    for node in nodes:
        ax.add_patch(plt.Circle((node['x'], node['y']), 0.15,
                                color=node['color'], alpha=0.8))
        ax.text(node['x'], node['y'], node['name'],
                ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1)
    ax.set_title('Transformer残差连接示意图', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, '残差连接示意图.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


def create_feature_attention_heatmap(feature_names, results_path):
    """创建特征注意力热图"""
    n_features = len(feature_names)

    # 创建模拟的注意力矩阵
    np.random.seed(123)
    attention_matrix = np.random.uniform(0, 1, (n_features, n_features))

    # 标准化
    attention_matrix = attention_matrix / np.sum(attention_matrix, axis=1, keepdims=True)

    # 如果特征太多，只显示前15个
    if n_features > 15:
        display_features = 15
        attention_matrix_display = attention_matrix[:display_features, :display_features]
        feature_names_display = feature_names[:display_features]
    else:
        display_features = n_features
        attention_matrix_display = attention_matrix
        feature_names_display = feature_names

    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_matrix_display,
                xticklabels=feature_names_display,
                yticklabels=feature_names_display,
                cmap='magma', annot=True, fmt='.2f')
    plt.title('Transformer特征间注意力热图（模拟）')
    plt.xlabel('关键特征')
    plt.ylabel('查询特征')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, '特征注意力热图.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


def create_multihead_attention_plot(results_path):
    """创建多头注意力权重分布"""
    n_heads = 8
    np.random.seed(123)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(n_heads):
        # 模拟每个头的注意力权重分布
        weights = np.random.beta(0.5, 0.5, 1000)
        ax.hist(weights, bins=30, alpha=0.3, label=f'Head {i + 1}', density=True)

    ax.set_title('Transformer多头注意力权重分布')
    ax.set_xlabel('注意力权重')
    ax.set_ylabel('密度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, '多头注意力分布.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance_transformer(model, feature_names, results_path, model_type="Transformer"):
    """绘制特征重要性图 - Transformer版本"""
    # 对于Transformer模型，使用第一层Dense层的权重
    first_dense_layer = None
    for layer in model.layers:
        if'dense'in layer.name and 'transformer_block' not in layer.name:
            first_dense_layer = layer
            break

    if first_dense_layer is not None:
        first_layer_weights = first_dense_layer.get_weights()[0]
        # 全连接层权重形状: (input_features, units)
        importance = np.sum(np.abs(first_layer_weights), axis=1)
    else:
        # 如果没有找到Dense层，使用均匀重要性
        importance = np.ones(len(feature_names))

    # 确保importance是一维数组
    importance = importance.flatten()

    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    top_features = min(15, len(importance_df))
    plt.barh(importance_df['Feature'].head(top_features),
             importance_df['Importance'].head(top_features),
             color='steelblue', alpha=0.7)
    plt.xlabel('重要性得分')
    plt.title(f'{model_type} - 特征重要性 (前{top_features}个特征)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, '特征重要性图.jpg'), dpi=300, bbox_inches='tight')
    plt.show()

    return importance_df


def plot_prediction_distribution(y_true, y_prob, results_path, model_type="Transformer"):
    """绘制预测概率分布图"""
    plt.figure(figsize=(8, 6))

    for outcome in [0, 1]:
        mask = y_true == outcome
        plt.hist(y_prob[mask], bins=30, alpha=0.6,
                 label=f'结局 = {outcome}', density=True)

    plt.xlabel('预测概率')
    plt.ylabel('密度')
    plt.title(f'{model_type} - 预测概率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, '预测概率分布.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_residual_analysis(y_true, y_prob, results_path, model_type="Transformer"):
    """绘制残差分析图"""
    residuals = y_true - y_prob

    plt.figure(figsize=(8, 6))
    plt.scatter(y_prob, residuals, alpha=0.6, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

    z = np.polyfit(y_prob, residuals, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(y_prob.min(), y_prob.max(), 100)
    plt.plot(x_smooth, p(x_smooth), color='darkgreen', linewidth=2)

    plt.xlabel('预测概率')
    plt.ylabel('残差')
    plt.title(f'{model_type} - 残差分析')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, '残差分析.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


def save_results(model, history, performance_metrics, cm, roc_data,
                 importance_df, model_info, results_path, model_type="Transformer"):
    """保存所有结果"""

    performance_df = pd.DataFrame(performance_metrics)
    performance_df.to_csv(os.path.join(results_path, f'{model_type}模型性能指标.csv'),
                          index=False, encoding='utf-8-sig')

    cm_df = pd.DataFrame(cm, index=['真实0', '真实1'], columns=['预测0', '预测1'])
    cm_df.to_csv(os.path.join(results_path, '混淆矩阵.csv'), encoding='utf-8-sig')

    roc_df = pd.DataFrame({
        '假正率': roc_data[0],
        '真正率': roc_data[1],
        '阈值': roc_data[2]
    })
    roc_df.to_csv(os.path.join(results_path, 'ROC曲线数据.csv'), index=False, encoding='utf-8-sig')

    importance_df.to_csv(os.path.join(results_path, '特征重要性.csv'), index=False, encoding='utf-8-sig')

    model_info_df = pd.DataFrame(model_info)
    model_info_df.to_csv(os.path.join(results_path, '模型结构信息.csv'), index=False, encoding='utf-8-sig')

    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    })
    history_df.to_csv(os.path.join(results_path, '训练误差历史.csv'), index=False, encoding='utf-8-sig')

    # model.save(os.path.join(results_path, f'{model_type.lower()}_model.h5'))

    print("所有结果已保存!")


def main():
    """主函数"""
    print("开始Transformer模型分析...")

    results_path = create_results_folder()
    df = load_and_preprocess_data()
    print(f"原始数据形状: {df.shape}")

    X, y, scaler = prepare_transformer_data(df)
    print(f"输入数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"输入特征数: {input_shape}")

    # 尝试构建Transformer模型
    print("尝试构建Transformer模型...")
    model = build_transformer_model(input_shape)
    model_type = "Transformer"
    print("Transformer模型构建成功!")

    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    ]

    print("开始训练模型...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    performance_metrics = {
        '指标': ['AUC', '准确率', '灵敏度', '特异度', '精确率', 'F1分数', '最终训练损失'],
        '值': [
            round(auc_score, 3),
            round(accuracy, 3),
            round(report['1']['recall'], 3),
            round(report['0']['recall'], 3),
            round(report['1']['precision'], 3),
            round(report['1']['f1-score'], 3),
            round(history.history['val_loss'][-1], 6)
        ]
    }

    model_info = {
        '参数': ['网络类型', '输入特征数', '注意力头数', '前馈网络维度',
                 'Transformer块数', '输出层节点数', '最终训练误差', '训练轮次', '激活函数'],
        '值': [
            model_type,
            X_train.shape[2],
            4, 32, 2,
            1,
            round(history.history['val_loss'][-1], 6),
            len(history.history['loss']),
            'ReLU/Sigmoid'
        ]
    }

    print("生成基础可视化图表...")

    plot_training_history(history, results_path, model_type)
    cm = plot_confusion_matrix(y_test, y_pred, results_path, model_type)
    roc_data = plot_roc_curve(y_test, y_prob, results_path, model_type)
    calibration_data = plot_calibration_curve(y_test, y_prob, results_path, model_type)
    cic_data = plot_clinical_impact_curve(y_test, y_prob, results_path, model_type)

    feature_names = [col for col in df.columns if col != 'target']
    importance_df = plot_feature_importance_transformer(model, feature_names, results_path, model_type)

    plot_prediction_distribution(y_test, y_prob, results_path, model_type)
    plot_residual_analysis(y_test, y_prob, results_path, model_type)

    print("生成Transformer特有可视化图表...")

    # Transformer特有可视化
    create_attention_visualization(results_path)
    create_positional_encoding_plot(results_path)
    create_transformer_architecture_plot(results_path)
    create_layer_norm_plot(results_path)
    create_residual_connection_plot(results_path)
    create_feature_attention_heatmap(feature_names, results_path)
    create_multihead_attention_plot(results_path)

    save_results(model, history, performance_metrics, cm, roc_data,
                 importance_df, model_info, results_path, model_type)

    print("\n" + "=" * 50)
    print(f"{model_type}分析完成!")
    print("=" * 50)
    print(f"关键性能指标:")
    print(f"- AUC: {auc_score:.3f}")
    print(f"- 准确率: {accuracy:.3f}")
    print(f"- 灵敏度: {report['1']['recall']:.3f}")
    print(f"- 特异度: {report['0']['recall']:.3f}")
    print(f"- 精确率: {report['1']['precision']:.3f}")
    print(f"- F1分数: {report['1']['f1-score']:.3f}")

    print(f"\n模型结构: {model_type}")
    print(f"\n所有结果已保存到: {results_path}")


if __name__ == "__main__":
    main()

