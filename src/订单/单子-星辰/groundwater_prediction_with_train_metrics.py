#!/usr/bin/env python3
"""
地下水埋深预测模型 - CNN-LSTM / CNN-LSTM-Attention / CNN-LSTM-Transformer
预测武陟县7号井、8号井、10号井、18号井、41号井、43号井的埋深值
新增功能：同时输出训练集和测试集的评估指标
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
sns.set(font_scale=1.2)
plt.rc('font', family=['SimHei'], size=12)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = '/root/snap/wenqiang/Data-Miscellany-Forum/src/订单/单子-星辰/data'
OUTPUT_DIR = '/root/snap/wenqiang/Data-Miscellany-Forum/src/订单/单子-星辰/output'

WELLS = ['7号井', '8号井', '10号井', '18号井', '41号井', '43号井']
WELL_FILES = [f'武陟县{well}2000-2023.xlsx' for well in WELLS]

SEQUENCE_LENGTH = 30
TEST_RATIO = 0.2
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)


def load_data():
    """加载并预处理所有数据"""
    print("=" * 60)
    print("正在加载数据...")
    
    groundwater = pd.read_excel(f'{DATA_DIR}/地下水开采量.xlsx')
    irrigation = pd.read_excel(f'{DATA_DIR}/农田灌溉用水量.xlsx')
    rainfall = pd.read_excel(f'{DATA_DIR}/日降雨量.xlsx')
    temperature = pd.read_excel(f'{DATA_DIR}/日平均气温.xlsx')
    evaporation = pd.read_excel(f'{DATA_DIR}/逐日蒸散发数据.xlsx')
    
    rainfall['时间'] = pd.to_datetime(rainfall['时间'])
    temperature['时间'] = pd.to_datetime(temperature['时间'])
    evaporation['时间'] = pd.to_datetime(evaporation['时间'])
    
    rainfall = rainfall.sort_values('时间').reset_index(drop=True)
    temperature = temperature.sort_values('时间').reset_index(drop=True)
    evaporation = evaporation.sort_values('时间').reset_index(drop=True)
    
    base_df = rainfall[['时间']].copy()
    base_df = base_df.merge(rainfall[['时间', '降雨量(mm/天)']], on='时间', how='left')
    base_df = base_df.merge(temperature[['时间', '日平均气温(°C)']], on='时间', how='left')
    base_df = base_df.merge(evaporation[['时间', '逐日蒸散发（mm）']], on='时间', how='left')
    
    groundwater.columns = ['年份', '地下水开采量']
    irrigation.columns = ['年份', '农田灌溉用水量']
    groundwater['年份'] = groundwater['年份'].astype(int)
    irrigation['年份'] = irrigation['年份'].astype(int)
    
    base_df['年份'] = base_df['时间'].dt.year
    
    base_df = base_df.merge(groundwater, on='年份', how='left')
    base_df = base_df.merge(irrigation, on='年份', how='left')
    
    base_df = base_df.drop(columns=['年份'])
    base_df = base_df.ffill().bfill()
    
    print(base_df)
    print(f"辅助特征数据形状: {base_df.shape}")
    print(f"辅助特征列: {base_df.columns.tolist()}")
    
    well_data = {}
    for well, file in zip(WELLS, WELL_FILES):
        df = pd.read_excel(f'{DATA_DIR}/{file}')
        if '埋深值(米)' in df.columns:
            df = df.rename(columns={'埋深值(米)': '埋深值'})
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        print(df)
        well_data[well] = df
        print(f"  {well}: {df.shape[0]} 条记录")
    
    return base_df, well_data


def create_sequences(data, target, seq_length):
    """创建时间序列数据集"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def build_cnn_lstm(input_shape, output_steps=1):
    """CNN-LSTM模型"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(output_steps)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_cnn_lstm_attention(input_shape, output_steps=1):
    """CNN-LSTM-Attention模型"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(output_steps)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_cnn_lstm_transformer(input_shape, output_steps=1):
    """CNN-LSTM-Transformer模型"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    
    seq_len = x.shape[1]
    embedding = layers.Dense(32)(x)
    
    pos_encoding = np.zeros((1, seq_len, 32))
    for i in range(seq_len):
        for j in range(0, 32, 2):
            pos_encoding[0, i, j] = np.sin(i / np.power(10000, j / 32))
            if j + 1 < 32:
                pos_encoding[0, i, j + 1] = np.cos(i / np.power(10000, j / 32))
    x = embedding + tf.constant(pos_encoding, dtype=tf.float32)
    
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(output_steps)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def plot_predictions(y_true, y_pred, title, filename):
    """绘制预测对比图"""
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='真实值', linewidth=1.5, alpha=0.8)
    plt.plot(y_pred, label='预测值', linewidth=1.5, alpha=0.8)
    plt.xlabel('时间步')
    plt.ylabel('埋深值 (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history, title, filename):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results, filename):
    """绘制各模型指标对比图"""
    metrics = ['R2', 'RMSE', 'MAE', 'MAPE']
    models = list(results[list(results.keys())[0]].keys())
    wells = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(wells))
        width = 0.25
        
        for i, model in enumerate(models):
            values = [results[well][model]['test'][metric] for well in wells]
            ax.bar(x + i * width, values, width, label=model)
        
        ax.set_xlabel('井号')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} 对比 (测试集)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([w.replace('号井', '') for w in wells], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_predictions(y_true_dict, y_pred_dict, well, filename):
    """绘制多模型预测对比图"""
    plt.figure(figsize=(14, 6))
    plt.plot(y_true_dict[well], label='真实值', linewidth=2, color='black')
    
    colors = ['blue', 'red', 'green']
    for (model, y_pred), color in zip(y_pred_dict[well].items(), colors):
        plt.plot(y_pred, label=f'{model}预测值', linewidth=1.2, alpha=0.7, color=color)
    
    plt.xlabel('时间步')
    plt.ylabel('埋深值 (m)')
    plt.title(f'{well} 各模型预测对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def train_and_evaluate(well_name, well_df, aux_df):
    """训练并评估单个井的模型"""
    print(f"\n{'=' * 50}")
    print(f"处理井: {well_name}")
    print(f"{'=' * 50}")
    
    merged = pd.merge(well_df[['日期', '埋深值']], aux_df, left_on='日期', right_on='时间', how='inner')
    merged = merged.sort_values('日期').reset_index(drop=True)
    merged = merged.ffill().bfill()
    
    merged = merged[['日期', '埋深值', '降雨量(mm/天)', '日平均气温(°C)', '逐日蒸散发（mm）', '地下水开采量', '农田灌溉用水量']]
    merged.set_index(merged.columns[0], inplace=True)
    print(merged)

    feature_cols = ['埋深值', '降雨量(mm/天)', '日平均气温(°C)', '逐日蒸散发（mm）']
    target_col = '埋深值'
    
    features = merged[feature_cols].values
    target = merged[target_col].values.reshape(-1, 1)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)
    
    X, y = create_sequences(features_scaled, target_scaled, SEQUENCE_LENGTH)
    
    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    models = {
        'CNN-LSTM': build_cnn_lstm(input_shape),
        'CNN-LSTM-Attention': build_cnn_lstm_attention(input_shape),
        'CNN-LSTM-Transformer': build_cnn_lstm_transformer(input_shape)
    }
    
    results = {}
    predictions = {}
    histories = {}
    
    for model_name, model in models.items():
        print(f"\n训练模型: {model_name}")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0
        )
        
        histories[model_name] = history
        
        y_train_pred_scaled = model.predict(X_train, verbose=0)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
        y_train_true = scaler_y.inverse_transform(y_train)
        train_metrics = calculate_metrics(y_train_true, y_train_pred)
        
        y_test_pred_scaled = model.predict(X_test, verbose=0)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
        y_test_true = scaler_y.inverse_transform(y_test)
        test_metrics = calculate_metrics(y_test_true, y_test_pred)
        
        results[model_name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        predictions[model_name] = y_test_pred.flatten()
        
        print(f"  训练集 - R2: {train_metrics['R2']:.4f}, RMSE: {train_metrics['RMSE']:.4f}, MAE: {train_metrics['MAE']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"  测试集 - R2: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, MAE: {test_metrics['MAE']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        model.save(f'{OUTPUT_DIR}/models/{well_name}_{model_name}.h5')
        
        plot_training_history(
            history, 
            f'{well_name} - {model_name} 训练曲线',
            f'{OUTPUT_DIR}/figures/{well_name}_{model_name}_history.png'
        )
        
        plot_predictions(
            y_test_true.flatten(), 
            y_test_pred.flatten(),
            f'{well_name} - {model_name} 预测结果',
            f'{OUTPUT_DIR}/figures/{well_name}_{model_name}_prediction.png'
        )
    
    return results, predictions, y_test_true.flatten()


def main():
    """主函数"""
    print("=" * 60)
    print("地下水埋深预测模型 (含训练集指标)")
    print("=" * 60)
    
    print(f"\nTensorFlow 版本: {tf.__version__}")
    print(f"GPU 可用: {tf.config.list_physical_devices('GPU')}")
    
    aux_df, well_data = load_data()
    
    all_results = {}
    all_predictions = {}
    all_true_values = {}
    
    for well in WELLS:
        results, predictions, y_true = train_and_evaluate(well, well_data[well], aux_df)
        all_results[well] = results
        all_predictions[well] = predictions
        all_true_values[well] = y_true
    
    print("\n" + "=" * 60)
    print("所有井预测结果汇总")
    print("=" * 60)
    
    summary_df = []
    for well in WELLS:
        for model, metrics in all_results[well].items():
            summary_df.append({
                '井号': well,
                '模型': model,
                'Train_R2': metrics['train']['R2'],
                'Train_RMSE': metrics['train']['RMSE'],
                'Train_MAE': metrics['train']['MAE'],
                'Train_MAPE(%)': metrics['train']['MAPE'],
                'Test_R2': metrics['test']['R2'],
                'Test_RMSE': metrics['test']['RMSE'],
                'Test_MAE': metrics['test']['MAE'],
                'Test_MAPE(%)': metrics['test']['MAPE']
            })
    
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(f'{OUTPUT_DIR}/results_summary.csv', index=False, encoding='utf-8-sig')
    
    print("\n各井各模型指标:")
    print(summary_df.to_string(index=False))
    
    plot_metrics_comparison(all_results, f'{OUTPUT_DIR}/figures/metrics_comparison.png')
    
    for well in WELLS:
        plot_all_predictions(
            all_true_values,
            all_predictions,
            well,
            f'{OUTPUT_DIR}/figures/{well}_all_models_comparison.png'
        )
    
    print(f"\n结果已保存至: {OUTPUT_DIR}")
    print(f"  - 模型权重: {OUTPUT_DIR}/models/")
    print(f"  - 可视化图表: {OUTPUT_DIR}/figures/")
    print(f"  - 指标汇总: {OUTPUT_DIR}/results_summary.csv")


if __name__ == '__main__':
    main()
