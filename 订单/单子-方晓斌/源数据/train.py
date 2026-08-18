import os
import sqlite3
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import psutil
import time
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd
import gc
import threading
import matplotlib.pyplot as plt
from model import LSTMModel

# 禁用 MKL - DNN
torch.backends.mkldnn.enabled = False

# 配置参数管理类
class Config:
    BATCH_SIZE = 16
    SYSTEM_MEMORY_RATIO = 0.85
    VAL_SPLIT = 0.2
    NUM_LAYERS = 1
    NUM_EPOCHS = 100
    CHECKPOINT_EPOCHS = 5
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 20
    WEIGHT_DECAY = 1e-5


# 数据加载模块
def load_files_to_db(fy4b_dir, gpm_dir, station_dir, station_info_file, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 创建复合索引以优化查询性能
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fy4b_combined ON fy4b_data (datetime, name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_gpm_combined ON gpm_data (datetime, name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_station_combined ON station_data (datetime, name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fy4b_name ON fy4b_data (name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_station_info_name ON station_info (name);")
        conn.commit()

        def load_data_from_files(data_dir, table_name, columns_to_keep):
            files = [file for file in os.listdir(data_dir) if not file.startswith('~$') and file.endswith(('.csv', '.xlsx'))]
            if not files:
                print(f"在 {data_dir} 目录下未找到符合条件的文件。")
            print(f"在 {data_dir} 目录下找到 {len(files)} 个文件: {files}")
            for file in tqdm(files, desc=f"Loading {table_name} data"):
                file_path = os.path.join(data_dir, file)
                try:
                    start_time = time.time()
                    data = pd.read_csv(file_path) if file.endswith('.csv') else pd.read_excel(file_path)
                    print(f"正在加载 {file_path}，数据行数: {len(data)}，列数: {len(data.columns)}")
                    print(f"列名: {data.columns}")
                    data = data[columns_to_keep]
                    # 批量插入数据
                    data.to_sql(table_name, conn, if_exists='append', index=False, chunksize=1000)
                    # 删除已成功加载的文件
                    os.remove(file_path)
                    end_time = time.time()
                    print(f"完成加载 {file_path}，耗时: {end_time - start_time:.2f} 秒")
                except Exception as e:
                    print(f"加载 {file_path} 时出错: {e}")

        # 加载 FY - 4B 数据
        load_data_from_files(fy4b_dir, 'fy4b_data', ['name', 'qpe24', 'datetime'])
        # 加载 GPM 数据
        load_data_from_files(gpm_dir, 'gpm_data', ['name', 'gpm', 'datetime'])
        # 加载站点数据
        load_data_from_files(station_dir, 'station_data', ['name', 'obv_24h', 'datetime'])

        # 加载站点信息
        try:
            start_time = time.time()
            station_info = pd.read_excel(station_info_file)
            print(f"正在加载 {station_info_file}，数据行数: {len(station_info)}，列数: {len(station_info.columns)}")
            print(f"列名: {station_info.columns}")
            station_info.to_sql('station_info', conn, if_exists='replace', index=False)
            end_time = time.time()
            print(f"完成加载 {station_info_file}，耗时: {end_time - start_time:.2f} 秒")
        except Exception as e:
            print(f"加载 {station_info_file} 时出错: {e}")

        conn.close()
    except Exception as e:
        print(f"数据加载到数据库时出现异常: {e}")


def load_training_data(db_path):
    try:
        conn = sqlite3.connect(db_path)
        # 先打印各表的行数，用于调试
        table_names = ['fy4b_data', 'gpm_data', 'station_data', 'station_info']
        for table_name in table_names:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"{table_name} 表的行数: {row_count}")

        query = """
        SELECT fy4b.qpe24, gpm.gpm, station.obv_24h, station_info.lat, station_info.lon, station_info.slope, station_info.elevation
        FROM fy4b_data fy4b
        LEFT JOIN gpm_data gpm ON fy4b.datetime = gpm.datetime AND fy4b.name = gpm.name
        LEFT JOIN station_data station ON fy4b.datetime = station.datetime AND fy4b.name = station.name
        LEFT JOIN station_info ON fy4b.name = station_info.name
        WHERE fy4b.datetime IS NOT NULL
        """
        chunksize = 1000
        input_data_chunks = []
        target_data_chunks = []
        scaler = StandardScaler()

        for chunk in tqdm(pd.read_sql_query(query, conn, chunksize=chunksize), desc="Loading training data"):
            if chunk.empty:
                continue
            # 提取输入特征和目标值
            input_chunk = chunk.drop('qpe24', axis=1).astype(np.float32)
            target_chunk = chunk['qpe24'].astype(np.float32)

            # 处理数据中的无效值
            input_chunk = input_chunk.replace([np.inf, -np.inf], np.nan)
            target_chunk = target_chunk.replace([np.inf, -np.inf], np.nan)

            # 合并处理无效值
            valid_mask = ~(input_chunk.isna().any(axis=1) | target_chunk.isna())
            input_chunk = input_chunk[valid_mask]
            target_chunk = target_chunk[valid_mask]

            if input_chunk.empty or target_chunk.empty:
                continue

            # 更严格的数据检查
            if input_chunk.isna().any().any() or target_chunk.isna().any():
                print("发现无效值，跳过当前数据块")
                continue

            # 检查输入数据的维度
            if input_chunk.shape[1] != 6:
                print(f"输入数据维度错误，期望维度为 6，实际维度为 {input_chunk.shape[1]}")
                print(f"当前数据列名: {chunk.columns}")
                continue

            # 使用 partial_fit 分批次计算均值和标准差
            scaler.partial_fit(input_chunk)

            # 对输入特征进行标准化处理
            input_chunk_scaled = scaler.transform(input_chunk)

            # 调整数据形状
            input_chunk_scaled = input_chunk_scaled.reshape(-1, 1, input_chunk_scaled.shape[1])
            target_chunk = target_chunk.values.reshape(-1, 1)

            # 存储处理后的数据块
            input_data_chunks.append(input_chunk_scaled)
            target_data_chunks.append(target_chunk)

            # 释放不再使用的变量并触发垃圾回收
            del chunk, input_chunk, target_chunk, input_chunk_scaled
            gc.collect()

            # 定期合并数据块以减少内存碎片
            if len(input_data_chunks) % 100 == 0:
                input_data_chunks = [np.concatenate(input_data_chunks)]
                target_data_chunks = [np.concatenate(target_data_chunks)]

            # 检查内存使用情况
            mem = psutil.virtual_memory()
            if mem.percent > Config.SYSTEM_MEMORY_RATIO * 100:
                print(f"内存使用超过限制（{Config.SYSTEM_MEMORY_RATIO * 100}%），暂停数据加载。")
                while mem.percent > Config.SYSTEM_MEMORY_RATIO * 100:
                    time.sleep(10)
                    mem = psutil.virtual_memory()
                print("内存使用回到安全范围，继续数据加载。")

        conn.close()

        if not input_data_chunks:
            print("SQL 查询结果为空，请检查数据库中的数据和查询语句。")
            return None, None, None

        # 拼接所有数据块
        input_data = np.concatenate(input_data_chunks, axis=0)
        target_data = np.concatenate(target_data_chunks, axis=0)

        # 释放中间数据块以节省内存
        del input_data_chunks, target_data_chunks
        gc.collect()

        return input_data, target_data, scaler
    except Exception as e:
        print(f"从数据库加载训练数据时出现异常: {e}")
        return None, None, None


# 内存监控函数
def monitor_memory():
    while True:
        mem = psutil.virtual_memory()
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"当前内存使用: {mem.percent}%，可用内存: {mem.available / (1024 ** 3):.2f} GB，当前显存使用: {gpu_mem:.2f} MB")
        else:
            print(f"当前内存使用: {mem.percent}%，可用内存: {mem.available / (1024 ** 3):.2f} GB")
        time.sleep(60)


def train_epoch(model, train_loader, optimizer, criterion, epoch, pbar, device):
    model.train()
    train_loss = 0
    batch_count = 0
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{Config.NUM_EPOCHS} - Training (Loss: {train_loss:.4f})'):
        try:
            inputs, targets = inputs.to(device), targets.to(device)
            # 检查输入数据的维度
            if inputs.size(-1) != 6:
                print(f"Epoch {epoch + 1} 训练时输入数据维度错误，期望维度为 6，实际维度为 {inputs.size(-1)}")
                continue
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            del inputs, targets, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
            batch_count += 1
            if batch_count % 10 == 0:
                pbar.update(1 / (len(train_loader) / 10))

            # 检查内存使用情况
            mem = psutil.virtual_memory()
            if mem.percent > Config.SYSTEM_MEMORY_RATIO * 100:
                print(f"内存使用超过限制（{Config.SYSTEM_MEMORY_RATIO * 100}%），暂停训练。")
                while mem.percent > Config.SYSTEM_MEMORY_RATIO * 100:
                    time.sleep(10)
                    mem = psutil.virtual_memory()
                print("内存使用回到安全范围，继续训练。")

        except Exception as e:
            print(f"Epoch {epoch + 1} 训练时出现异常: {e}")
    train_loss /= len(train_loader)
    return train_loss


def validate_epoch(model, val_loader, criterion, epoch, pbar, device):
    model.eval()
    val_loss = 0
    batch_count = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{Config.NUM_EPOCHS} - Validation (Loss: {val_loss:.4f})'):
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                # 检查输入数据的维度
                if inputs.size(-1) != 6:
                    print(f"Epoch {epoch + 1} 验证时输入数据维度错误，期望维度为 6，实际维度为 {inputs.size(-1)}")
                    continue
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                del inputs, targets, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()
                batch_count += 1
                if batch_count % 10 == 0:
                    pbar.update(1 / (len(val_loader) / 10))

                # 检查内存使用情况
                mem = psutil.virtual_memory()
                if mem.percent > Config.SYSTEM_MEMORY_RATIO * 100:
                    print(f"内存使用超过限制（{Config.SYSTEM_MEMORY_RATIO * 100}%），暂停验证。")
                    while mem.percent > Config.SYSTEM_MEMORY_RATIO * 100:
                        time.sleep(10)
                        mem = psutil.virtual_memory()
                    print("内存使用回到安全范围，继续验证。")

            except Exception as e:
                print(f"Epoch {epoch + 1} 验证时出现异常: {e}")
    val_loss /= len(val_loader)
    return val_loss


def main():
    try:
        # 启动内存监控线程
        memory_monitor = threading.Thread(target=monitor_memory, daemon=True)
        memory_monitor.start()

        # 数据加载与预处理
        fy4b_dir = r'F:\model\qpe'
        gpm_dir = r'F:\model\gpm'
        station_dir = r'F:\model\gcdata'
        station_info_file = r'F:\model\cxzsta2024.xlsx'

        model_save_dir = r'F:\model\para'
        # 检查并创建模型保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"已创建或使用模型保存目录: {model_save_dir}")

        db_dir = r'F:\model\db'
        # 检查并创建数据库目录
        os.makedirs(db_dir, exist_ok=True)
        print(f"已创建或使用数据库目录: {db_dir}")

        db_path = os.path.join(db_dir, 'precipitation.db')

        # 每次训练前先执行数据输入
        start_time = time.time()
        load_files_to_db(fy4b_dir, gpm_dir, station_dir, station_info_file, db_path)
        end_time = time.time()
        print(f"完成数据加载到数据库，耗时: {end_time - start_time:.2f} 秒")

        # 加载训练数据
        start_time = time.time()
        input_data, target_data, scaler = load_training_data(db_path)
        end_time = time.time()
        print(f"完成从数据库加载训练数据，耗时: {end_time - start_time:.2f} 秒")

        # 检查数据是否为空
        if input_data is None or target_data is None:
            print("加载的数据为空，请检查数据加载过程。")
            return
        print(f"Loaded {len(input_data)} training samples")

        # 数据集划分
        indices = np.arange(len(input_data))
        train_indices, val_indices = train_test_split(indices, test_size=Config.VAL_SPLIT, shuffle=False)

        train_dataset = TensorDataset(
            torch.tensor(input_data[train_indices], dtype=torch.float32),
            torch.tensor(target_data[train_indices], dtype=torch.float32)
        )

        val_dataset = TensorDataset(
            torch.tensor(input_data[val_indices], dtype=torch.float32),
            torch.tensor(target_data[val_indices], dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        # 初始化模型、损失函数和优化器
        input_size = 6
        hidden_size = 128
        num_layers = Config.NUM_LAYERS
        output_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = torch.nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # 加载检查点
        start_epoch = 0
        best_val_loss = float('inf')
        checkpoint_path = os.path.join(model_save_dir, 'checkpoint.pth')

        if os.path.exists(checkpoint_path):
            print("加载检查点...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"从 epoch {start_epoch} 开始继续训练")

        # 整体训练进度条
        total_start_time = time.time()
        train_losses = []
        val_losses = []
        with tqdm(total=Config.NUM_EPOCHS, desc="Overall Training Progress") as pbar:
            pbar.update(start_epoch)
            early_stopping_counter = 0

            for epoch in range(start_epoch, Config.NUM_EPOCHS):
                start_time = time.time()
                print(f"开始训练第 {epoch + 1} 个 epoch...")
                train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, pbar, device)
                val_loss = validate_epoch(model, val_loader, criterion, epoch, pbar, device)

                end_time = time.time()
                epoch_time = end_time - start_time
                elapsed_time = end_time - total_start_time
                remaining_epochs = Config.NUM_EPOCHS - (epoch + 1)
                remaining_time = remaining_epochs * epoch_time
                print(f'Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, 耗时: {epoch_time:.2f} 秒, 剩余时间预估: {remaining_time:.2f} 秒')

                # 保存检查点
                if (epoch + 1) % Config.CHECKPOINT_EPOCHS == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }
                    print("准备保存检查点...")
                    torch.save(checkpoint, checkpoint_path)
                    print(f"检查点已保存到 {checkpoint_path}")

                # 更新最佳验证损失
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # 保存最佳模型
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    best_model_path = os.path.join(model_save_dir, f'best_model_{current_time}.pth')
                    print("准备保存最佳模型...")
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'scaler': scaler
                    }
                    torch.save(checkpoint, best_model_path)
                    print(f"最佳模型已保存到 {best_model_path}")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                        print("Early stopping triggered.")
                        break

                pbar.update(1)
                scheduler.step(val_loss)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

        # 训练完成后删除检查点
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"检查点 {checkpoint_path} 已删除")

        # 绘制训练和验证损失曲线
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"训练过程中出现异常: {e}")
        # 可选：将错误信息写入文件
        with open("training_error.log", "a") as f:
            f.write(f"{datetime.now()}: {str(e)}\n")


if __name__ == "__main__":
    main()