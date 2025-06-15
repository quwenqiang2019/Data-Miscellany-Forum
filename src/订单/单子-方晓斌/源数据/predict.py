import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
from numpy.core.multiarray import _reconstruct, scalar
from numpy import ndarray, dtype, object_
from numpy.dtypes import ObjectDType, Float64DType, Int64DType
from model import LSTMModel  # 确保与训练代码的模型定义一致

# 添加所有必要的全局对象（与训练代码一致）
torch.serialization.add_safe_globals([
    StandardScaler,
    _reconstruct,
    scalar,
    ndarray,
    dtype,
    ObjectDType,
    Float64DType,
    Int64DType,
    np.float64,
    np.int64,
    np.ndarray,
    object_
])

# 配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DB_DIR = r'F:\model\db'
DB_PATH = os.path.join(DB_DIR, 'precipitation.db')
MODEL_SAVE_DIR = r'F:\model\para'
INPUT_DIR = r'F:\model\pre'
OUTPUT_DIR = r'F:\model\result'
HISTORICAL_YEARS = 10
REQUIRED_FEATURES = ['satellite_qpe', 'historical_obv', 'historical_gpm', 'slope']  # 4个特征，与训练一致


# 数据加载函数（与训练代码逻辑一致）
def load_obs_data(db_path, target_date):
    conn = sqlite3.connect(db_path)
    obs_data = pd.read_sql("""
        SELECT name, obv_24h 
        FROM station_data 
        WHERE datetime = ?
    """, conn, params=(target_date.strftime('%Y-%m-%d'),))
    conn.close()
    return obs_data


def load_gpm_data(db_path, target_date):
    conn = sqlite3.connect(db_path)
    gpm_data = pd.read_sql("""
        SELECT name, gpm 
        FROM gpm_data 
        WHERE datetime = ?
    """, conn, params=(target_date.strftime('%Y-%m-%d'),))
    conn.close()
    return gpm_data


def load_station_info(db_path):
    conn = sqlite3.connect(db_path)
    station_info = pd.read_sql("SELECT name, slope FROM station_info", conn)
    conn.close()
    return station_info


def load_historical_data(db_path, target_date):
    try:
        conn = sqlite3.connect(db_path)

        # 1. 当前卫星数据（FY4B）
        current_satellite = pd.read_sql("""
            SELECT name, qpe24 AS satellite_qpe 
            FROM fy4b_data 
            WHERE datetime = ?
        """, conn, params=(target_date.strftime('%Y-%m-%d'),))
        if current_satellite.empty:
            raise ValueError("无当前日期卫星数据")

        # 2. 历史同期数据（近10年同日）
        historical_obs = []
        for year_offset in range(1, HISTORICAL_YEARS + 1):
            hist_date = target_date - relativedelta(years=year_offset)
            obs = load_obs_data(db_path, hist_date)
            gpm = load_gpm_data(db_path, hist_date)
            merged = obs.merge(gpm, on='name', how='outer')
            merged['obv_24h'] = merged['obv_24h'].fillna(merged['gpm'])
            historical_obs.append(merged[['name', 'obv_24h', 'gpm']])

        historical_df = pd.concat(historical_obs)
        historical_mean = historical_df.groupby('name').agg(
            historical_obv=('obv_24h', 'mean'),
            historical_gpm=('gpm', 'mean')
        ).reset_index()

        # 3. 站点信息（坡度）
        station_info = load_station_info(db_path)
        if station_info.empty:
            raise ValueError("无站点信息")

        # 4. 合并特征（4个，与训练一致）
        correction_data = current_satellite.merge(historical_mean, on='name')
        correction_data = correction_data.merge(station_info, on='name')
        correction_data = correction_data[REQUIRED_FEATURES]

        return correction_data

    except Exception as e:
        logging.error(f"加载历史数据失败: {e}")
        raise


# 预报订正函数（匹配训练模型的输入尺寸）
def correct_forecast(forecast_date, model_path):
    try:
        # 1. 加载数据（4个特征）
        correction_data = load_historical_data(DB_PATH, forecast_date)

        # 2. 加载模型（与训练参数一致）
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model = LSTMModel(
            input_size=4,  # 必须为4
            hidden_size=128,
            num_layers=2,
            output_size=1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']  # 加载训练时的标准化器
        model.eval()

        # 3. 数据标准化（使用训练时的scaler）
        scaled_data = scaler.transform(correction_data)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(1)  # [B, 1, 4]

        # 4. 预测
        with torch.no_grad():
            predictions = model(input_tensor)

        # 5. 反标准化
        corrected_values = scaler.inverse_transform(predictions.numpy())

        # 6. 生成结果
        return pd.DataFrame({
            'name': correction_data['name'],
            'corrected_qpe24': corrected_values.squeeze(),
            'original_satellite_qpe': correction_data['satellite_qpe'],
            'forecast_date': forecast_date.strftime('%Y-%m-%d')
        })

    except Exception as e:
        logging.error(f"订正失败: {e}")
        raise


# 辅助函数（获取最新模型）
def get_latest_model():
    models = [f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith('best_model_')]
    if not models:
        logging.error("无可用模型")
        return None
    latest_model = max(models, key=lambda x: x.split('_')[-1].replace('.pth', ''))
    return os.path.join(MODEL_SAVE_DIR, latest_model)


# 主函数
def main():
    model_path = get_latest_model()
    if not model_path:
        return

    for file in os.listdir(INPUT_DIR):
        if not file.startswith('FY4B_') or not file.endswith('.xlsx'):
            continue

        try:
            # 解析日期（假设文件名格式: FY4B_YYYY-MM-DD.xlsx）
            date_str = file.split('_')[1].split('.')[0]
            forecast_date = datetime.strptime(date_str, '%Y-%m-%d')

            # 执行订正
            result = correct_forecast(forecast_date, model_path)

            # 保存结果
            output_file = os.path.join(OUTPUT_DIR, f"corrected_{date_str}.csv")
            result.to_csv(output_file, index=False)
            logging.info(f"保存结果: {output_file}")

        except Exception as e:
            logging.error(f"处理文件 {file} 失败: {e}")


if __name__ == "__main__":
    main()