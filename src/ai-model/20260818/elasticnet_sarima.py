"""
ElasticNet + SARIMA 混合预测模型
用于月度经济时间序列（BASPL）的回归预测
数据来源：GitHub - regression-prediction-algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.style.use('fivethirtyeight')

# =============================================
# 1. 读取数据
# =============================================
data = pd.read_excel('data.xlsx')
data = data.iloc[:, :]
choose = 'BASPL'  # 选取目标列
print(f"数据集大小: {data.shape}")
print(f"目标列: {choose}")
print(data.tail())

# 可视化原始时间序列
plt.figure(figsize=(12, 5))
plt.plot(data[choose].values)
plt.title(choose)
plt.xlabel('时间')
plt.ylabel(choose)
plt.tight_layout()
plt.savefig('raw_ts.png', dpi=100)
plt.show()

# =============================================
# 2. 滑动窗口构造监督学习数据
# =============================================
ts = data[choose].values.astype(float)
window = 12  # 用过去12个月预测下一个月

X, y = [], []
for i in range(len(ts) - window):
    X.append(ts[i:i + window] / 1e7)  # 伪标准化
    y.append(ts[i + window] / 1e7)

X, y = np.array(X), np.array(y)
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

# 训练集 / 测试集划分（后6个月为测试集）
X_train, X_test = X[:-6], X[-6:]
y_train, y_test = y[:-6], y[-6:]
print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

# =============================================
# 3. 辅助函数
# =============================================
def mape(y_true, y_pred):
    """平均绝对百分比误差"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def up_down_accuracy(y_true, y_pred):
    """方向准确率"""
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    correct = np.sum(np.sign(y_true_diff) == np.sign(y_pred_diff))
    return correct / len(y_true_diff) * 100

def output(y_true, y_pred, name):
    """输出评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    acc = up_down_accuracy(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"MAPE: {mape_val:.4f}%")
    print(f"方向准确率: {acc:.1f}%")
    return mae, mse, mape_val, acc

# =============================================
# 4. ElasticNet 回归模型
# =============================================
enet = ElasticNet(alpha=0.1, l1_ratio=0.9, random_state=3)
enet.fit(X_train, y_train)

# 训练集预测（用于计算残差方差）
enet_train_pred = enet.predict(X_train)
var1 = np.var(y_train - enet_train_pred)
print(f"\nElasticNet 训练集残差方差: {var1:.6f}")

# 测试集预测
enet_test_pred = enet.predict(X_test)
output(y_test, enet_test_pred, "ElasticNet")

# =============================================
# 5. SARIMA 季节性时间序列模型
# =============================================
model_sarima = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model_sarima.fit(disp=False)
print(results.summary())

# 训练集拟合值 & 残差方差
sarima_train_pred = results.fittedvalues
var2 = np.var(y_train - sarima_train_pred)
print(f"\nSARIMA 训练集残差方差: {var2:.6f}")

# 向前6步预测
sarima_forecast = results.forecast(steps=6)
output(y_test, sarima_forecast, "SARIMA")

# =============================================
# 6. 逆方差加权融合
# =============================================
coef1 = (1 / var1) / (1 / var1 + 1 / var2)
coef2 = (1 / var2) / (1 / var1 + 1 / var2)
print(f"\nElasticNet 权重: {coef1:.4f}")
print(f"SARIMA 权重: {coef2:.4f}")

final_pred = coef1 * enet_test_pred + coef2 * sarima_forecast

# 评估融合模型
mae_final, mse_final, mape_final, acc_final = output(y_test, final_pred, "融合模型")

# =============================================
# 7. 可视化结果
# =============================================
# 测试集对比图
test_indices = range(len(y_test))
plt.figure(figsize=(10, 5))
plt.plot(test_indices, y_test, 'o-', label='真实值', linewidth=2)
plt.plot(test_indices, final_pred, 's--', label='融合预测', linewidth=2)
plt.plot(test_indices, enet_test_pred, '^:', label='ElasticNet', alpha=0.7)
plt.plot(test_indices, sarima_forecast, 'd:', label='SARIMA', alpha=0.7)
plt.legend()
plt.title('测试集预测对比')
plt.xlabel('测试样本')
plt.ylabel('标准化值')
plt.tight_layout()
plt.savefig('test_comparison.png', dpi=100)
plt.show()

# =============================================
# 8. 汇总结果
# =============================================
print("\n" + "=" * 50)
print("模型评估汇总")
print("=" * 50)
print(f"{'模型':<15} {'MAE':<10} {'MAPE':<10} {'方向准确率':<10}")
print("-" * 50)

# ElasticNet
enet_mae = mean_absolute_error(y_test, enet_test_pred)
enet_mape = mape(y_test, enet_test_pred)
enet_acc = up_down_accuracy(y_test, enet_test_pred)
print(f"{'ElasticNet':<15} {enet_mae:<10.4f} {enet_mape:<10.4f} {enet_acc:<10.1f}%")

# SARIMA
sarima_mae = mean_absolute_error(y_test, sarima_forecast)
sarima_mape_val = mape(y_test, sarima_forecast)
sarima_acc = up_down_accuracy(y_test, sarima_forecast)
print(f"{'SARIMA':<15} {sarima_mae:<10.4f} {sarima_mape_val:<10.4f} {sarima_acc:<10.1f}%")

# 融合
print(f"{'融合模型':<15} {mae_final:<10.4f} {mape_final:<10.4f} {acc_final:<10.1f}%")
print("=" * 50)
