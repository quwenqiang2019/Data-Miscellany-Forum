import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics

# 导入数据
filename = 'data.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, names=names, delim_whitespace=True)
df = pd.DataFrame(dataset)
print(df)

# 首先将数据集划分为训练集和测试集
X_temp, X_test, y_temp, y_test = train_test_split(df.iloc[:,0:7], df['MEDV'], test_size=0.2, random_state=42)
# 然后将训练集进一步划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)  # 0.125 x 0.8 = 0.1
# 输出数据集的大小
print(f"训练集维度: {X_train.shape}")
print(f"验证集维度: {X_val.shape}")
print(f"测试集维度: {X_test.shape}")

# 归一化目标变量
def normalize_dataframe(y_train, y_val, y_test):
    scaler = MinMaxScaler()
    scaler.fit(np.array(y_train).reshape(-1, 1))  # 在训练集上拟合归一化模型 MinMaxScaler输入数据形状为二维数组
    train = pd.DataFrame(scaler.transform(np.array(y_train).reshape(-1, 1)), index=y_train.index)
    val = pd.DataFrame(scaler.transform(np.array(y_val).reshape(-1, 1)), index=y_val.index)
    test = pd.DataFrame(scaler.transform(np.array(y_test).reshape(-1, 1)), index=y_test.index)
    return train, val, test, scaler

train_y, val_y, test_y, scaler_y = normalize_dataframe(y_train, y_val, y_test)

# 定义和训练模型
# LightGBM模型参数
params_lgb = {
             'learning_rate': 0.02,          # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
             'boosting_type': 'gbdt',        # 提升方法，这里使用梯度提升树（Gradient Boosting Decision Tree，简称GBDT）
            'objective': 'mse',             # 损失函数
            'metric': 'rmse',               # 评估指标
            'num_leaves': 127,              # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合
            'verbose': -1,                  # 控制 LightGBM 输出信息的详细程度，-1表示无输出，0表示最少输出，正数表示输出更多信息
            'seed': 42,                     # 随机种子，用于重现模型的结果
            'n_jobs': -1,                   # 并行运算的线程数量，-1表示使用所有可用的CPU核心
            'feature_fraction': 0.8,        # 每棵树随机选择的特征比例，用于增加模型的泛化能力
            'bagging_fraction': 0.9,        # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
            'bagging_freq': 4               # 每隔多少次迭代进行一次bagging操作，用于增加模型的泛化
              }
model_lgb = lgb.LGBMRegressor(**params_lgb)

# XGBoost模型参数
params_xgb = {
            'learning_rate': 0.02,          # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
            'booster': 'gbtree',            # 提升方法，这里使用梯度提升树（Gradient Boosting Tree）
            'objective': 'reg:squarederror',# 损失函数
            'max_leaves': 127,              # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合
            'verbosity': 1,                 # 控制 XGBoost 输出信息的详细程度，0表示无输出，1表示输出进度信息
            'seed': 42,                     # 随机种子，用于重现模型的结果
            'nthread': -1,                  # 并行运算的线程数量，-1表示使用所有可用的CPU核心
            'colsample_bytree': 0.6,        # 每棵树随机选择的特征比例，用于增加模型的泛化能力
            'subsample': 0.7,               # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
            'early_stopping_rounds': None   # 早停参数在fit时单独设置
            }

model_xgb = xgb.XGBRegressor(**params_xgb)

# 定义平均模型
class AverageModel:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y, X_val, y_val):
        for model in self.models:
            if isinstance(model, lgb.LGBMRegressor):
                model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=100)])
            elif isinstance(model, xgb.XGBRegressor):
                model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='rmse', early_stopping_rounds=model.get_params()['early_stopping_rounds'], verbose=False)

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return sum(predictions) / len(predictions)
# 创建平均模型
average_model = AverageModel([model_lgb, model_xgb])
# 训练模型
average_model.fit(X_train, train_y, X_val, val_y)

# 预测测试集
y_pred = average_model.predict(X_test)
print(y_pred)

# 评估模型
y_pred_list = y_pred.tolist()  # 或者 y_pred_array = np.array(y_pred)
mse = metrics.mean_squared_error(test_y, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(test_y, y_pred_list)
r2 = metrics.r2_score(test_y, y_pred_list)
print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("平均绝对误差 (MAE):", mae)
print("拟合优度 (R-squared):", r2)

# 绘制预测值与真实值对比图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
# 反归一化
train_min = np.min(y_train)
train_max = np.max(y_train)
pred = y_pred * (train_max - train_min) + train_min
y_test = np.array(y_test)
# 计算预测值 真实值差值的绝对值
alpha_values = abs(pred-y_test.reshape(-1))  # 值越大alpha越大
# 确保 alpha 值在 0 到 1 之间
alpha_values = np.clip(alpha_values, 0, 1)
plt.scatter(pred, y_test, color='blue', edgecolor='k', s=50, alpha=alpha_values, label='预测值 vs 真实值')
plt.title('预测值与真实值对比图', fontsize=16)
plt.xlabel('预测值', fontsize=14)
plt.ylabel('真实值', fontsize=14)
max_val = max(max(pred), max(y_test))
min_val = min(min(pred), min(y_test))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='x=y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()