import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 定义平均模型
class AverageModel:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y, X_val, y_val):
        for model in self.models:
            if isinstance(model, lgb.LGBMRegressor):
                model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='rmse',
                          callbacks=[lgb.early_stopping(stopping_rounds=100)])
            elif isinstance(model, xgb.XGBRegressor):
                model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='rmse',
                          early_stopping_rounds=model.get_params()['early_stopping_rounds'], verbose=False)

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return sum(predictions) / len(predictions)



#  1、数据读取及预处理
df = pd.read_csv('dataset.csv')
df = df.iloc[:,:-1]
# 缺失值处理
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())
# 编码
le = LabelEncoder()
df['origin'] = le.fit_transform(df['origin'])
cols = df.columns
print(df.head())

#  2、划分数据集
features = cols[1:]
target = ['mpg']
# 首先将数据集划分为训练集和测试集
X_temp, X_test, y_temp, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
# 然后将训练集进一步划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)
# 输出数据集的大小
print(f"训练集维度: {X_train.shape}")
print(f"验证集维度: {X_val.shape}")
print(f"测试集维度: {X_test.shape}")



# LightGBM模型参数
params_lgb = {
    'learning_rate': 0.02,  # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'boosting_type': 'gbdt',  # 提升方法，这里使用梯度提升树（Gradient Boosting Decision Tree，简称GBDT）
    'objective': 'mse',  # 损失函数
    'metric': 'rmse',  # 评估指标
    'num_leaves': 127,  # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合
    'verbose': -1,  # 控制 LightGBM 输出信息的详细程度，-1表示无输出，0表示最少输出，正数表示输出更多信息
    'seed': 42,  # 随机种子，用于重现模型的结果
    'n_jobs': -1,  # 并行运算的线程数量，-1表示使用所有可用的CPU核心
    'feature_fraction': 0.8,  # 每棵树随机选择的特征比例，用于增加模型的泛化能力
    'bagging_fraction': 0.9,  # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
    'bagging_freq': 4  # 每隔多少次迭代进行一次bagging操作，用于增加模型的泛化能力
}
model_lgb = lgb.LGBMRegressor(**params_lgb)

# XGBoost模型参数
params_xgb = {
    'learning_rate': 0.02,  # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'booster': 'gbtree',  # 提升方法，这里使用梯度提升树（Gradient Boosting Tree）
    'objective': 'reg:squarederror',  # 损失函数
    'max_leaves': 127,  # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合
    'verbosity': 1,  # 控制 XGBoost 输出信息的详细程度，0表示无输出，1表示输出进度信息
    'seed': 42,  # 随机种子，用于重现模型的结果
    'nthread': -1,  # 并行运算的线程数量，-1表示使用所有可用的CPU核心
    'colsample_bytree': 0.6,  # 每棵树随机选择的特征比例，用于增加模型的泛化能力
    'subsample': 0.7,  # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
    'early_stopping_rounds': None  # 早停参数在fit时单独设置
}


model_xgb = xgb.XGBRegressor(**params_xgb)


# 3、 创建平均模型
average_model = AverageModel([model_lgb, model_xgb])

# 4、 训练模型
average_model.fit(X_train, y_train, X_val, y_val)

# 5、 进行预测
y_test_pred = average_model.predict(X_test)
y_train_pred = average_model.predict(X_train)

# 6、 模型评价
y_test_pred = np.array(y_test_pred)
mse = metrics.mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_test_pred)
r2 = metrics.r2_score(y_test, y_test_pred)

print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("平均绝对误差 (MAE):", mae)
print("拟合优度 (R-squared):", r2)



# 7、 预测结果可视化
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
# 训练集预测值与真实值的对比
plt.plot(list(range(0,len(X_train))),y_train,marker='o')
plt.plot(list(range(0,len(X_train))),y_train_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.xlabel('序列')
plt.ylabel('油耗')
plt.title('训练集预测值与真实值的对比')
plt.show()
# 验证集预测值与真实值的对比
plt.plot(list(range(0,len(X_test))),y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_test_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.xlabel('序列')
plt.ylabel('油耗')
plt.title('验证集预测值与真实值的对比')
plt.show()
