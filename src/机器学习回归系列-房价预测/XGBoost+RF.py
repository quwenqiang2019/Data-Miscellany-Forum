import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 导入数据
filename = 'data.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, names=names, delim_whitespace=True)
print(dataset)
df = pd.DataFrame(dataset)

#  划分数据集
features = names[:-1]
target = ['MEDV']
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 训练 XGBoost 模型
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# 训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 创建融合模型输入
blend_X = np.vstack([xgb_preds, rf_preds]).T
blend_model = LinearRegression()
blend_model.fit(blend_X, y_test)
blend_preds = blend_model.predict(blend_X)

# 评估指标
xgb_mse = mean_squared_error(y_test, xgb_preds)
rf_mse = mean_squared_error(y_test, rf_preds)
blend_mse = mean_squared_error(y_test, blend_preds)

xgb_r2 = r2_score(y_test, xgb_preds)
rf_r2 = r2_score(y_test, rf_preds)
blend_r2 = r2_score(y_test, blend_preds)


# 可视化分析
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
# 绘图1：真实值 vs 预测值（融合模型）
plt.figure(figsize=(8, 6))
plt.scatter(y_test, blend_preds, c='darkorange', label="Blended Prediction", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("图 1：融合模型预测结果 vs 实际值")
plt.legend()
plt.tight_layout()
plt.show()

# 绘图2：MSE 对比条形图
plt.figure(figsize=(8, 6))
mse_vals = [xgb_mse, rf_mse, blend_mse]
r2_vals = [xgb_r2, rf_r2, blend_r2]
models = ['XGBoost', 'Random Forest', 'Blended']
colors = ['#1f77b4', '#2ca02c', '#d62728']
sns.barplot(x=models, y=mse_vals, palette=colors)
plt.ylabel("MSE")
plt.title("图 2：各模型的均方误差（MSE）对比")
plt.tight_layout()
plt.show()

# 绘图3：R² 分数对比
plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=r2_vals, palette=colors)
plt.ylabel("R² Score")
plt.title("图 3：各模型的R^2分数对比")
plt.tight_layout()
plt.show()

# 绘图4：XGBoost 特征重要性图
xgb_feat_imp = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
xgb_feat_imp.head(10).plot(kind='bar', color='dodgerblue')
plt.title("图 4：XGBoost 模型前10重要特征")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()