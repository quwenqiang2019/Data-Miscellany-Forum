import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 准备数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(base_dir, 'data', 'Expression.csv'))

df = pd.DataFrame(data)
df = df.dropna()  # 直接删除记录


# 目标变量和特征变量
target = 'age'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features].values, df[target].values, test_size=0.2, random_state=0)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=0)
# model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# 提取特征重要性
feature_importance = model.feature_importances_
feature_names = features

# 创建特征重要性的DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# 对特征重要性进行排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df = importance_df.head(10)

# 可视化特征重要性
sns.set_style('darkgrid')
font1 = {'family': ['SimSun'], 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
plt.rcParams["axes.unicode_minus"] = False
plt.subplots_adjust(left=0.3)
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('RF特征重要性', fontproperties=font1)
plt.xlabel('重要性', fontproperties=font1)
plt.gca().set_ylabel('')
# plt.ylabel('变量', fontproperties=font1)
plt.savefig(os.path.join(base_dir, 'res', 'RF.tif'))
plt.show()


# 推理
y_pred_test = model.predict(X_test)


# 评价
plt.scatter(list(np.array(y_test).flatten()), list(np.array(y_pred_test).flatten()), s=50, c="red", label='测试样本点')
x2 = np.arange(np.min(y_test), np.max(y_test), 1)
y2 = x2
plt.plot(x2, y2, "blue", label='y = x')
plt.title("模型验证", fontproperties=font1)
plt.xlabel('测量值', fontproperties=font1)
plt.ylabel('预测值', fontproperties=font1)
plt.legend(fontsize=14)
plt.savefig(os.path.join(base_dir, 'res', 'RF验证1.tif'))
plt.show()

plt.plot(list(range(0, len(X_test))), y_test, marker='o', label='真实值')
plt.plot(list(range(0, len(X_test))), y_pred_test, marker='*', label='预测值')
plt.title('真实值与预测值对比', fontproperties=font1)
plt.legend()
plt.savefig(os.path.join(base_dir, 'res', 'RF验证2.tif'))
plt.show()


print('explained_variance_score:', explained_variance_score(y_test, y_pred_test))  # 越小效果越差
print('MAE:', mean_absolute_error(y_test, y_pred_test))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_test)))  # 越小效果越好
print('R² score:', r2_score(y_test, y_pred_test))  # 越小效果越差