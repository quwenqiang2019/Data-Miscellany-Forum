import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 准备数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_csv(os.path.join(base_dir, 'data', 'feature selection.csv'))
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'Landcover'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建BP神经网络模型
bp_nn = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=0)

# 训练模型
bp_nn.fit(X_train_scaled, y_train)

# 获取每个特征的重要性
# 由于神经网络不直接提供特征重要性，我们可以通过观察权重来间接估计
# 获取输入层到第一个隐藏层的权重
weights = bp_nn.coefs_[0]
print(weights)

# 计算权重的平均绝对值，用作特征重要性的代理
importance = np.mean(np.abs(weights), axis=1)

# 创建特征重要性的DataFrame
nn_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})

# 对特征重要性进行排序
nn_importance_df = nn_importance_df.sort_values(by='Importance', ascending=False)
nn_importance_df.to_excel(os.path.join(base_dir, 'result', 'nn特征重要性排序.xlsx'))
# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=nn_importance_df)
plt.title('Feature Importance from BP Neural Network')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'result', 'nn.jpg'), bbox_inches='tight', dpi=600)
plt.show()
