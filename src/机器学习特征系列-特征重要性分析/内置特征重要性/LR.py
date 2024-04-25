import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)


# 归一化
mm1 = MinMaxScaler()   # 特征进行归一化
X_train_m = mm1.fit_transform(X_train)
mm2 = MinMaxScaler()     # 标签进行归一化
y_train_m = mm2.fit_transform(y_train)

# 训练模型
model = LogisticRegression()
model.fit(X_train_m, y_train_m)

# 提取特征重要性
feature_importance = model.coef_[0]
feature_names = features

# 创建特征重要性的DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# 对特征重要性进行排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()