import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# 准备数据
data = pd.read_csv(r'data.csv')
df = pd.DataFrame(data)
print(df.describe())

# 提取目标变量和特征变量
target = 'Type'
features = df.columns.drop(target)
print(data["Type"].value_counts()) # 顺便查看一下样本是否平衡

# 划分训练集和测试集
df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 模型的构建与训练
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# 提取特征重要性
feature_importance = model.feature_importances_
feature_names = features

# 创建特征重要性的DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
print(importance_df)

# 对特征重要性进行排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
