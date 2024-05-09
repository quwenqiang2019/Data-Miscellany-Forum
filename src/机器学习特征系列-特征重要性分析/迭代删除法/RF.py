from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import numpy as np

# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features].values, df[target].values, test_size=0.2, random_state=0)


# 创建随机森林分类器作为选择器的基模型
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 获取测试数据上的基准准确率
base_acc = accuracy_score(y_test, model.predict(X_test))

# 初始化一个空列表来存储特征重要性
importances = []

# 遍历所有特征，并逐个移除特征后重新训练模型，计算准确率变化
for i in range(X_train.shape[1]):
    X_temp = np.delete(X_train, i, axis=1)  # 移除一个特征
    model.fit(X_temp, y_train)  # 重新训练模型
    acc = accuracy_score(y_test, model.predict(np.delete(X_test, i, axis=1)))  # 计算测试准确率
    importances.append(base_acc - acc)  # 计算特征重要性，即准确率变化量

# 获取特征排名
feature_ranking = importances
# 创建特征排名的DataFrame
ranking_df = pd.DataFrame({'Feature': features, 'Ranking': feature_ranking})
print(ranking_df)

ranking_df = ranking_df.sort_values(by='Ranking')
print(ranking_df)

# 绘制特征重要性柱状图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances, color='skyblue')
plt.xlabel('特征编号')
plt.ylabel('准确率变化量')
plt.title('Leave-one-out 特征重要性分析')
plt.xticks(range(len(importances)), range(1, len(importances) + 1))
plt.show()

