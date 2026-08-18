from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 创建随机森林分类器作为选择器的基模型
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
# baseline = model.score(X_test, y_test)
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=1, scoring='accuracy')

# 获取特征排名
feature_ranking = result.importances_mean

# 创建特征排名的DataFrame
ranking_df = pd.DataFrame({'Feature': features, 'Ranking': feature_ranking})

# 对特征排名进行排序
ranking_df = ranking_df.sort_values(by='Ranking')

# 可视化特征排名
plt.figure(figsize=(10, 6))
sns.barplot(x='Ranking', y='Feature', data=ranking_df)
plt.title('Feature Ranking from Linear RF')
plt.xlabel('Ranking')
plt.ylabel('Feature')
plt.show()
