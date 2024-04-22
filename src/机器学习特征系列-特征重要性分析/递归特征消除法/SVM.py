from sklearn.svm import SVC
from sklearn.feature_selection import RFE
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


# 创建支持向量机分类器作为选择器的基模型
model = SVC(kernel="linear", C=1)
# RFE模型定义，选择保留1个最重要的特征
selector = RFE(estimator=model, n_features_to_select=1, step=1)
selector.fit(X_train, y_train)

# 获取特征排名
feature_ranking = selector.ranking_

# 创建特征排名的DataFrame
ranking_df = pd.DataFrame({'Feature': features, 'Ranking': feature_ranking})

# 对特征排名进行排序
ranking_df = ranking_df.sort_values(by='Ranking')

# 可视化特征排名
plt.figure(figsize=(10, 6))
sns.barplot(x='Ranking', y='Feature', data=ranking_df)
plt.title('Feature Ranking from Linear SVC')
plt.xlabel('Ranking')
plt.ylabel('Feature')
plt.show()
