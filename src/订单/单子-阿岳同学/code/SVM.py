import os
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
使用线性支持向量机（SVM）模型和递归特征消除（RFE）进行特征重要性分析
'''


# 准备数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_csv(os.path.join(base_dir, 'data', 'feature selection.csv'))
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'Landcover'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 训练线性支持向量机（SVR）模型
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)

# 使用递归特征消除（RFE）进行特征选择
selector = RFE(svr, n_features_to_select=1, step=1)
selector = selector.fit(X_train, y_train)

# 获取特征排名
feature_ranking = selector.ranking_

# 创建特征排名的DataFrame
ranking_df = pd.DataFrame({'Feature': features, 'Ranking': feature_ranking})

# 对特征排名进行排序
ranking_df = ranking_df.sort_values(by='Ranking')
ranking_df.to_excel(os.path.join(base_dir, 'result', 'svm特征重要性排序.xlsx'))
# 可视化特征排名
plt.figure(figsize=(10, 6))
sns.barplot(x='Ranking', y='Feature', data=ranking_df)
plt.title('Feature Ranking from Linear SVM')
plt.xlabel('Ranking')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'result', 'svm.jpg'), bbox_inches='tight', dpi=600)
plt.show()
