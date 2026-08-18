import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)
print(df)
# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


# 对训练集做PCA主成分分析
pca = PCA(n_components='mle')
pca.fit(X_train)
var_ratio = pca.explained_variance_ratio_
for idx, val in enumerate(var_ratio, 1):
    print("Principle component %d: %.2f%%" % (idx, val * 100))
print("total: %.2f%%" % np.sum(var_ratio * 100))

# 打印出每个特征对于主成分的系数，这反映了原始特征的重要性
print(pca.components_)
# 计算原始特征与主成分的相关性（绝对值）
feature_importance = np.abs(pca.components_)
print(feature_importance)
# 计算每个主成分中原始特征的权重（系数）和
feature_importance_sum = np.sum(feature_importance, axis=0)
print(feature_importance_sum)
# 打印原始特征的重要性（贡献度）
print("\n原始特征的重要性（贡献度）:")
ranking_df = pd.DataFrame({'特征': features, '贡献度': feature_importance_sum})
print(ranking_df)
ranking_df = ranking_df.sort_values(by='贡献度')
print(ranking_df)

# 绘制特征重要性柱状图
sns.set(font_scale=1.2)
plt.rc('font', family=['SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance_sum)), feature_importance_sum, color='skyblue')
plt.xlabel('特征编号')
plt.ylabel('贡献度')
plt.title('PCA特征重要性分析')
plt.xticks(range(len(feature_importance_sum)), range(1, len(feature_importance_sum) + 1))
plt.show()

# 查看累计解释方差比率与主成分个数的关系
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(np.arange(1, len(var_ratio) + 1), np.cumsum(var_ratio), "-ro")
ax.set_title("Cumulative Explained Variance Ratio", fontsize=15)
ax.set_xlabel("number of components")
ax.set_ylabel("explained variance ratio(%)")
plt.show()


# 设定累计解释方差比率的目标，让sklearn自动选择最优的主成分个数
target = 0.9  # 保留原始数据集90%的变异
res = PCA(n_components=target).fit_transform(X_train)
print("original shape: ", X_train.shape)
print("transformed shape: ", res.shape)



# 选择两个主成分，并进行可视化
pca=PCA(n_components=2)  #加载PCA算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(X_train)#对样本进行降维
principalDf = pd.DataFrame(data = reduced_x, columns = ['principal component 1', 'principal component 2'])
print(principalDf)
y_train = np.array(y_train)
yes_x,yes_y=[],[]
no_x,no_y=[],[]
for i in range(len(reduced_x)):
    if y_train[i] ==1:
        yes_x.append(reduced_x[i][0])
        yes_y.append(reduced_x[i][1])
    elif y_train[i]==0:
        no_x.append(reduced_x[i][0])
        no_y.append(reduced_x[i][1])
plt.scatter(yes_x,yes_y,c='r',marker='x')
plt.scatter(no_x,no_y,c='b',marker='D')
plt.xlabel("First Main Component")
plt.ylabel("Second Main Component")
plt.show()



