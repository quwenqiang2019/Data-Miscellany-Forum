import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns


# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)
print(df)
# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 将特征值转换为非负数
X_positive = np.abs(X_train)

# 进行卡方分析，通过计算每个特征与目标变量之间的卡方统计量，来判断哪些特征与目标变量最相关。
chi_scores, p_values = chi2(X_positive, y_train)

# 打印每个特征的卡方统计量和 p 值
for i, feature_name in enumerate(features):
    print(f"特征 '{feature_name}' 的卡方统计量：{chi_scores[i]}, p 值：{p_values[i]}")

# 可视化卡方统计量
sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.bar(features, chi_scores, color='skyblue')
plt.xticks(rotation=45)
plt.xlabel('特征')
plt.ylabel('卡方统计量')
plt.title('各特征的卡方统计量')
plt.show()