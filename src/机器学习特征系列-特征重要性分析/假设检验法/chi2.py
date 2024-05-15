import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2

from sklearn.datasets import make_classification



# 使用示例数据集
X, y = make_classification(n_samples=100, n_features=4, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
feature_names = ['内功心法强度', '招式复杂度', '使用难度', '对体力的要求']
print(X)
print(y)
# 将特征值转换为非负数
X_positive = np.abs(X)

# 进行卡方检验
chi_scores, p_values = chi2(X_positive, y)

# 打印每个特征的卡方统计量和 p 值
for i, feature_name in enumerate(feature_names):
    print(f"特征 '{feature_name}' 的卡方统计量：{chi_scores[i]}, p 值：{p_values[i]}")

# 可视化卡方统计量
plt.figure(figsize=(10, 6))
plt.bar(feature_names, chi_scores, color='skyblue')
plt.xlabel('特征')
plt.ylabel('卡方统计量')
plt.title('各特征的卡方统计量')
plt.show()