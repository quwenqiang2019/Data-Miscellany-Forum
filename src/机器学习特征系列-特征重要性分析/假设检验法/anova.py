import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif

from sklearn.datasets import make_classification

# 生成模拟数据集
X, y = make_classification(n_samples=100, n_features=4, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
# 特征名称
feature_names = ['内功心法强度', '招式复杂度', '使用难度', '对体力的要求']
print(X)
print(y)



# 进行 ANOVA 分析
f_scores, p_values = f_classif(X, y)
# 打印每个特征的 F 值和 p 值
for i, feature_name in enumerate(feature_names):
    print(f"特征 '{feature_name}' 的 F 值：{f_scores[i]}, p 值：{p_values[i]}")
# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.bar(feature_names, f_scores, color='skyblue')
plt.title("特征重要性：ANOVA F 值")
plt.xlabel("特征")
plt.ylabel("F 值")
plt.xticks(rotation=45)
plt.show()

