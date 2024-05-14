import numpy as np


n = np.array([[0, -1, 3], [1, 2, 3]])
print(n)
feature_importance = np.abs(n)
print(feature_importance)
# 计算每个主成分中原始特征的权重（系数）和
feature_importance_sum = np.sum(feature_importance, axis=0)
print(feature_importance_sum)