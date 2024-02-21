import numpy as np
from sklearn.utils import resample
# Bootstrap 方法计算置信区间
def calculate_confidence_interval(scores):
    n_iterations = 1000
    n_size = len(scores)
    stats = []
    for _ in range(n_iterations):
        sample = resample(scores, n_samples=n_size)
        stat = np.mean(sample)
        stats.append(stat)
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    return lower, upper

# 计算置信区间
accuracy_ci = calculate_confidence_interval(0.8)
print(accuracy_ci)
