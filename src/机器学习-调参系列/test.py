import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sparrow_algorithm import SparrowAlgorithm

# 生成一些示例数据集
X, y = np.random.rand(100, 5), np.random.randint(0, 2, 100)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型性能评价函数
def evaluate_model(learning_rate, n_estimators):
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# 定义参数搜索范围和麻雀算法参数
parameters = {'learning_rate': (0.01, 0.1), 'n_estimators': (50, 200)}
sparrow = SparrowAlgorithm(population_size=20, iterations=50, mutation_rate=0.1)

# 使用麻雀算法优化参数
best_params = sparrow.optimize(evaluate_model, parameters)

print("最优参数：", best_params)

