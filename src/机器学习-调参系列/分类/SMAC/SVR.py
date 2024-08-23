import pandas as pd
from sklearn.model_selection import train_test_split
from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
## 数据基本信息
print(df.head())
print(df.info())
print(df.shape)
print(df.columns)
print(df.dtypes)
cat_cols = [col for col in df.columns if df[col].dtype == "object"] # 类别型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"] # 数值型变量名

# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡

# 划分训练集和测试集
# df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 定义训练函数
def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    return 1 - np.mean(scores)

# 定义配置空间
configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

# 定义优化环境的方案对象
scenario = Scenario(configspace, deterministic=True, n_trials=200)

# 使用SMAC寻找最优的配置或参数
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()

print("Best found configuration: ", incumbent)