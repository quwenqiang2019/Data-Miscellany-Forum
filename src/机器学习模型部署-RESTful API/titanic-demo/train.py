from seaborn import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
import joblib

warnings.filterwarnings("ignore")  # 忽略模块变动警告

df = load_dataset("titanic")  # 加载泰坦尼克数据集
X = df[["pclass", "sex", "embarked"]]  # 特征
y = df["alive"]  # 目标
X = pd.get_dummies(X)  # 独热编码


model = RandomForestClassifier()  # 随机森林
np.mean(cross_val_score(model, X, y, cv=5))  # 5 次交叉验证求平均

model.fit(X, y)  # 训练模型
joblib.dump(model, "titanic.pkl")  # 保存模型