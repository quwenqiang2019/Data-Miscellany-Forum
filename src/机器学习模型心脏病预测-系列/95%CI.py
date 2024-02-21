import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# 加载数据集
data = pd.read_csv(r'Dataset.csv')

# 假设数据集中的最后一列是目标变量，其余列是特征
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 建立逻辑回归模型
model = LogisticRegression()

# 交叉验证计算指标
kf = KFold(n_splits=5, shuffle=True, random_state=0)

specificity_scores = []
sensitivity_scores = []
accuracy_scores = []
auc_scores = []


for train_index, test_index in kf.split(y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    specificity_scores.append(specificity)
    sensitivity_scores.append(sensitivity)

    auc_scores.append(roc_auc_score(y_test, y_pred_proba))
    accuracy_scores.append(accuracy_score(y_test, y_pred))


# 计算95%置信区间
def calculate_confidence_interval(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)
    z = 1.96  # 95% 置信水平的Z值
    lower_bound = mean - (z * std / np.sqrt(n))
    upper_bound = mean + (z * std / np.sqrt(n))
    return lower_bound, upper_bound

auc_ci = calculate_confidence_interval(auc_scores)
accuracy_ci = calculate_confidence_interval(accuracy_scores)
specificity_ci = calculate_confidence_interval(specificity_scores)
sensitivity_ci = calculate_confidence_interval(sensitivity_scores)

print("AUC 95% 置信区间:", auc_ci)
print("准确率 95% 置信区间:", accuracy_ci)
print("特异度 95% 置信区间:", specificity_ci)
print("灵敏度 95% 置信区间:", sensitivity_ci)
