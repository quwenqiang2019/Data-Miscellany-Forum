import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# 加载数据集
data = pd.read_csv(r'Dataset.csv')

# 假设数据集中X是特征，y是标签
X = data.drop('target', axis=1)
y = data['target']

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 计算AUC
auc = roc_auc_score(y_test, y_pred_proba)

# 计算准确率、特异度和灵敏度
accuracy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# 自助法计算95%置信区间
n_bootstraps = 1000
auc_bootstrap = []
accuracy_bootstrap = []
specificity_bootstrap = []
sensitivity_bootstrap = []

for _ in range(n_bootstraps):
    indices = np.random.choice(len(y_test), len(y_test), replace=True)
    y_test_bootstrap = y_test.iloc[indices]
    y_pred_proba_bootstrap = y_pred_proba[indices]
    y_pred_bootstrap = (y_pred_proba_bootstrap > 0.5).astype(int)

    auc_bootstrap.append(roc_auc_score(y_test_bootstrap, y_pred_proba_bootstrap))
    accuracy_bootstrap.append(accuracy_score(y_test_bootstrap, y_pred_bootstrap))
    tn, fp, fn, tp = confusion_matrix(y_test_bootstrap, y_pred_bootstrap).ravel()
    specificity_bootstrap.append(tn / (tn + fp))
    sensitivity_bootstrap.append(tp / (tp + fn))

# 计算95%置信区间
confidence_level = 0.95
alpha = 1 - confidence_level
lower_percentile = alpha / 2 * 100
upper_percentile = (1 - alpha / 2) * 100

auc_ci = np.percentile(auc_bootstrap, [lower_percentile, upper_percentile])
accuracy_ci = np.percentile(accuracy_bootstrap, [lower_percentile, upper_percentile])
specificity_ci = np.percentile(specificity_bootstrap, [lower_percentile, upper_percentile])
sensitivity_ci = np.percentile(sensitivity_bootstrap, [lower_percentile, upper_percentile])

print("AUC 95% Confidence Interval:", auc_ci)
print("Accuracy 95% Confidence Interval:", accuracy_ci)
print("Specificity 95% Confidence Interval:", specificity_ci)
print("Sensitivity 95% Confidence Interval:", sensitivity_ci)
