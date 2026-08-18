import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.stats import t
from sklearn.preprocessing import MinMaxScaler


# 加载数据集
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_excel(os.path.join(base_dir, 'data', '机器学习数据.xlsx'))
df = pd.DataFrame(data)

# 提取目标变量和特征变量
target = 'class'
features = df.columns.drop(target)
X = df[features]
y = df[[target]]


# 建立模型
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

    # 归一化
    mm1 = MinMaxScaler()  # 特征进行归一化
    X_train_m = mm1.fit_transform(X_train)
    mm2 = MinMaxScaler()  # 标签进行归一化
    y_train_m = mm2.fit_transform(y_train)

    model.fit(X_train_m, y_train_m)

    # 对测试集特征进行相同规则mm1的归一化处理，然后输入到模型进行预测
    X_test_m = mm1.transform(X_test)
    predicted_y_m = model.predict(X_test_m)
    y_pred = mm2.inverse_transform(np.reshape(predicted_y_m, (-1, 1)))
    y_pred_proba = model.predict_proba(X_test_m)[:, 1]

    # y_pred = model.predict(X_test)
    # y_pred_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    specificity_scores.append(specificity)
    sensitivity_scores.append(sensitivity)

    auc_scores.append(roc_auc_score(y_test, y_pred_proba))
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(auc_scores, accuracy_scores, specificity_scores, sensitivity_scores)
print(np.mean(auc_scores),np.mean(accuracy_scores),np.mean(specificity_scores),np.mean(sensitivity_scores),)

# 计算95%置信区间
# def calculate_confidence_interval(scores):
#     mean = np.mean(scores)
#     std = np.std(scores)
#     n = len(scores)
#     z = 1.96  # 95% 置信水平的Z值
#     lower_bound = mean - (z * std / np.sqrt(n))
#     upper_bound = mean + (z * std / np.sqrt(n))
#     return lower_bound, upper_bound



def calculate_confidence_interval(scores):
    # 计算样本均值和标准差
    sample_mean = np.mean(scores)
    sample_std = np.std(scores, ddof=1)  # 使用ddof=1来计算样本标准差

    # 置信水平和自由度
    confidence_level = 0.95  # 置信水平为95%
    df = len(scores) - 1  # 自由度为n-1

    # 计算t分布的置信区间
    alpha = 1 - confidence_level
    t_value = t.ppf(1 - alpha / 2, df)  # 计算t分布的临界值
    margin_of_error = t_value * sample_std / np.sqrt(len(scores))  # 计算误差边界


    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    return lower_bound, upper_bound


auc_ci = calculate_confidence_interval(auc_scores)
accuracy_ci = calculate_confidence_interval(accuracy_scores)
specificity_ci = calculate_confidence_interval(specificity_scores)
sensitivity_ci = calculate_confidence_interval(sensitivity_scores)


# 计算95%置信区间
# confidence_level = 0.95
# alpha = 1 - confidence_level
# lower_percentile = alpha / 2 * 100
# upper_percentile = (1 - alpha / 2) * 100
#
# auc_ci = np.percentile(auc_scores, [lower_percentile, upper_percentile])
# accuracy_ci = np.percentile(accuracy_scores, [lower_percentile, upper_percentile])
# specificity_ci = np.percentile(specificity_scores, [lower_percentile, upper_percentile])
# sensitivity_ci = np.percentile(sensitivity_scores, [lower_percentile, upper_percentile])




print("AUC 95% 置信区间:", auc_ci)
print("准确率 95% 置信区间:", accuracy_ci)
print("特异度 95% 置信区间:", specificity_ci)
print("灵敏度 95% 置信区间:", sensitivity_ci)
