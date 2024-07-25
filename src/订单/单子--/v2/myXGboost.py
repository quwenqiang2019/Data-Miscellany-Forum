import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier  # 导入XGBoost分类器
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
data = pd.read_csv("train.csv")
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

# 下采样处理
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(df[features], df[target])

# XGBoost模型设置
model = XGBClassifier(random_state=42, n_estimators=100)

# 10倍交叉验证
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = []
for train_idx, test_idx in kfold.split(X_res, y_res):
    X_train, X_test = X_res.iloc[train_idx], X_res.iloc[test_idx]
    y_train, y_test = y_res.iloc[train_idx], y_res.iloc[test_idx]

    # 训练模型
    model.fit(X_train, y_train)

 # 预测
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    # 模型评价（可选，根据你的需求进行评价）
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 绘图部分（可选）
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # 打印评价指标（可选）
    print(cm)
    print("Accuracy:", acc)
    print(cr)