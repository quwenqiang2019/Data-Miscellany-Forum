import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import shap

# 准备数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_csv(os.path.join(base_dir, 'data', 'pCR.csv'))
df = pd.DataFrame(data)
print(df)

# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
print(df["target"].value_counts()) # 顺便查看一下样本是否平衡

# 下采样数据
negative_eg = data[data['target'] == 0]
positive_eg = data[data['target'] == 1]
np.random.seed(seed=2)          #随机种子，是保证每次你执行这个代码，随机抽选的结果都是一样
negative_eg = negative_eg.sample(len(positive_eg)+1)
df = pd.concat([positive_eg, negative_eg])
print(df)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 模型的构建与训练
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 创建Explainer
explainer = shap.TreeExplainer(model, X_test)
# 以numpy数组的形式输出SHAP值
shap_values = explainer.shap_values(X_test)
# # 以SHAP的Explanation对象形式输出SHAP值
shap_obj = explainer(X_test)

# 特征分析
# shap.summary_plot(shap_values, X_test, show = False)
# plt.savefig(os.path.join(base_dir, 'result', 'summary_plot.png'), bbox_inches='tight', dpi=600)
# shap.plots.bar(shap_obj[:,:,0], show = False)
# plt.savefig(os.path.join(base_dir, 'result', 'summary_bar.png'), bbox_inches='tight', dpi=600)
# shap.plots.beeswarm(shap_obj[:,:,0], show = False)
# plt.savefig(os.path.join(base_dir, 'result', 'summary_beeswarm_0.png'), bbox_inches='tight', dpi=600)
shap.plots.beeswarm(shap_obj[:,:,1], show = False)
plt.savefig(os.path.join(base_dir, 'result', 'summary_beeswarm_1.png'), bbox_inches='tight', dpi=600)

# 模型推理与评价
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred) # 准确率acc
cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
cr = classification_report(y_test, y_pred) # 分类报告
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1) # 计算ROC曲线和AUC值,绘制ROC曲线
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(base_dir, 'result', 'DT.jpg'), bbox_inches='tight', dpi=600)
plt.show()

