import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_excel(os.path.join(base_dir, 'data', '机器学习数据.xlsx'))
df = pd.DataFrame(data)
print(df.shape)
print(df.head())
cat_cols = [col for col in df.columns if df[col].dtype == "object"] # 类别型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"] # 数值型变量名

# 提取目标变量和特征变量
target = 'class'
features = df.columns.drop(target)
print(df['class'].value_counts()) # 顺便查看一下样本是否平衡
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


# 模型的构建与训练
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

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
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'result', 'DT-ROC.jpg'), bbox_inches='tight')
plt.show()

# 提取特征重要性
feature_importance = model.feature_importances_
feature_names = features
# 创建特征重要性的DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
# 对特征重要性进行排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_excel(os.path.join(base_dir, 'result', 'DT特征重要性排序.xlsx'))
# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'result', 'DT-feature-imprtance-top20.jpg'), bbox_inches='tight')
plt.show()