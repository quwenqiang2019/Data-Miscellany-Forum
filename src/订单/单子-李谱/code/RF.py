import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
# #显示所有列，把行显示设置成最大
# pd.set_option('display.max_columns', None)
# #显示所有行，把列显示设置成最大
# pd.set_option('display.max_rows', None)

#  读取数据集
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_excel(os.path.join(base_dir, 'data', 'Rice_MSC_Dataset.xlsx'))
df = pd.DataFrame(data)
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())


# 提取目标变量和特征变量
target = 'CLASS'
features = df.columns.drop(target)
print(df['CLASS'].value_counts()) # 顺便查看一下样本是否平衡
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=True, random_state=0)


# 将划分后的数据重构为dataframe
train_X = pd.DataFrame(X_train, columns=features)
train_y = pd.DataFrame(y_train, columns=[target])
train_df = pd.concat([train_X, train_y],axis = 1).reset_index(drop=True)

test_X = pd.DataFrame(X_test, columns=features)
test_y = pd.DataFrame(y_test, columns=[target])
test_df = pd.concat([test_X, test_y],axis = 1).reset_index(drop=True)


# 训练集数据预处理
le = LabelEncoder()  # 对于标签采用LabelEncoder 编码：文本型数据转为数值型数据
train_df = train_df.dropna()    #  删除缺失值
train_df['CLASS'] = le.fit_transform(train_df['CLASS'])  # LabelEncoder转换
X_train = train_df.iloc[:,:-1]
y_train = train_df['CLASS']


# 模型的构建与训练
model = RandomForestClassifier()
model.fit(X_train, y_train)


# 测试集数据预处理
test_df = test_df.dropna()
test_df['CLASS'] = le.transform(test_df['CLASS'])
X_test = test_df.iloc[:,:-1]
y_test = test_df['CLASS']


# 模型推理与评价
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) # 准确率acc
cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
cr = classification_report(y_test, y_pred) # 分类报告
print(acc)
print(cm)
print(cr)
print("----------------------------- precision（精确率）-----------------------------")
precision_score_average_None = precision_score(y_test, y_pred, average=None)
precision_score_average_micro = precision_score(y_test, y_pred, average='micro')
precision_score_average_macro = precision_score(y_test, y_pred, average='macro')
precision_score_average_weighted = precision_score(y_test, y_pred, average='weighted')
print('precision_score_average_None = ', precision_score_average_None)
print('precision_score_average_micro = ', precision_score_average_micro)
print('precision_score_average_macro = ', precision_score_average_macro)
print('precision_score_average_weighted = ', precision_score_average_weighted)

print("\n\n----------------------------- recall（召回率）-----------------------------")
recall_score_average_None = recall_score(y_test, y_pred, average=None)
recall_score_average_micro = recall_score(y_test, y_pred, average='micro')
recall_score_average_macro = recall_score(y_test, y_pred, average='macro')
recall_score_average_weighted = recall_score(y_test, y_pred, average='weighted')
print('recall_score_average_None = ', recall_score_average_None)
print('recall_score_average_micro = ', recall_score_average_micro)
print('recall_score_average_macro = ', recall_score_average_macro)
print('recall_score_average_weighted = ', recall_score_average_weighted)

print("\n\n----------------------------- F1-value-----------------------------")
f1_score_average_None = f1_score(y_test, y_pred, average=None)
f1_score_average_micro = f1_score(y_test, y_pred, average='micro')
f1_score_average_macro = f1_score(y_test, y_pred, average='macro')
f1_score_average_weighted = f1_score(y_test, y_pred, average='weighted')
print('f1_score_average_None = ', f1_score_average_None)
print('f1_score_average_micro = ', f1_score_average_micro)
print('f1_score_average_macro = ', f1_score_average_macro)
print('f1_score_average_weighted = ', f1_score_average_weighted)


# 提取特征重要性
feature_importance = model.feature_importances_
feature_names = features
# 创建特征重要性的DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
# 对特征重要性进行排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_excel(os.path.join(base_dir, 'result', 'RF特征重要性排序.xlsx'))
# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'result', 'RF-feature-imprtance-top20.jpg'), bbox_inches='tight')
plt.show()