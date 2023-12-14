import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 1、准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)

# 2、数据预处理

## 2.1 数据基本信息
print(df.info())
### 2.1.1 数据量
print(df.shape)
### 2.1.2 字段名和类型
print(df.columns)
print(df.dtypes)
cat_cols = [col for col in df.columns if df[col].dtype == "object"] # 类别型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"] # 数值型变量名

## 2.2 错误数据处理
for i in df.columns:
    print(df[i].value_counts())
    df["pcv"] = pd.to_numeric(df["pcv"], errors="coerce")

## 2.3 特征编码
#（略）

## 2.4 数据清洗
### 2.4.1 重复值
print('存在' if any(df.duplicated()) else '不存在', '重复观测值')
df.drop_duplicates()
### 2.4.2 缺失值处理
print(df.isnull())
print('不存在' if any(df.isnull()) else '存在', '缺失值')
print(df.isnull().sum())   #检测每列中缺失值的数量
print(df.isnull().T.sum())    #检测每行缺失值的数量
df.dropna()  # 直接删除记录
df.fillna(method='ffill')  # 前向填充
df.fillna(method='bfill')  # 后向填充
df.fillna(value=2)  # 值填充
df.fillna(value={'resting_blood_pressure': df['resting_blood_pressure'].mean()})  # 统计值填充
### 2.4.3 异常值处理
df1 = df['resting_blood_pressure']
# 标准差监测
xmean =  df1.mean()
xstd = df1.std()
print('存在' if any(df1 > xmean + 2 * xstd) else '不存在', '上限异常值')
print('存在' if any(df1 < xmean - 2 * xstd) else '不存在', '下限异常值')
# 箱线图监测
q1 = df1.quantile(0.25)
q3 = df1.quantile(0.75)
up = q3 + 1.5 * (q3 - q1)
dw = q1 - 1.5 * (q3 - q1)
print('存在' if any(df1 > up) else '不存在', '上限异常值')
print('存在' if any(df1 < dw) else '不存在', '下限异常值')
df1[df1 > up] = df1[df1 <  up].max()
df1[df1 < dw] = df1[df1 >  dw].min()

## 2.5 数据探索
### 2.5.1 特征分布
### 2.6.2 特征相关性


# 3 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡


# 4、 归一化



# 5、划分训练集和测试集
df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


# 6、特征重要性分析与筛选


# 7、模型的构建与训练
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)


# 8、模型的特征重要性分析


# 9、模型推理与评价
y_pred = model.predict(X_test)
print(y_pred)
y_scores = model.predict_proba(X_test)
print(y_scores[:, 1])
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
plt.show()


# 10、模型的优化与部署