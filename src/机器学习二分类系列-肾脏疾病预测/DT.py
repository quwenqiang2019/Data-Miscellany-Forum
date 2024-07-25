import pandas as  pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ===================================================读取数据并做大致分析=================================================
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

df = pd.read_csv('data.csv')
print('数据：', df, sep='\n')

cat_cols = [col for col in df.columns if df[col].dtype == "object"] # 类别型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"] # 数值型变量名
print(cat_cols)
print(num_cols)

# 划分数据集
train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)


# ============================================对训练集进行特征工程=========================================================
le = LabelEncoder()  # 对于标签采用LabelEncoder
ohe = OneHotEncoder(sparse=False)  # 对于特征采用OneHotEncoder
mm = MinMaxScaler()   # 特征进行归一化

# 1、删除第一列id
train_df.drop("id",axis=1,inplace=True)

# 2、缺失值处理（类别型+数值型）
train_df[num_cols[1:]] = train_df[num_cols[1:]].fillna(train_df[num_cols[1:]].mean())
train_df_num_cols = train_df[num_cols[1:]]
for i in cat_cols:
    train_df[i] = train_df[i].fillna(train_df[i].mode()[0])

# 3、异常值处理（数值型）
for i in num_cols[1:]:
    train_df_num_col = train_df_num_cols[i]
    xmean = train_df_num_col.mean()
    xstd = train_df_num_col.std()
    train_df_num_col[train_df_num_col > xmean + 2 * xstd] = train_df_num_col[train_df_num_col < xmean + 2 * xstd].max()
    train_df_num_col[train_df_num_col < xmean - 2 * xstd] = train_df_num_col[train_df_num_col > xmean - 2 * xstd].min()

# 4、编码(类别变量)+归一化(数值变量)
train_df_num_cols = mm.fit_transform(train_df_num_cols)
train_df_num_cols = pd.DataFrame(data=train_df_num_cols, columns=num_cols[1:])
train_df_cat_cols = pd.DataFrame(ohe.fit_transform(train_df[cat_cols[:-1]].values), columns=ohe.get_feature_names()).reset_index(drop=True)
train_df_label = pd.DataFrame(le.fit_transform(train_df['classification'].values), columns=['classification']).reset_index(drop=True)

# # 5、重构
train_df = pd.concat([train_df_num_cols.reset_index(drop=True), train_df_cat_cols, train_df_label], axis=1)
X_train = train_df.iloc[:,:-1]
y_train = train_df['classification']

# 模型的构建与训练
# model = RandomForestClassifier()
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# =================================测试集做相同的特征处理================================================
# 1、删除第一列id
test_df.drop("id",axis=1,inplace=True)

# 2、缺失值处理（类别型+数值型）
test_df[num_cols[1:]] = test_df[num_cols[1:]].fillna(test_df[num_cols[1:]].mean())
test_df_num_cols = test_df[num_cols[1:]]
for i in cat_cols:
    test_df[i] = test_df[i].fillna(test_df[i].mode()[0])

# 3、异常值处理（数值型）
for i in num_cols[1:]:
    test_df_num_col = test_df_num_cols[i]
    xmean = test_df_num_col.mean()
    xstd = test_df_num_col.std()
    test_df_num_col[test_df_num_col > xmean + 2 * xstd] = test_df_num_col[test_df_num_col < xmean + 2 * xstd].max()
    test_df_num_col[test_df_num_col < xmean - 2 * xstd] = test_df_num_col[test_df_num_col > xmean - 2 * xstd].min()

# 4、编码(类别变量)+归一化(数值变量)
test_df_num_cols = mm.transform(test_df_num_cols)
test_df_num_cols = pd.DataFrame(data=test_df_num_cols, columns=num_cols[1:])
test_df_cat_cols = pd.DataFrame(ohe.transform(test_df[cat_cols[:-1]].values), columns=ohe.get_feature_names()).reset_index(drop=True)
test_df_label = pd.DataFrame(le.transform(test_df['classification'].values), columns=['classification']).reset_index(drop=True)

# # 5、重构
test_df = pd.concat([test_df_num_cols.reset_index(drop=True), test_df_cat_cols, test_df_label], axis=1)
X_test = test_df.iloc[:,:-1]
y_test = test_df['classification']


# ============================================模型推理与评价============================================
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred) # 准确率acc
cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
cr = classification_report(y_test, y_pred) # 分类报告
print('acc:', acc, sep='\n')
print('cm:', cm, sep='\n')
print('cr:', cr, sep='\n')


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

