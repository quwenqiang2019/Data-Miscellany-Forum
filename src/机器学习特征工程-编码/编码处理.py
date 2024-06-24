import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.DataFrame(data = [['green', 'M', 10.1, 'class1'],
                          ['red', 'L', 13.5, 'class2'],
                          ['blue', 'XL', 15.3, 'class1'],
                          ['yellow', 'L', 13.5, 'class2'],
                          ['red', 'L', 13.5, 'class2'],
                          ['blue', 'S', 10.5, 'class2'],
                          ['yellow', 'L', 13.5, 'class1'],
                          ['red', 'L', 11.5, 'class2'],
                          ['blue', 'M', 12.5, 'class1'],
                          ['red', 'XL', 14.5, 'class1'],],
                  columns=['color', 'size', 'prize', 'class label'])

print(df.head())


# 提取目标变量和特征变量
target = 'class label'
features = df.columns.drop(target)
print(df['class label'].value_counts()) # 顺便查看一下样本是否平衡

# 数据划分
train_df = df.sample(frac=0.8, random_state=2)
test_df = df.drop(train_df.index)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(train_df)
print(test_df)


# 训练集预处理（编码和归一化）
le = LabelEncoder()  # 对于标签采用LabelEncoder
ohe = OneHotEncoder(sparse=False)  # 对于类别特征采用OneHotEncoder
mm = MinMaxScaler() # 对于数值特征采用MinMaxScaler
train_df_X = pd.DataFrame(ohe.fit_transform(train_df[['color','size']].values), columns=ohe.get_feature_names())
train_df['class label'] = le.fit_transform(train_df['class label'])
train_df[['prize']] = mm.fit_transform(train_df[['prize']])
train_df = pd.concat([train_df_X, train_df['prize'], train_df['class label']], axis=1)
print(train_df)
X_train = train_df.iloc[:,:-1]
y_train = train_df['class label']

# 模型的构建与训练
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 测试集做和训练集相同的预处理
test_df_X = pd.DataFrame(ohe.transform(test_df[['color','size']].values), columns=ohe.get_feature_names())
test_df['class label'] = le.transform(test_df['class label'])
test_df[['prize']] = mm.transform(test_df[['prize']])
test_df = pd.concat([test_df_X, test_df['prize'], test_df['class label']], axis=1)
print(test_df)
X_test = test_df.iloc[:,:-1]
y_test = test_df['class label']


# # 模型推理与评价
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) # 准确率acc
print(acc)