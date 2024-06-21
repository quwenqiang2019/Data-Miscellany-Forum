import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

print(df)


# 提取目标变量和特征变量
target = 'class label'
features = df.columns.drop(target)
print(df['class label'].value_counts()) # 顺便查看一下样本是否平衡
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=True, random_state=0)


# 将划分后的数据重构为dataframe
train_X = pd.DataFrame(X_train, columns=features)
train_y = pd.DataFrame(y_train, columns=[target])
train_df = pd.concat([train_X, train_y],axis = 1).reset_index(drop=True)

test_X = pd.DataFrame(X_test, columns=features)
test_y = pd.DataFrame(y_test, columns=[target])
test_df = pd.concat([test_X, test_y],axis = 1).reset_index(drop=True)

print(train_df)
print(test_df)

# 编码：文本型数据转为数值型数据
le = LabelEncoder()  # 对于标签采用LabelEncoder
ohe = OneHotEncoder(sparse=False)  # 对于特征采用OneHotEncoder
train_df = pd.DataFrame(ohe.fit_transform(train_df[['color','size']].values), columns=ohe.get_feature_names()).join(train_df[['prize','class label']])
train_df['class label'] = le.fit_transform(train_df['class label'])
X_train = train_df.iloc[:,:-1]
y_train = train_df['class label']


# 模型的构建与训练
model = RandomForestClassifier()
model.fit(X_train, y_train)

# # 模型推理与评价
test_df = pd.DataFrame(ohe.transform(test_df[['color','size']].values), columns=ohe.get_feature_names()).join(test_df[['prize','class label']])
test_df['class label'] = le.transform(test_df['class label'])
X_test = test_df.iloc[:,:-1]
y_test = test_df[['class label']]

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) # 准确率acc
print(acc)

