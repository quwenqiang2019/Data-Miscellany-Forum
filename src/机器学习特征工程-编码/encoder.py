import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 读取数据并做大致分析
df = pd.read_csv('data.csv')
print('数据前5行：', df.head(), sep='\n')
print('数据形状：', df.shape, sep='\n')
print("数据列：", df.columns, sep='\n')
print("数据统计描述：", df.describe(), sep='\n')   # 只会统计数值型变量（int、float）
print("数据类型：", df.dtypes, sep='\n')
print("数据缺失值统计：", df.isnull().sum(), sep='\n')
print("数据重复值统计", df.duplicated().sum(), sep='\n')

cat_cols = [col for col in df.columns if df[col].dtype == "object"] # 类别型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"] # 数值型变量名
print(cat_cols)
print(num_cols)

# 查看一下类别型变量每个类别的数量,主要是看一下有没有脏数据和错误数据，有些本应该是数值型的变量被划分为类别型表明该列存在一些脏数据，有的类别型变量无端多出奇怪的类也表明存在脏数据
for i in cat_cols:
    print(df[i].value_counts())

# 划分数据集
train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)
print(train_df)
print(test_df)




# 对训练集进行特征工程
# 1、数据清洗
# 2、缺失值处理
# 3、异常值处理
# 2、数值型特征变量归一化
# 3、类别型特征变量编码（特征用one_hot，标签用label编码）

le = LabelEncoder()  # 对于标签采用LabelEncoder
ohe = OneHotEncoder(sparse=False)  # 对于特征采用OneHotEncoder

train_df['class label'] = le.fit_transform(train_df['class label'])

train_df.drop("id",axis=1,inplace=True)   # 删除第一列id





train_df[train_df > train_df.mean() + 2 * train_df.std()] = train_df[train_df < train_df.mean() + 2 * train_df.std()].max()
train_df[train_df < train_df.mean() - 2 * train_df.std()] = train_df[train_df > train_df.mean() - 2 * train_df.std()].min()


# train_df = pd.DataFrame(ohe.fit_transform(train_df[['color','size']].values), columns=ohe.get_feature_names()).join(train_df[['prize','class label']])
# train_df['class label'] = le.fit_transform(train_df['class label'])
# X_train = train_df.iloc[:,:-1]
# y_train = train_df['class label']
#
#
# # 模型的构建与训练
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
#
# # # 模型推理与评价
# test_df = pd.DataFrame(ohe.transform(test_df[['color','size']].values), columns=ohe.get_feature_names()).join(test_df[['prize','class label']])
# test_df['class label'] = le.transform(test_df['class label'])
# X_test = test_df.iloc[:,:-1]
# y_test = test_df[['class label']]
#
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred) # 准确率acc
# print(acc)
#
