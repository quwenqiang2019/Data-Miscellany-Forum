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
df.drop("id",axis=1,inplace=True)
print('数据：', df, sep='\n')
print('数据缺失值情况：', df.isnull().sum(), sep='\n')


cat_cols = [col for col in df.columns if df[col].dtype == "object"] # 类别型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"] # 数值型变量名
print(cat_cols)
print(num_cols)

print('数值型数据缺失情况：', df[num_cols].isnull().sum(), sep='\n')


# 法1：均值填充
# df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# 法2：中位数填充
# df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 法3：众数填充
df[num_cols] = df[num_cols].fillna(df[num_cols].mode().iloc[0])

# 法4：前后数据填充
# df[num_cols] = df[num_cols].fillna(method='pad')
# df[num_cols] = df[num_cols].fillna(method='bfill')

# 法5：自定义填充
# 略

# 法6：interpolate()插值方法填充
# df[num_cols] = df[num_cols].interpolate()

# 法7：机器学学习预测填充
# 略

print('处理后数值型数据缺失情况：', df[num_cols].isnull().sum(), sep='\n')
