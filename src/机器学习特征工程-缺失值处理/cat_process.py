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
print('字符型数据缺失情况：', df[cat_cols].isnull().sum(), sep='\n')


# 法1：众数填充
# for i in cat_cols:
#     df[i] = df[i].fillna(df[i].mode()[0])

# 法2：自定义填充
# 略

# 法3：前后数据填充
df[cat_cols] = df[cat_cols].fillna(method='pad')
# df[num_cols] = df[num_cols].fillna(method='bfill')

# 法4：机器学学习预测填充
# 略

print('处理后字符型数据缺失情况：', df[cat_cols].isnull().sum(), sep='\n')

