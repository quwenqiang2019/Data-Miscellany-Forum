import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("G:\数据杂坛\datasets\kidney_disease.csv")
df=pd.DataFrame(df)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
df.drop("id",axis=1,inplace=True)
print(df.head())

df["classification"] = df["classification"].apply(lambda x: x if x == "notckd" else "ckd")
# 数值型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"]
# 分类型变量名
cat_cols = [col for col in df.columns if df[col].dtype == "object"]
print(df.isnull().sum().sort_values(ascending = False))
# ======================缺失值处理============================
def random_value_imputate(col):
    """
    函数：随机填充方法（缺失值较多的字段）
    """
    # 1、确定填充的数量；在取出缺失值随机选择缺失值数量的样本
    random_sample = df[col].dropna().sample(df[col].isna().sum())
    # 2、索引号就是原缺失值记录的索引号
    random_sample.index = df[df[col].isnull()].index
    # 3、通过loc函数定位填充
    df.loc[df[col].isnull(), col] = random_sample

def mode_impute(col):
    """
    函数：众数填充缺失值
    """
    # 1、确定众数
    mode = df[col].mode()[0]
    # 2、fillna函数填充众数
    df[col] = df[col].fillna(mode)

for col in num_cols:
    random_value_imputate(col)

for col in cat_cols:
    if col in ['rbc','pc']:
        # 随机填充
        random_value_imputate('rbc')
        random_value_imputate('pc')
    else:
        mode_impute(col)

print(df.isnull().sum().sort_values(ascending = False))
print(df.head())