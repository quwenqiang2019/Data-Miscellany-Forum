import pandas as pd
import warnings
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
warnings.filterwarnings("ignore")


class ML():
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = BASE_DIR

    def read_data(self, file):
        #=======================读取原始数据============================
        df = pd.read_csv(os.path.join(self.base_dir, 'datasets', file))
        df=pd.DataFrame(df)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        df.drop("id",axis=1,inplace=True)   # 删除非特征/标签列

        return df

    def dirty_data_processing(self, df):
        #=======================脏数据处理================================
        for col in df.columns:
            df[col].value_counts()  # 修改前
        df["classification"] = df["classification"].apply(lambda x: x if x == "notckd" else "ckd")
        df["pcv"] = pd.to_numeric(df["pcv"], errors="coerce")#将字符串列转换为数值类型，并使用errors='coerce'参数将无法转换的值设置为缺失值
        df["rc"] = pd.to_numeric(df["rc"], errors="coerce")
        df["dm"] = df["dm"].str.strip()  # 去除空格
        df["cad"] = df["cad"].str.strip() 

        return df

    def random_value_imputate(self, col):
        """
        函数：随机填充方法（缺失值较多的字段）
        """
        # 1、确定填充的数量；在取出缺失值随机选择缺失值数量的样本
        random_sample = df[col].dropna().sample(df[col].isna().sum())
        # 2、索引号就是原缺失值记录的索引号
        random_sample.index = df[df[col].isnull()].index
        # 3、通过loc函数定位填充
        df.loc[df[col].isnull(), col] = random_sample

    def mode_impute(self, col):
        """
        函数：众数填充缺失值
        """
        # 1、确定众数
        mode = df[col].mode()[0]
        # 2、fillna函数填充众数
        df[col] = df[col].fillna(mode)

    def missing_value_processing(self, df):
        #======================特征：缺失值处理================
        # print(df.isnull().sum().sort_values(ascending = False))
        # df = df.dropna()
        # 数值型变量名
        num_cols = [col for col in df.columns if df[col].dtype != "object"]
        # 类别型变量名
        cat_cols = [col for col in df.columns if df[col].dtype == "object"]
        # 随机填充
        for col in num_cols:
            self.random_value_imputate(col)
        # 随机填充
        self.random_value_imputate('rbc')
        self.random_value_imputate('pc')
        # 其他字段是众数填充
        for col in cat_cols:
            self.mode_impute(col)

        return df


    def abnorml_value_processing(self, df):
        #=====================特征：异常值处理========================
        pass


    def feature_encoding(self, df):
        #====================特征：编码================================
        # print(df.dtypes)
        # 数值型变量名
        num_cols = [col for col in df.columns if df[col].dtype != "object"]
        # 类别型变量名
        cat_cols = [col for col in df.columns if df[col].dtype == "object"]
        zs_scaler = StandardScaler()
        df[num_cols] = zs_scaler.fit_transform(df[num_cols])

        led = LabelEncoder()
        for col in cat_cols:
            df[col] = led.fit_transform(df[col])

        return df

    def train_test_split(self, df):
        #===================数据集划分===============================
        df = pd.DataFrame(df)
        print(df["classification"].value_counts())
        X = df.drop("classification",axis=1)
        y = df["classification"]
        df = shuffle(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        return X_train, X_test, y_train, y_test

    def feature_importance_analysis(self):
        #===============特征工程：特征重要性分析==================
        # 常见的特征重要性分析方法
        pass

    #==============建模1:KNN=================
    def knn_model(self, X_train, X_test, y_train, y_test):
        print(X_test)
        model = KNeighborsClassifier(n_neighbors=10)
        # 模型训练
        model.fit(X_train, y_train)
        # 模型预测
        y_pred = model.predict(X_test)
        # 准确率acc
        acc = accuracy_score(y_test, y_pred)
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        # 分类报告
        cr = classification_report(y_test, y_pred)

        print(f"Test Accuracy of {model} : {acc}")
        print(f"Confusion Matrix of {model}: \n{cm}")
        print(f"Classification Report of {model} : \n {cr}")


if __name__ == '__main__':
    df = ML().read_data('kidney_disease.csv')
    df = ML().dirty_data_processing(df)
    df = ML().missing_value_processing(df)
    df = ML().feature_encoding(df)
    X_train, X_test, y_train, y_test = ML().train_test_split(df)
    ML().knn_model(X_train, X_test, y_train, y_test)
