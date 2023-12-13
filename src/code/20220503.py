import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def Box_Cox(file,sheet_name):
    df1 = pd.read_excel(file,sheet_name)
    print(df1["患者密度（人/10万人）"])
    sns.distplot(df1["患者密度（人/10万人）"],color = "#D86457")
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    stats.boxcox_normplot(df1["患者密度（人/10万人）"], -20, 20,plot = ax)
    plt.axvline(x = stats.boxcox_normmax(df1["患者密度（人/10万人）"]),color = "#D86457")
    plt.show()

    print(stats.boxcox_normmax(df1["患者密度（人/10万人）"]))
    x = stats.boxcox(df1["患者密度（人/10万人）"],stats.boxcox_normmax(df1["患者密度（人/10万人）"]))
    sns.distplot(x,color = "#D86457")
    plt.show()

    df=pd.DataFrame(x,columns=['转换'])
    print(df)

if __name__=='__main__':
    Box_Cox("F:\医学大数据课题\论文终稿修改\实验\差异性分析.xlsx",sheet_name='人口密度分组')
