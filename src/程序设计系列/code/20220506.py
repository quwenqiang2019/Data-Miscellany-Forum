import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def merge(file1,file2):
    data1=pd.read_excel(file1)
    data1=pd.DataFrame(data1)
    print(data1[['病例系统ID号','性别','民族']])

    data2=pd.read_excel(file2)
    data2=pd.DataFrame(data2)
    print(data2[['病例系统ID号','NAME','市']])

    result=pd.merge(data1[['病例系统ID号','性别','民族']],data2[['病例系统ID号','NAME','市']],on='病例系统ID号',how='outer')

    print(result)
    result.to_excel("F:\数据杂坛\\result\合并.xls")

if __name__=="__main__":
    merge("F:\数据杂坛\data\江苏省SLE数据库（整理）.xlsx","F:\数据杂坛\data\患者按地区性别住址_2231.xls")
