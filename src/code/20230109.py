import pandas as pd

def deal_1():
    # 列表
    company_name_list = ['腾讯', '阿里巴巴', '字节跳动', '腾讯']
    # list转dataframe
    df = pd.DataFrame(company_name_list, columns=['company_name'])
    # 保存到本地excel
    df.to_excel("D:\数据杂坛\素材\\0109\company_name_li_1.xlsx", index=False)

def deal_2():
    # 二维list
    company_name_list = [['腾讯', '北京'], ['阿里巴巴', '杭州'], ['字节跳动', '北京']]
    # list转dataframe
    df = pd.DataFrame(company_name_list, columns=['company_name', 'local'])
    # 保存到本地excel
    df.to_excel("D:\数据杂坛\素材\\0109\company_name_li_2.xlsx", index=False)

if __name__ == '__main__':
    deal_1()
    deal_2()
