import xlrd  # 引入模块

def count(file):
    # 打开文件，获取excel文件的workbook（工作簿）对象
    workbook = xlrd.open_workbook(file)  # 文件路径
    worksheet=workbook.sheet_by_index(0)

    nrow=worksheet.nrows
    print(nrow)

    inter_n_save=0
    inter_n_death=0
    miss_inter_n=0

    for i in range(1,nrow):
        row= worksheet.row_values(i)
        if row[4]=='死亡':
            inter_n_death=inter_n_death+1
        if row[4]=='存活':
            inter_n_save=inter_n_save+1
        elif row[4]=='失访':
            miss_inter_n=miss_inter_n+1

    print(inter_n_death)
    print(inter_n_save)
    print(miss_inter_n)


if __name__=="__main__":
    count("F:\医学大数据课题\饮水源诱发SLE\患者随访分析.xls")
