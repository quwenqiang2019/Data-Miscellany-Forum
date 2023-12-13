from xlutils.copy import copy
import xlrd
import datetime
from xlrd import xldate_as_datetime

def time_subtraction(file):
    workbook = xlrd.open_workbook(file)
    worksheet=workbook.sheet_by_index(0)

    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(0)

    nrow=worksheet.nrows

    for i in range(0,nrow):
        if i==0:
            new_worksheet.write(i, 21, '病程')
        else:
            row= worksheet.row_values(i)
            if row[13]!='' and row[20]!='':
                d1=xldate_as_datetime(row[13],0).strftime('%Y%m')
                d2=xldate_as_datetime(row[20],0).strftime('%Y%m')
                print(d1,d2)
                v_year_end = datetime.datetime.strptime(d1, '%Y%m').year
                v_month_end = datetime.datetime.strptime(d1, '%Y%m').month
                v_year_start = datetime.datetime.strptime(d2, '%Y%m').year
                v_month_start = datetime.datetime.strptime(d2, '%Y%m').month
                interval = (v_year_end - v_year_start) * 12 + \
                           (v_month_end - v_month_start)
                print('时间差（月数）：%s'%interval)
                new_worksheet.write(i, 21, interval)
            else:
                new_worksheet.write(i, 21, 'null')
        new_workbook.save("F:\数据杂坛\\result\时间差.xls")

if __name__=="__main__":
    time_subtraction("F:\数据杂坛\data\患者按地区研究信息_2231 .xls")
