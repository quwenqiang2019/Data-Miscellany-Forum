from xlutils.copy import copy
import xlrd

def Find_keywords(file):
    workbook = xlrd.open_workbook(file)
    worksheet=workbook.sheet_by_index(0)

    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(0)

    nrow=worksheet.nrows

    keylist=['面部红斑','关节','发热','咳嗽','浮肿','皮疹']
    for i in range(0,nrow):
        if i==0:
            for j in range(1,len(keylist)+1):
                new_worksheet.write(i,j,keylist[j-1])
        else:
            row= worksheet.row_values(i)
            print(row[0])
            for a in keylist:
                if row[0].count(a):
                    j = 1 + keylist.index(a)
                    new_worksheet.write(i,j,1)

        new_workbook.save("F:\数据杂坛\\0509\合合.xls")

if __name__=="__main__":
    Find_keywords("F:\数据杂坛\\0509\data.xlsx")
