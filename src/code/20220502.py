import json
from urllib.request import urlopen,quote
import xlrd

def readXLS(XLS_FILE,sheet0):
    rb= xlrd.open_workbook(XLS_FILE)
    rs= rb.sheets()[sheet0]
    return rs

def getlnglat(adress):
    url = 'http://api.map.baidu.com/geocoding/v3/?address='
    output = 'json'
    ak = 'fdi11GHN3GYVQdzVnUPuLSScYBVxYDFK'
    add = quote(adress)#使用quote进行编码 为了防止中文乱码
    # add=adress
    url2 = url + add + '&output=' + output + '&ak=' + ak
    req = urlopen(url2)
    res = req.read().decode()
    temp = json.loads(res)
    return temp

def getlatlon(sd_rs):
    nrows_sd_rs=sd_rs.nrows
    for i in range(4,nrows_sd_rs):
    # for i in range(4, 7):
        row=sd_rs.row_values(i)
        print(i,i/nrows_sd_rs)
        b = (row[11]+row[12]+row[9]).replace('#','号') # 第三列的地址
        print(b)
        try:
            lng = getlnglat(b)['result']['location']['lng']  # 获取经度并写入
            lat = getlnglat(b)['result']['location']['lat']  #获取纬度并写入
        except KeyError as e:
            lng=''
            lat=''
            f_err=open('f_err.txt','a')
            f_err.write(str(i)+'\t')
            f_err.close()
            print(e)
        print(lng,lat)
        f_latlon = open('f_latlon.txt', 'a')
        f_latlon.write(row[0]+'\t'+b+'\t'+str(lng)+'\t'+str(lat)+'\n')
        f_latlon.close()




if __name__=='__main__':
    # sle_xls_file = 'F:\医学大数据课题\江苏省SLE数据库（两次随访合并）.xlsx'
    sle_xls_file = "F:\医学大数据课题\数据副本\江苏省SLE数据库（两次随访合并） - 副本.xlsx"
    sle_data_rs = readXLS(sle_xls_file, 1)
    getlatlon(sle_data_rs)
