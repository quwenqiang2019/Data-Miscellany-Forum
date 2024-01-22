# coding:utf-8
import requests
from lxml import etree
import json

def spider():
    header={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    url=requests.get("http://101.203.175.254:8138/backstage/api/PlantingArea/plantingarea/queryWorkplace?userCode=admin",headers=header)
    #为了防止中文乱码，编码使用原网页编码
    url.raise_for_status()
    url.encoding = url.apparent_encoding
    text=url.text

    user_info= '{"data" }'
    user_dict = json.loads(text)
    #print(user_dict['data'])

    #1号大棚
    #print(user_dict['data'][0]['areapoint'][1])
    temperature=user_dict['data'][0]['areapoint'][0]['value']#温度
    humidity=user_dict['data'][0]['areapoint'][1]['value']#湿度
    light_intensity=user_dict['data'][0]['areapoint'][2]['value']#光照强度
    soil_temperature=user_dict['data'][0]['areapoint'][3]['value']#土壤温度
    soil_humidity=user_dict['data'][0]['areapoint'][4]['value']#土壤湿度
    co2=user_dict['data'][0]['areapoint'][5]['value']#二氧化碳
    rain=1

    input_list=[0,temperature,humidity,light_intensity,soil_temperature,soil_humidity,co2,rain]
    input_list=[float(_) for _ in input_list]
    print(input_list)



# object=etree.HTML(url.text)
# print(object)
# #正则匹配搜索出来答案的所有网址
# #获取词条
# #head =object.xpath('/html/head//meta[@name="description"]/@content')
# #详细内容
# para=object.xpath('/html/body//div[@class="para"]/text()')
# #print(head[0])
# result=''
# for i in para:
#     result+=i
# print(result)



