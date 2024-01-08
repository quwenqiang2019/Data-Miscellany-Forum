import asyncio
import aiohttp
import openpyxl
from bs4 import BeautifulSoup

async def fetch_movie_info(session, url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Cookie": "bid=m9sDMeuTWp4; ap_v=0,6.0; _pk_id.100001.4cf6=d6615bd2530852c6.1700447648.; _pk_ses.100001.4cf6=1; __utma=30149280.633232779.1700447649.1700447649.1700447649.1; __utmb=30149280.0.10.1700447649; __utmc=30149280; __utmz=30149280.1700447649.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utma=223695111.1435231277.1700447649.1700447649.1700447649.1; __utmb=223695111.0.10.1700447649; __utmc=223695111; __utmz=223695111.1700447649.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); _cc_id=748927837a892b664c1f1ab42fbe510a; panoramaId_expiry=1700534054317; panoramaId=18a92c0e9b136f927d0f0871ae33a9fb927a9d987bb8aa39557c58077684bc2c; panoramaIdType=panoDevice; _pbjs_userid_consent_data=3524755945110770; __gads=ID=7617c807b66fd695:T=1700447653:RT=1700448285:S=ALNI_MY0jxMNVX0GooLXe8dtdh74vfdLvQ; __gpi=UID=00000cdbaaf33934:T=1700447653:RT=1700448285:S=ALNI_MYekZkuVr46VHfZjhuhdX2kpLxOkw; cto_bundle=xIP-n181MjZFSVBGdlMlMkJEY3hvY3dycER1QjhISjdGU2dzOWxWZUFSMmNZd25VQ1Y0REdtaXZPdTh2aEJGUCUyQlo3WjVETzVNc2VUSFR3dHFXQVRRZU1ZejdOMXk5RDM4VjV1WkJsRWVXd1dQdjRvRE1JQjhEVkJQUVEyV0M1dlgzVkFBclZDTnJWM1g3MWZERDltRFR1UDZZNXp3JTNEJTNE; cto_bidid=vr7nBV8lMkZGJTJCOGVQWjhWREJUelpJYm1UdFBWaWd5bk9WT1JCdyUyRjlpN1duSWFZd3JPR2dkdmh1Q2tNa3NJa25rQTExSFlPM1p2YzdpT1U2cDE5UUowU3p1VHk3YkhVWWw4aFBmUExiZmtZdWtPS3U4byUzRA; cto_dna_bundle=14GGU181MjZFSVBGdlMlMkJEY3hvY3dycER1QiUyQmxhTVFwSEdNWHZ6OE5MZ2olMkJQbjlyODR2SWtIJTJCUGZmYm40Z3p5b1AxbSUyRkJKVDBVUVlXbGE1ZWRQeVUlMkJmeTR5dyUzRCUzRA",
    }
    async with session.get(url, headers=headers) as response:
        data = await response.text()
        return data

async def parse_movie_info(html):
    # 在这里编写提取信息的代码，根据豆瓣电影页面的结构
    # 提取导演、主演等信息，并返回一个字典
    soup = BeautifulSoup(html, 'html.parser')
    movie_elements = soup.find_all('div', class_='item')
    movie_info_list = []

    for movie_element in movie_elements:
        # title_element = movie_element.find('div', class_='hd')
        # title = title_element.find('span', class_='title').text
        title = movie_element.find('span', class_='title').text
        # print(title)
        detail_link = movie_element.find('a')['href']
        image_link = movie_element.find('img')['src']
        rating = float(movie_element.find('span', class_='rating_num').text)

        movie_info = {
            'title': title,
            'detail_link': detail_link,
            'image_link': image_link,
            'rating': rating
        }

        movie_info_list.append(movie_info)

    return movie_info_list

async def write_to_excel(movie_info_lists):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['title', 'detail_link', 'image_link', 'rating'])
    for movie_info in movie_info_lists:
        ws.append([movie_info['title'], movie_info['detail_link'], movie_info['image_link'], movie_info['rating']])
    wb.save('douban_top250.xlsx')

async def main():
    base_url = 'https://movie.douban.com/top250?start={}'
    movie_info_lists = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_movie_info(session, base_url.format(i)) for i in range(0, 250, 25)]
        pages = await asyncio.gather(*tasks)

        for page in pages:
            movie_info_list = await parse_movie_info(page)
            movie_info_lists.extend(movie_info_list)

    await write_to_excel(movie_info_lists)

if __name__ == '__main__':
    asyncio.run(main())

