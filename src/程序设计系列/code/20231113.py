import requests
from bs4 import BeautifulSoup

def search_baidu(keyword, page):
    url = f"https://www.baidu.com/s?wd={keyword}&pn={page}&rn=10"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        return None


def parse_search_results(html):
    soup = BeautifulSoup(html, "html.parser")
    news_results = soup.find_all("h3", class_="t")
    news_list = []
    for result in news_results:
        title = result.a.text
        link = result.a["href"]
        news_list.append({"title": title, "link": link})
    return news_list


def crawl_latest_news(keyword, num_news):
    news_list = []
    num_pages = num_news // 10 + 1  # 每页10条新闻，计算需要请求的页面数
    for page in range(num_pages):
        html = search_baidu(keyword, page * 10)
        if html:
            page_news = parse_search_results(html)
            news_list.extend(page_news)
        else:
            print(f"无法获取第 {page+1} 页的搜索结果")
    return news_list[:num_news]


if __name__ =="__main__":
    keyword = "开源之夏"
    num_news = 500
    news_list = crawl_latest_news(keyword, num_news)
    if news_list:
        for news in news_list:
            print(news["title"])
            print(news["link"])
            print()
    else:
        print("无法获取搜索结果")