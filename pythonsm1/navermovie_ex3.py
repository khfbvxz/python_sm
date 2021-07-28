from bs4 import BeautifulSoup
import requests
# import lxml
import re
# url = 'https://sports.news.naver.com/wfootball/news/index?isphoto=N'
#
# res = requests.get(url)
#
# soup = BeautifulSoup(res.content,"lxml")  #lxml

# for parents in soup.ul.parents:
#     print(parents)

# print(soup.find_all("div"))

# soup.find_all(re.compile("[ou]l"))
# soup.find_all(['div','a'])

from bs4 import BeautifulSoup
from urllib.request import urlopen
#
# with urlopen('https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8C%80%EB%AC%B8') as response:
#     soup = BeautifulSoup(response,'html.parser')
#     for anchor in soup.find_all('a'):
#         print(anchor.get('href','/'))


keyword= input('검색할 단어를 입력하세요')
req = requests.get(f"url"/{keyword})#위키디피아
html = req.text
soup = BeautifulSoup(html,"lxml")

definition = soup.select_one("copy select").text   #f12  그부분 하이라이트 찾고  마우스 우클릭 copy select
# print(definition)





''' 
 
 #_newsList > ul > li:nth-child(1) > div  
'''








