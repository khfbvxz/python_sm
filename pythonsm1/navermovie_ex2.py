import requests
from bs4 import BeautifulSoup

# '트롤' 의 네이버 영화 리뷰 링크
url = "https://movie.naver.com/movie/bi/mi/review.nhn?code=191633"
# url = "https://sports.news.naver.com/wfootball/news/index?isphoto=N"
#html 소스가져오기
res = requests.get(url)

# html 파싱
soup = BeautifulSoup(res.text,'lxml')  # html.parser

# #리뷰리스트
''''''
# ul = soup.find('ul',class_="rvw_list_area")  # 코드설명 class  class_
# lis = ul.find_all('li')
# '''
# loaskfkasfh li
# akslfhlakf
# asfkjaklsf
# '''
# #리뷰 제목 출력
# count=0
# for li in lis:
#     count += 1
#     print(f"[{count}th] ", li.a.string)

# # print(lis)

#리뷰리스트
# div = soup.find('div', class_="news_list")
# lis = div.find_all('li')
# # lis = div.find_all('a')
# # #리뷰 제목 출력
# count=0
# for li in lis:
#     count += 1
#     print(f"[{count}th] ", li.a.span.string)
# #
# print(li)

# for link in soup.find_all('a'):
#     print(link.text.strip(), link.get('href'))