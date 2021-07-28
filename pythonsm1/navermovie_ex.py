from bs4 import BeautifulSoup

html="""
<html><head><title>The Dormouse's story</title></head>
<body><p class="title"><b>The Dormouse's story</b></p></body>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="https://example.com/elsie" class="sister" id="link1">Elise</a>,
<a href="https://example.com/lacie" class="sister" id="link2">lacie</a>and
<a href="https://example.com/tillie" class="sister" id="link3">tillie</a>:
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""
soup = BeautifulSoup(html,'lxml')  # 책 lmxl인데 파서랑 왜 다른지?
# print("soup.find()의 결과 : ", soup.find('a',attrs={'class' : 'sister'}))
'''
#객체.태그 이름
#.태그 이름으로 하위 태그로의 접근이 가능하다.
print("soup.body.p의 결과 : ", soup.body.p)

# <p class="title"><b>The Dormouse's story</b></p>
'''

'''
객체.태그 ['속성이름']
객체의 태그 속성은 파이썬의 딕셔너리처럼 태그['속성 이름']으로 접근으로 가능하다.
print("soup.a['href']의 결과 ", soup.a['href'])

# soup.a['href']의 결과  https://example.com/elsie
'''
'''
객체.name
#name 변수
print("soup.title.name 의 결과 :  ", soup.title.name)
#title
'''

'''
# 객체.string
# string 변수 (참고)  NavigableString: 문자열은 태그 안의 텍스트에 상응한다. BeautifulSoup은
# 이런 텍스트를 포함하는 NavigableString 클래스를 사용한다.
print("soup.title.string의 결과 : ", soup.title.string)
# soup.title.string의 결과 :  The Dormouse's story
'''

'''
# 객체.contents
## 태그의 자식들을 리스트로 반환
print("soup.contents의 결과 : " , soup.contents)
# soup.contents의 결과 :  [<html><head><title>The Dormouse's story</title></head>
# <body><p class="title"><b>The Dormouse's story</b></p></body>
# <p class="story">Once upon a time there were three little sisters; and their names were
# <a class="sister" href="https://example.com/elsie" id="link1">Elise</a>,
# <a class="sister" href="https://example.com/lacie" id="link2">lacie</a>and
# <a class="sister" href="https://example.com/tillie" id="link3">tillie</a>:
# and they lived at the bottom of a well.</p>
# <p class="story">...</p>
# </html>]
'''
'''
# find() : 태그 하나만 가져옴

# find(name, sttrs, recursive, string, **kwargs)
# 
# [옵션]
# name - 태그
# attrs - 속성(딕셔너리로)
# recursive - 모든 자식 or 자식
# string - 태그 안에 텍스트
# keyword - 속성(키워드) 
# % (주의) class 는 파이썬 예약어이므로 , class_ 를 사용한다.

print("soup.find()의 결과 : ", soup.find('a',attrs={'class' : 'sister'}))

#soup.find()의 결과 :  <a class="sister" href="https://example.com/elsie" id="link1">Elise</a>
'''
# print("soup.find()의 결과 : ", soup.find('a',attrs={'class' : 'sister'}))
'''
# find_all() : 해당 태그가 여러 개 있을경우
find_all(name, attrs, recursive, string, limit, **kwargs)
[옵션]Limit 몇개까지 찾을 것인가? find_all() 로 검색했을 떄, 수천 수만개가 된다면 시간이 오래걸릴 것이다. 
이떄까지 몇개 까지만 찾을 수 있도록 제한을 둘수 있는 인자다.


print("soup.find_all()의 결과 : ", soup.find_all('a', limit=2))
#soup.find_all()의 결과 :  [<a class="sister" href="https://example.com/elsie" id="link1">Elise</a>, <a class="sister" href="https://example.com/lacie" id="link2">lacie</a>]
'''

