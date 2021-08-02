
import numpy as np
import pandas as pd
from pprint import *
import matplotlib.pyplot as plt
import matplotlib
import openpyxl

''' 
pandas는 강력한 데이터 구조를 사용하여 고성능 데이터 조작 및
데이터 분석에 사용되는 오픈소스파이선라이브러리이다.
'''
'''
판다스로 CSV파일이나 TSV파일 엑셀파일등을 열 수 있다.
mean() 모든 열의 평균 # axis 사용하면됨
corr()로 데이터 프레임의 열 사이의 상관관계를 계산할 수 있다.
조건을 사용하여 데이터를 필터링 할 수 있다.
sort_values()로 데이터를 정렬할 수 이싿.
groupby()를 이용하여 기준에 따라 몇개의 그룹으로 데이터를 분할 할 수있다.
데이터의 누락 값을 확인할 수 있다.

데이터 프레임에서는 행이나 열에 붙인 레이블을 중요시 여긴다.
index 객체는 행들의 레이블(label)이고 columns 객체는 열들의
레이블이 저장된 객체이다.
'''

'''타이타닉 데이터셋
⚫ PassengerId: 승객의 ID이다. 
⚫ Survived: 생존 여부
⚫ Pclass: 탑승 등급을 나타낸다. 클래스 1, 클래스 2, 클래스 3의 3가지 클
래스가 있다.
⚫ Name: 승객의 이름.
⚫ Sex: 승객의 성별.
⚫ Age: 승객의 나이.
⚫ SibSp: 승객에게 형제 자매와 배우자가 있음을 나타낸다. 
⚫ Parch: 승객이 혼자인지 또는 가족이 있는지 여부.
⚫ Ticket: 승객의 티켓 번호.
⚫ Fare: 운임.
⚫ Cabin : 승객의 선실.
⚫ Embarked: 탑승한 지역.
'''

titanic = pd.read_csv('titanic.csv')
# pprint(titanic)
# pprint(titanic["Age"])
'''최고령자'''
# pprint(titanic["Age"].max())

'''타이타닉 승객 데이터에 대한 기본통계를 알고 싶다면?'''
# describe() 메소드는 숫자 데이터에 대한 간략한 개요를 제공한다.
# 문자열 데이터는 처리하지 않는다.
# pprint(titanic.describe())


'''데이터 시리즈 생성하기 '''
#시리즈는 이름이 붙여진 1차원적인 배열이나 마찬가지
# 기본적인 방법은 파이썬의 리스트에서 생성하는 것이다.
# data = ['Kim',"Park","Lee","Choi"]
# ser = pd.Series(data)
# pprint(ser)

''' 데이터 프레임 생성하기 '''
data = {'Name':['Kim','Park','Lee','Choi'],'Age':[20,23,21,26]}
# df = pd.DataFrame(data)
# pprint(data)
# pprint(df)

# 데이터 프레임에 index를 붙이려면 다음과 같이 index 매개 변수를 사용할 수 있다.
# df = pd.DataFrame(data,index=["학번1","학번2","학번3","학번4"])
# pprint(df)

'''csv파일을 읽어서 데이터 '''
# pprint(titanic.dtypes)

'''인덱스 변경   '''
# 파일에서 읽을 떄 index를 변경할 수 있다. 예를 들어서 첫번째 열을 index객체로 사용할 수도 있다.
# titanic = pd.read_csv('titanic.csv',index_col=0)
# pprint(titanic)

'''데이터 프레임의 몇개의 행을 보려면?'''
# pprint(titanic.head(8))

'''데이터 프레임을 엑셀 파일로 저장하려면 ?'''
# titanic.to_excel('titanic.xlsx',sheet_name='passengers',index=False)

# 이렇게 저장한 엑셀 파일을 다시 읽으려면 다음과 같이 한다.
# titanic = pd.read_excel('titanic.xlsx',sheet_name='passengers')
# pprint(titanic)

'''난수로 데이터 프레임 채우기 '''
# df = pd.DataFrame(np.random.randint(0,100,size=(5,4)),columns=list('ABCD'))
# pprint(df)

'''데이터 프레임 만들어 보기  '''
# countries = pd.read_csv('countries.csv')
# pprint(countries)

'''타이타닉 데이터에서 승객의 나이만 추출하려면?'''
# ages = titanic["Age"]
# pprint(ages.head())

'''타이타닉 탑승객의 이름, 나이, 성별을 동시에 알고 싶으면 '''
# pprint(titanic[["Name","Age","Sex"]])

'''20세 미만의 승객만 추리려면?(필터링)'''
# below_20 = titanic[titanic['Age']<20]
# pprint(below_20.head())

'''1등석이나 2등석에 탑승한 승객들을 출력하려면>?  '''
# 조건식과 유사하게 isin()함수는 제공된 리스트에 있는 값들이 들어있는
# 각 행에 대하여 True를 반환한다. df["Pclass"].isin([1,2])은 Pclass열이 1 또는 2인 행을 확인한다.
# pprint(titanic[titanic["Pclass"].isin([1,2])])

'''20세 미만의 승객 이름에만 관심이 있다면?'''
# (데이터프레임) df.loc[조건(추출을 원하는 조건),열 레이블(추출을 원하는 열레이블)]
# pprint(titanic.loc[titanic["Age"]<20,"Name"])


'''20행~23행  5열~7열에만 관심이 있다면  '''
# (데이터프레임) df.iloc[행범위,열범위]
# pprint(titanic.iloc[20:23,5:7])

'''데이터를 정렬하는 방법 '''
# pprint(titanic.sort_values(by='Age').head())

# pprint(titanic.sort_values(by=['Pclass','Age'],ascending=False).head()) # False 내림차순 True 오름차순


'''열 추가 '''
countries = pd.read_csv('countries.csv')
countries["density"] = countries["population"]/countries["area"]
pprint(countries)
print("==========")
'''행추가'''
df = pd.DataFrame({"code":["CA"],"country":["Canada"],"area":[9984670],"capital":["Ottawa"],"population":[34300000]})
df2 = countries.append(df,ignore_index=True)
pprint(df2)
# ignore_index = True 이부분
'''행 삭제 drop()'''
print("==========")
countries.drop(index=2, axis=0 , inplace= True)
pprint(countries)

'''열 삭제'''
print("==========")
countries.drop(["capital"],axis=1 ,inplace=True)
pprint(countries)

'''타이타닉 승객의 평균 연령은 얼마입니깜?'''
print("==========")
pprint(titanic["Age"].mean())

'''타이타닉 승객 연령과 탑승권 요금의 중간값은 얼마일까'''
print("==========")
pprint(titanic[["Age","Fare"]].median())

'''카테고리별로 그룹화된 통계''' # 잘 쓸수 있도록 연습할 것
print("==========")
pprint(titanic[["Sex","Age"]].groupby("Sex").mean())
# 우리의 관심은 각 성별의 평균 연령이므로 ti sex age에 의하여 이 두열의 선탣이 먼저 이루어짐
# 다음 groupby() 메소드가 "sex"열에 적용되어 'sex'값에 따라서 그룹을 만든다.
# 하여서 각 성별의 평균연령이 계산되어 나온다.

'''성별 및 승객 등급 조합의 평균 탑승권에 요금은 얼마인가>'''
print("==========")
# pprint(titanic.groupby(["Sex","Pclass"])["Fare"].mean())

'''각 승객 등급의 수는 몇명인가?'''
# pprint(titanic["Pclass"].value_counts())

'''데이터로 차트 그리기
df.plot() 와 같이 호출하면 인덱스에 대하여 모든 열을 그린다.
df.plot(x='col1')와 같이 호출하면 하나의 열만을 그린다.
df.plot(x='col1',y='col2')와 같이 호출하면 특정 열에 대하여 다른열을 그리게 된다.

'''

df = pd.DataFrame({
    'name':['Kim','Lee','Park','Choi','Hong','Chung','Jang'],
    'age':[22,26,78,17,46,32,21],
    'city':['Seoul','Busan','Seoul','Busan','Seoul','Daejun','Daejun'],
    'children':[2,3,0,1,3,4,3],
    'pets':[0,1,0,2,2,0,3]
})
# df.plot(kind='line',x='name',y='pets',color='red')

'''중첩 차트 그리기 '''
#gca()로 현재의 Axes를
# ax = plt.gca()
# df.plot(kind='line',x='name',y='children',ax=ax)
# df.plot(kind='line',x='name',y='pets',color='red',ax=ax)
# plt.show()

'''막대 그래프 그리기   '''
# df.plot(kind='bar',x='name',y='age')
# plt.show()

'''산포도 그릭기'''
# df.plot(kind='scatter',x='children',y='pets',color='red')
# plt.show()

'''그룹핑 하여 그리기'''
# df.groupby('city')['name'].nunique().plot(kind='bar')
# plt.show()
#


'''히스토그램 그리기기'''
# bins = 리스트로
# df[['age']].plot(kind='hist', bins=[0, 20, 40, 60, 80, 100], rwidth=0.8)
# plt.show()

'''피벗 테이블   '''
#판다스라이브러리는 값을 깔끔한 2차원 테이블로 요약한 pivot_table()이라는 함수를 제공한다.
titanic.drop(['PassengerId', 'Ticket', 'Name'], inplace=True, axis=1)
# pprint(titanic.head())

#피벗 테이블에서 인덱스를 사용하여 데이터를 그룹화 하자
# table = pd.pivot_table(data=titanic,index=['Sex'])
# pprint(table)
# table.plot(kind='bar')
# plt.show()

'''다중 인덱스로 피봇하기'''
# table = pd.pivot_table(titanic,index=['Sex','Pclass'])
# pprint(table)

'''특징 별로 다른 집계 함수 적용'''
# table = pd.pivot_table(titanic,index=['Sex','Pclass'],
#                        aggfunc={'Age':np.mean, 'Survived':np.sum})
# pprint(table)

'''value 매개변수를 사용하여 특정한 데이터에 대한 집계'''
# table = pd.pivot_table(titanic,index=['Sex','Pclass'],
#                        values=['Survived'],aggfunc=np.mean)
# pprint(table)
# table.plot(kind='bar')
# plt.show()
'''데이터간의 관계찾기'''
# table = pd.pivot_table(titanic,index=['Sex'],
#                        columns=['Pclass'],values=['Survived'],aggfunc=np.sum)
# pprint(table)
# table.plot(kind='bar')
# plt.show()

'''데이터 병합'''
'''
merge()을 사용하면 공통 열이나 인덱스를 사용하여 데이터를 결합한다.
join()을 사용하면 키열이나 인덱스를 사용하여 데이터를 결합한다.
concat()을 사용하면 테이블의 행이나 열을 결합한다.
'''
#merge() 를 사용하면 데이터베이스의 조인연산을 할 수 있다.

# df1 = pd.DataFrame({'employee': ['Kim', 'Lee', 'Park', 'Choi'],
# 'department': ['Accounting', 'Engineering', 'HR', 'Engineering']})
#
# df2 = pd.DataFrame({'employee': ['Kim', 'Lee', 'Park', 'Choi'],
# 'age': [27, 34, 26, 29]})
#
# df3 = pd.merge(df1,df2)
# pprint(df3)

'''결손값 삭제하기 '''
'''
실제 데이터셋들은 완벽하지 않다.
판다스에서는 결손값을 NaN으로 나타낸다. 판다스는 결손값을 탐지하고 수정하는 함수를 제공한다.
'''
df = pd.read_csv('countries1.csv',index_col=0)
# pprint(df)
print("======")
df1 = df.dropna(how="all") # 왜 삭제가 안되지??
print(df1)

'''결손값 보정하기'''

# df_0 = df.fillna(0)
# pprint(df_0)

df_1 = df.fillna(df.mean()['area'])
pprint(df_1)