# pandas 사용하기
'''
Pandas

'''
import numpy as np # numpy 도 함께 import
import pandas as pd
# Series 정의하기
obj = pd.Series([4,7,-5,3])
print("=====2====")
print(obj)  # dtype 데이터 타입 뜸
'''
0    4
1    7
2   -5
3    3
dtype: int64
'''
#Series 의 값만 확인하기
print("=====3====")
print(obj.values)  # dtype 데이터 타입 안뜸


print("=====4 ====")
print(obj.dtype)
# 인덱스를 바꿀 수 있다.
print("=====5 ====")
obj2 = pd.Series([4,7,-5,3], index=['d','b','a','c'])
print(obj2)


print("=====6 ====")
# python 의 dictionary 자료형을 Series data 로 만들 수 있다.
#dictionary의 key 가 Series의 index 가 된다.
sdata = {"Kim":35000, "Beomwoo":67000, "Joan":12000, "Choi": 4000 }
obj3 = pd.Series(sdata)
print(obj3)
print("===== 7 ====")
obj3.name = "Saiary"
obj3.index.name = "Names"
print(obj3)
print("===== 8 ====")
# 인덱스 변경
obj3.index = ['A', 'B', 'C', 'D']
print(obj3)


print("===== 9 ====")
# Data Frame 정의하기
# 이전에 Data Frame 에 들어갈 데이터를 정의해주어야 하는데,
# 이는 python의 dictionary 또는 numpy의 array로 정의할 수 있다.
data = {"name" : ["Beomwoo", "Beomwoo", "Beomwoo", "Kim", "Park"],
        'year':[2013, 2014, 2015, 2016, 2015],
        'points':[1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)
print(df)

print("===== 10 ====")
# 행과 열의 구조를 가진 데이터가 생긴다.
# 행 방향의 index

print(df.index)

print("===== 11 ====")
# 열 방향의 index
print(df.columns)

print("===== 12 ====")
#값 얻기
print(df.values)  # dtype이 안나옴 자료는 나옴

print("===== 13 ====")
# 각 인덱스에 대한 이름 설정하기
df.index.name = 'Num'
df.columns.name = 'info'
print(df)

print("===== 14 ====")
# Data Frame을 만들면서 columns 와 index를 설정 할 수 있다.
df2 = pd.DataFrame(data, columns=['year', 'name', 'points', 'penalty'],
                   index=['one', 'two', 'three', 'four', 'five'])
print(df2)
'''
Data Frame을 정의하면서, data로 들어가는 python dictionary와 columns의 순서가 달라도
알아서 맞춰서 정의된다. 하지만 data에 포함되어 있지 않은 값은 NaN(Not a Number)으로 나타나게 되는데,
이는 null과 같은 개념이다.
NaN값은 추후에 어떠한 방법으로도 처리가 되지 않는 데이터이다.
따라서 올바른 데이터 처리를 위해 추가적으로 값을 넣어줘야 한다. 
'''

print("===== 15 ====")
# describe () 함수는 DataFrame의 계산 가능한 값들에 대한 다양한 계산 값을 보여준다.
print(df2.describe())

print("===== 16 ====")
data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
        "year" : [2014, 2015, 2016, 2015, 2016],
        "points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data, columns=["year", "names", "points", "penalty"],
                        index=["one", "two", "three", "four", "five"])
print(df)


print("===== 17 18 ====")
print(df['year'])

print("===== 19 ====")
print(df[['year','points']])

print("===== 21 ====")
# 특정 열에 대해 위와 같이 선택하고, 우리가 원하는 값을 대입할 수 있다.
df['penalty'] = 0.5
print(df)

print("===== 22 ====")
df['penalty'] = [0.1, 0.2, 0.3, 0.4, 0.5]
print(df)
print("===== 23 ====")
# 새로운 열을 추가하기
df['zeros'] = np.arange(5)
print(df)

print("===== 25 ====")
# Series 를 추가할 수도 있다.
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
df['debt'] = val
print(df)

'''
하지만 Series 로 넣을 때는 val와 같이 넣으려는 data의 index에 맞춰서 데이터가 들어간다.
이점이 python list 나 numpy array로 데이터를 넣을떄와 가장 큰 차이점이다.

'''
print("===== 26 ====")
df['net_points'] = df['points'] - df['penalty']
df['high_points'] = df['net_points'] > 2.0
print(df)

print("===== 27 ====")
#열 삭제 하기
del df['high_points']
print(df)

print("===== 28 ====")
del df['net_points']
del df['zeros']
print(df)

print("===== 29 ====")
print(df.columns)

print("===== 30 ====")
df.index.name = 'Order'
df.columns.name = 'info'
print(df)

print("===== 32 ====")
'''
DataFrame 에서 행을 선택하고 조작하기
pandas 에서는 DataFramedptj 행을 인덱싱 하는 방법이 무수히 많다.
물론 위에서 소개했던 열을 선택하는 방법도 수많은 방법중에 하나에 불과하다. 
'''
# 0번째 부터 2(3-1)번 쨰까지 가져온다.
# 뒤에 써준 숫자번째의 행은 뺀다.
print(df[0:3])

print("===== 33 ====")
# two 라는 행부터 four라는 행까지 가져온다.
# 뒤에 써준 이름의 행을 뺴지 않는다.
print(df['two':'four']) # 비추천 와이?


print("===== 34 ====")
# 요 방법을 권장함
# .loc 또는 .iloc 함수를 사용하는 방법.
print(df.loc['two']) # 반환 형태는 Series

print("===== 35 ====")
print(df.loc['two':'four'])

print("===== 36 ====")
print(df.loc['two':'four', 'points'])

print("===== 37 ====")
print(df.loc[:, 'year']) # == df['year']

print("===== 38 ====")
print(df.loc[:,['year', 'names']]) # == df['year']

print("===== 39 ====")
print(df.loc['three':'five', 'year':'penalty']) # == df['year']

print("===== 41 ====")
# 새로운 행 삽입 하기
df.loc['six',:] = [2013, 'Jone', 2.0, 0.1, 2.1]
print(df)

print("===== 42 ====")
# .iloc 사용 :: index 번호를 사용한다.
print(df.iloc[3]) # 3번째 행을 가져온다.

print("===== 43 ====")
print(df.iloc[3:5, 0:2])

print("===== 46 ====")
print(df.iloc[[0,1,3], [1,2]])

print("===== 47 ====")
print(df.iloc[:, 1:4])

print("===== 48 ====")

print(df.iloc[1,1])
print("===== 49 ====")
'''
DataFrame boolean indexing
'''
print(df)

print("===== 50 ====")
# year가 2014 보다 큰 boolean data
print(df['year'] > 2014)

print("===== 51 ====")
print(df.loc[df['names'] == 'Kilho',['names','points']])

print("===== 52 ====")
# numpy 에서와 같이 논리연산을 응용할 수 있다.
print(df.loc[(df['points']>2)&(df['points']<3),:])

print("===== 53,54 ====")
# 새로운 값을 대입할 수도 있다.
df.loc[df['points']>3,'penalty'] = 0
print(df)

print("===== 55 ====")
#DataFrame을 만들 때 index, column 을 설정하지 않으면 기본값으로 0부터 시작하는 정수형 숫자로 입력된다.
df = pd.DataFrame(np.random.randn(6,4))
print(df)
print("===== 56 ====")
df.columns = ['A', 'B', 'C', 'D']
df.index = pd.date_range('20160701', periods=6)
# pandas에서 제공하는 date_range함수는 datetime 자료형으로 구성된 날짜 시각등을 알 수 있는 자료형을 만드는 함수
print(df.index)

print("===== 57 ====")
print(df)
print("===== 58 ====")
# np.nan 은 NaN값을 의미한다.
df['F'] = [1.0, np.nan, 3.5, 6.1, np.nan, 7.0]
print(df)

print("===== 59 ====")
# 행의 값중 하나라도 nan인 경우 그 행을 없앤다.
print(df.dropna(how='any'))
'''
주의 drop함수는 특정 행 또는 열을 drop하고 난 DataFrame을 반환한다.
즉, 반환을 받지 않으면 기존의 DataFrame은 그대로이다.
아니면, inplace = True라는 인자를 추가하여, 반환을 받지 않고서도 
기존의 DataFrame이 변경되도록 한다.
'''
print("===== 61 ====")
# nan 값에 값 넣기
print(df.fillna(value=0.5))
print("===== 62 ====")

# nan  값인지 확인하기
print(df.isnull())

print("===== 63 ====")
# F열에서 nan값을 포함하는 행만 추출하기  F열
print(df.loc[df.isnull()['F'],:])

print("===== 64 ====")
print(pd.to_datetime('20160701'))

print("===== 65 ====")
# 특정행 drop하기 ??????
print(df.drop(pd.to_datetime('20160701')))

print("===== 66 ====")
# 2개 이상도 가능
print(df.drop([pd.to_datetime('20160702'),pd.to_datetime('20160703')]))

print("===== 67 ====")
#특정 열 삭제하기
print(df.drop('20160701',axis = 0)) #
print(df.drop('F',axis = 1))

print("===== 68 ====")
#2 개 이상의 열도 가능
print(df.drop(['B','D'],axis = 1))

print("===== 70 ====")
'''Data 분석용 함수들 '''
data=[
    [1.4, np.nan],
    [7.1, -4.5],
    [np.nan, np.nan],
    [0.75, -1.3]
      ]
df = pd.DataFrame(data, columns=["one", "two"], index=["a", "b", "c", "d"])
print(df)

print("===== 71 ====")
# 행 방향으로의 합 ( 즉, 각 열의 합)
print(df.sum(axis=0))
'''
이떄 출력값에서 볼 수 있듯이 NaN값은 배제하고 계산한다.
NaN 값을 배제하지 않고 계산하려면 아래와 같이 skipna에 대해 false로 지정해준다
'''
print("===== 72 ====")
print(df.sum(axis=1,skipna=False))

print("===== 73 ====")
# 특정 행 또는 특정 열에서만 계산하기
print(df['one'].sum())
print(df['two'].sum())

print("===== 74 ====")
print(df.loc['a'].sum())

print("===== 75 ====")
#  행값 periods 값 같아야지~ 돈두르마돈두르마 잡아야쥐~
df3 = pd.DataFrame(np.random.randn(6,4),
                   columns=['A','B','C','D'],
                   index=pd.date_range("20160701",periods=6))
print(df3)
print("===== 76 ====")
# A열과 B열의 상관계수 구하기
print(df3['A'].corr(df3['B']))

print("===== 77 ====")
#B열 과 C열의 공분산 구하기
print(df3['B'].cov(df3['C']))

print("===== 78 ====")
dates = df3.index
random_dates = np.random.permutation(dates)
df3 = df3.reindex(index = random_dates, columns = ["D","B","C","A"])
print(df3)

print("===== 79 ====")
# index 와 column 의 순서가 섞여있다.
# 이때 index가 오름차순이 되도록 정렬해보자
print(df3.sort_index(axis=0))

print("===== 80 ====")
# col을 기준
print(df3.sort_index(axis=1))

print("===== 81 ====")
# 내림차순으로 ㄴ,ㄴ?
print(df3.sort_index(axis=1, ascending=False))

print("===== 82 ====")
# 값기준 정렬
# D열의 값이 오름차순이 되도록
print(df3.sort_values(by='D'))

print("===== 83 ====")
# B열의 값이 내림차순이 되도록 정렬하기
print(df3.sort_values(by='B',ascending=False))


print("===== 84 ====")

df3["E"] = np.random.randint(0,6,size=6)
df3["F"] = ["alpha", "beta", "gamma", "gamma","alpha", "gamma"]
print(df3)

print("===== 85 ====")
#E열과 F열을 동시에 고려하여, 오름차순으로 하려면?
print(df3.sort_values(by=['E','F']))

print("===== 86 ====")
# 지정한 행 또는 열에서 중복값을 제외한 유니크한 값만 얻기
print(df3['F'].unique())

print("===== 87 ====")
# 지정한 행 또는 열에서 값에 따른 개수 얻기
print(df3['F'].value_counts())
print("===== 88 ====")
#지정한 행 또는 열에서 입력한 값이 있는지 확인하기
print(df3['F'].isin(["alpha", "beta"]))
print("===== 89 ====")
#F열의 값이 알파나 베타인 모든 행 구하기
print(df3.loc[df3['F'].isin(["alpha", "beta"]),:])
print("===== 90 ====")
df4 = pd.DataFrame(np.random.randn(4,3), columns=["b","d","e"],
                   index= ["서울","인천","부산","대구"])
# print(df4)
# print("===== 91 ====")
# func = lambda x: x.max() =x.min()  # 여기 마지막 왜지?
# print("===== 91 ====")
# df4.apply(func, axis=0)