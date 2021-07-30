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
        'point':[1.5, 1.7, 3.6, 2.4, 2.9]}
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
df2 = pd.DataFrame(data, columns=['year', 'name', 'point', 'penalty'],
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
print("===== 17 ====")
