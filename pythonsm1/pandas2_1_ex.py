import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import *

'''2.1 판다스 데이터 분석'''

''' pandas 로 데이터 읽어오기   '''
print("====17=====")
iris_df = pd.read_csv('iris.csv') # read_ 파일 확장자 맞춰야하는듯
# print(iris_df) # dataframe

print("====18=====")
# print(iris_df.info()) # info() 정보 메서드
print("====19=====")
print(iris_df.describe()) # describe() 메서드 데이터 요약 다양한 통계량 요약
print("====20=====")
'''
마지막 속성인 품종을 숫자로 바꾸기
# 마지막 속성인 품종은 문자열로 되어 있다.
머신러닝에서는 숫자로 된 데이터만 처리할 수 있다.
그러므로, 품종 세가지를 각각 0,1,2 인 숫자로 바꾸어 보자

'''
# print(iris_df['Name'])

print("====21=====")
pprint(iris_df['Name'].unique())
print("====22=====")
label = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2} # 딕셔너리
pprint(iris_df['Name'].map(label)) # 위의 딕셔너리 value 들로 매핑 해주는 함수
print("====23=====")
iris_df['Name'] = iris_df['Name'].map(label)
print("====24=====")
pprint(iris_df)


''' numpy 데이터로 바꾸기 '''
# 머신러닝의 다양한 알고리즘 들은 주로 numpy 데이터 타입을 다룬다.
# 위에서 구한 pandas 데이터를 numpy데이터로 바꾸어 주자
print("====26=====")
iris = iris_df.values # pandas 데이터 모두 numpy 데이터로 가져온다,
pprint(iris)
pprint(type(iris))
# help(iris) # 얘는 출력 다시해서 보자
# p62 로 뛰셈

print("====29=====")
pprint(iris[0]) # 첫번째 샘플
print("====30=====")
pprint(iris[1,2]) # 두번째 샘플의 세번째 값
print("====31=====")
pprint(iris[:,0]) # 첫 번째 속성값

print("====32=====")
pprint(iris[:,0].sum()) # 첫번째 속성의 합계
pprint(iris[:,1].sum()) # 두번째 속성의 합계

print("====33=====")
a = [[1,2], [3,4], [5,6]]
r = 0
for i in a:
    r = r + i[0]
print(r)
print("====34=====")
pprint(sum([i[0] for i in a]))
print("====35=====")
a_array = np.array(a)
pprint(a_array[:,0].sum())
print("====36=====")
pprint(iris[:,0].mean())
print("====37=====")
pprint(iris[:,0].std())

''' 정리 '''
print("=====38=====")
iris_df = pd.read_csv('iris.csv')

label = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris_df['Name'] = iris_df['Name'].map(label)

iris = iris_df.values
pprint(type(iris))
pprint(iris.dtype)

print("=====39=====")
# plt.plot(iris)
# plt.legend(iris_df.columns)
# plt.show()

print("=====40=====")
# plt.plot(iris[:50,:4].T, 'r-', alpha=0.1)  # 띄어쓰기 주의
# plt.plot(iris[50:100,:4].T, 'g-', alpha=0.1)
# plt.plot(iris[100:,:4].T, 'b-', alpha=0.1)
# plt.show()
print("=====41=====")

# scatter() 다시 !!!!
# plt.scatter(iris[:,2], iris[:,3], c=iris[:,4], s=100, alpha=0.2) # c는 컬러 #s는 size #
# plt.colorbar()
#
# plt.title('PeralLength vs PetalWidth')
# plt.xlabel('PetalLength')
# plt.ylabel('PetalWidth')
# plt.show()

print("=====43=====")
# # subplot !!
# plt.subplot(2,2,1)
# plt.title('SepalLength')
# plt.hist(iris[:,0], bins=30)
#
# plt.subplot(2,2,2)
# plt.title('SepalWidth')
# plt.hist(iris[:,1], bins=30)
#
# plt.subplot(2,2,3)
# plt.title('SepalLength')
# plt.hist(iris[:,2], bins=30)
#
# plt.subplot(2,2,4)
# plt.title('SepalWidth')
# plt.hist(iris[:,3], bins=30)
# plt.show()

print("=====48=====")
''' 머신러닝 적용 '''
# 가까운 이웃 알고리즘을 이용한 분류
# 여기서 다시 해보자

