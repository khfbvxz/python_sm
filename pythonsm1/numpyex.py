# numpy 사용하기
import numpy as np



data1= [1,2,3,4,5]
print(data1)
data2 = [1,2,3,3.5,4]
print(data2)
# np 이용 array 정의
# 위에서 만든 python list 를 이용

arr1 = np.array(data1)
print(arr1)

# array 의 형태(크기)를 확인할 수 있다.
arr1.shape
print(arr1.shape)

# 바로 파이썬 리스트를 넣어 줌으로써 만들기
arr2 = np.array([1,2,3,4,5])
print(arr2)

print(arr2.shape) #shape

# array 의 자료형을 확인할 수 있다.
arr2.dtype # 자료형을 확인할 수 있다.
print(arr2.dtype)  # int32

arr3 = np.array(data2)
print(arr3)
print(arr3.shape)
print(arr3.dtype)

arr4 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(arr4)
print(arr4.shape) # 4,5  행렬 크기
'''
부로가 있는 정수 int(8,16,32,64)
부호가 없는 정수 uint(8,16,32,64)
실수    float(16,32,64,128)
복소수   complex(64,128,256)
불리언 bool
문자열 string_
파어썬 오브젝트 object
유니코드 unicode

'''

# np.zeros(), np.ones(), np.arange()
print(np.zeros(10))

print(np.zeros((3,5)))
print(np.ones(9))
# np.ones() 함수는 인자로 받는 크기만큼, 모든요소가 1인 array를 만든다.
print(np.arange(10)) #[0 1 2 3 4 5 6 7 8 9]  # 1씩 증가
print(np.arange(3,10)) # [3 4 5 6 7 8 9]
# 3이상 9 미만
arrr1 = np.array([[1,2,3],[4,5,6]])
print(arrr1)
print(arrr1.shape)
arrr2 = np.array([[10,11,12],[13,14,15]])
print(arrr2)
print(arrr2.shape)

print(arrr1+arrr2)
# 행렬곱셈 주의 #각 요소별로 곱셈이 진행된다.
print(arrr1*arrr2)
# [[10 22 36]
#  [52 70 90]]
print(arrr1/arrr2)

# array의 브로드 캐스트
#서로 크기가 다른 array가 연산이 가능하게끔하는것

print(arrr1) #arrr1 = np.array([[1,2,3],[4,5,6]])
print(arrr1.shape)
arrr3 = np.array([10,11,12])  # 각각을 넣어줌
print(arrr3)
print(arrr3.shape)
print(arrr1+arrr3)  # arrr3 가 [[10,11,12],[10,11,12]] 로 확장되어 계산됨
# #[[11 13 15]
 # [14 16 18]]
print(arrr1*arrr3)
# [[10 22 36]
#  [40 55 72]]
#스칼라 연산
print(arrr1*10)
# 요소에 대해 제곱처리
print(arrr1**2) # dtype = int 32

'''Array 인덱싱'''
ar1 = np.arange(10)
print(ar1)
print(ar1[0])
print(ar1[3:9])
print(ar1[:])

# 1차원이 그이상의 차원에서도 인덱싱 가능
ar2 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

'''
ar2 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
[[1,2,3,4],       0행 row   열 col 
 [5,6,7,8],        1행 
 [9,10,11,12]]      2행  
'''
print(ar2)
# 2차원의 인덱싱을 하기위해선 2개의 인자를 입력해야 합니다.
print(ar2[0,0])  # 요소 꺼내기 r , c 넣으면 0부터 시작!
print(ar2[2,:]) #[ 9 10 11 12]  # 3행 모든요소
print(ar2[:,3]) # 모든행 3번째 요소

# Array booleab 인덱싱  (마스크) 주로 마스크라고 함
# booleab 인덱싱
# 우리가 원하는 행 또는 열 의 값만 뽑아낼 수 있다.
# 마스크 처럼 우리가 가리고 싶은 부분은 가리고 원하는 요소만 꺼내고 있다. 졸리다

names = np.array(['Beomwoo','Beomwoo','Kim', 'Jone', 'Lee','Beomwoo','Park','Beomwoo'])
print(names)    #    0          1       2       3       4
print(names.shape)

#아래에서 사용되는 np,random,randn() 함수는 기대값이 0이고,
#표준편차가 1인 가우시안 정규분포를 따르는 난수를 발생시키는 함수이다.
#이 외에도 0~1의 난수를 발생시키는 np.random.rand() 함수도 존재한다.

data3 = np.random.randn(8,4)
print(data3)
print(data3.shape)

#요소가 범우인 항목에 대한 마스크 생성

names_Beonmwoo_mask = (names == 'Beomwoo')
print(names_Beonmwoo_mask) # [ True  True False False False  True False  True]
print('fsf')
# data3에서 0,1,5,7, 행의 요소를 꺼내 와야 한다.
#이를 위해 요소가 범우인 것에 대한 boolean 값을 가지는 mask를 만들었고
# 마스크룰 인덱싱에 응용하여 data의 0157행 을 꺼냈다.
print(data3[names_Beonmwoo_mask,:])
print('fsf')
# 요소가 Kim 인 행의 데이터만 내기
print(data3[names == 'Kim',:])
# 논리연산을 응용하여 요소가 kim 또는 park인 행의 데이터만 꺼내기
print('fsf')
print(data3[(names == 'Kim') | (names == 'Park'),:])
print('fsf')

# data array 자체적으로도 마스크를 만들고, 이를 응용하여 인덱싱이 가능하다.
# data array에서 0번쨰 열의 값이 0보다 작은 행을 구해보자

# 먼저 마스크 만들기
# data array에서 0 번째 열이 0보다 작은 요소의 boolean 값은 다음과 같다.

print(data3[:,0] < 0)
# 랜덤임 실행할떄마다 바뀌 왜냐 위에서 data3를 랜덤 난수로 했기 때문
#[False False False False False  True  True False]
print('fsf')
#위에서 만든 마스크를 이용하여 0번째 열의 값이 0보다 작은 행을 구한다.
print(data3[data3[:,0]<0,:])
# 이런식으로 특정 위치에만 우리가 원하는 값을 대입할 수 있대
print('fsf')
print(data3[data3[:,0]<0,2:4])
print(data3)
data3[data3[:,0]<0,2:4]=0

print(data3)

print('fsf')
# numpy 함수
ar3 = np.random.randn(5,3)
print(ar3)
print('fsf')
# 각 성분의 절대값 계산 하기
print(np.abs(ar3))
print('fsf')
# 각 성분읜 제곱근 계산하기 (== array **0.5)
print(np.sqrt(ar3))
print('fsf')
'''
[[       nan 1.08411123 1.29757525]
 [       nan 0.91029063        nan]
 [1.08974343        nan 0.9413238 ]
 [0.46633752        nan 0.87936018]
 [0.1196997         nan        nan]]
fsf
C:/Users/yuhwan/PycharmProjects/pythonsm1/numpyex.py:176: RuntimeWarning: invalid value encountered in sqrt
  print(np.sqrt(ar3))

Process finished with exit code 0

'''

print(np.square(ar3)) # 제곱
print('fsf')

# 각 성분을 무리수 e의 지수로 삼은 값을 계산하기
print(np.exp(ar3))
print('fsf')
#각 성분을 자연로그 , 상용로그. 밑이 2인 로그를 씌운 값을 계산하기
print(np.log(ar3))
print('fsf67')
print(np.log10(ar3))
print('fsf68')
print(np.log2(ar3))
print('fsf69')

# 각 성분의 부호 계산하기 (+인 경우 1,  -인 경우 -1 , 0 인 경우 0)
print(np.sign(ar3))
print('fsf70')

# 각 성분의 소수 첫 번째 자리에서 올림한 값을 계산하기
print(np.ceil(ar3))
print('fsf71')
# 소수 첫번쨰 자리에서 내림한 값
print(np.floor(ar3))
print('fsf72')
# 각 성분이 NaN인 경우 True 를 아닌경우 False 를 반환하기
print(np.isnan(ar3))
print('fsf74')

# print(np.isnan(np.log(ar3)))
print('fsf75')

# 각 성분이 무한대인 경우 True 를 아닌 경우 False 를 반환 하기
print(np.isinf(ar3))
print('fsf76')

# 각 성분에 대해 삼각함수 값을 계산하기 (cos, cosh, sin, sinh, tan, tanh)
print(np.cos(ar3))
print('fsf77')


''' 6.2 '''
print(ar3)
print('fsf79')
ar4 = np.random.randn(5,3)
print(ar4)
print('fsf80')

# 두개의  array 에 대해 동일한 위치 성분끼지 연산 값을 계산
# add subtract multiply divide
print(np.multiply(ar3,ar4))
print('fsf81')
#
print(np.maximum(ar3,ar4))
print('fsf82')

# 통계함수
# 통계 함수를 통해 array의 합이나 평균들을 구할 떄,
# 추가로 axis라는 인자에 대한 값을 지정하여 열 또는 평균등을 구할 수 있다.
print(ar4)

# 전체 성분 의 합을 계산
print(np.sum(ar4))
print('fsf84')

# 열 간의 합을 계산
print(np.sum(ar4, axis=1))
print('fsf85')

# 행 간의 합을 계삽
print(np.sum(ar4, axis=0))
print('fsf86')

# 전체 성분의 평균을 계산
print(np.mean(ar4))
print('fsf87')
# 행 간의 평균 계산
print(np.mean(ar4, axis=0))
print('fsf88')

# 전체 성분의 표준 편차 ,. 분산 , 최소 최대값 std var min max
print(np.std(ar4))
print('fsf89')
#

print(np.min(ar4, axis=1))
print('fsf90')
#
# 전체 성분의 최소값, 최대값이 위치한 인덱스를 반환(argmin, argmax)
print(ar4)
print(np.argmin(ar4))  # defult 값을 주어졌을 떄
print('fsf91')


print(np.argmin(ar4,axis=0))  # 0 col기준  1 이면 row기준
print('fsf93')

# 맨 처음 성분부터 각 성분까지의 누적곱을 계산 (cumsum , cumprod)

print(np.cumsum(ar4))  # 0 col기준  1 이면 row기준
print('fsf94')

print(np.cumsum(ar4,axis=1))  # 0 col기준  1 이면 row기준
print('fsf95')
print(np.cumprod(ar4))  # 0 col기준  1 이면 row기준
print('fsf95')

# 6.4 기타합수
print(np.sort(ar4))
print('fsf97')
print(np.sort(ar4)[::-1])
print('fsf98')


print(np.sort(ar4,axis=0)) # 행 방향으로 오름차순
print('fsf99')

print(data3[data3[:,0]<0,2:4])
print(data3)

