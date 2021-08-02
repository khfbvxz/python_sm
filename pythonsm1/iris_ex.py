import numpy as np
from pprint import *
import matplotlib.pyplot as plt
# # plot_test.py
# data1=np.random.random(10)*10
# print(data1)
# data2=list(range(10))
# print(data2)
# plt.plot(data1, label='numpy')
# plt.plot(data2, ':', label='list')
# plt.legend()
# plt.show()
'''iris  불러오기 1'''
f = open('iris.csv')   # 아이리스 파일을 열어라
line = f.readline()    # 연 파일을 한 줄 씩 읽어라  ,  ,
features = line.strip().split(',')[:4] # .strip()함수 whitespace 제거 , 기준으로 split 3열 까지
# 숫자만 features 저장
data = []
for line in f:
    l = line.strip().split(',') # , 기반으로 나눠라
    l[0] = float(l[0])  # 실수형 형변환
    l[1] = float(l[1])
    l[2] = float(l[2])
    l[3] = float(l[3])
    if l[4] == 'Iris-setosa': l[4] = 0
    elif l[4] == 'Iris-versicolor': l[4] = 1
    else: l[4] = 2
    data.append(l) # 빈 리스트 data 에 저장
f.close() # 파일 닫아라
iris = np.array(data)  # 그 데이터를 array해서 iris에 저장
print(iris)

'''ilis 데이터 불러오기 2'''
import numpy as np
from pprint import *
f = open('iris.csv')
line = f.readline()
features = line.strip().split(',')[:4]
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  #
#             0             1                     2
data = []
for line in f:
    l = line.strip().split(',')
    l[:4] = [float(i) for i in l[:4]]
    l[4] = labels.index(l[4])
    data.append(l)
f.close()
iris = np.array(data)
print(iris)

'''데이터 불러오기'''
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#        dataset파일 읽어오기
#              파일 경로               파일에서 사용한 구분자 데이터 타입 지정
# decode는 컴퓨터가 문자열을 불러올때 숫자형태로 불러오는데 그것을 우리가 아는 문자로 불러오기 위한 규칙 <> incode
# iris = np.loadtxt("iris.csv", skiprows=1, delimiter=',', converters={4: lambda s: labels.index(s.decode)})
iris = np.loadtxt('iris.csv', skiprows=1, delimiter=',', converters={4: lambda s: labels.index(s.decode())})
iris2_norm = (iris - iris.mean(axis=0))/iris.std(axis=0)

# Iris = np.loadtxt('iris.csv', skiprows=1, delimiter=',',
#                  converters={4: lambda s: labels.index(s)},
#                   encoding = 'latin1')
# print(iris.shape)
# print(iris[0])
# print(iris[:,4])
# # pprint(iris.shape)
# # pprint(iris[0])
# # pprint(iris[:,4])
# # print(iris2_norm.shape)
# #
# columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
# plt.plot(iris[:,:4])
# plt.legend(columns)  #범례표시
# plt.title('don durrrrma~~~~')
# plt.xlabel('X Data')
# plt.ylabel('Y Data')
# plt.show()

'''
Numpy 어레이
Numpy는 ndarray형태의 자료구조를 처리하는 모듈이다.
ndarray 는 다차원 배열인데 2차원에서는 직사각형 3차원에서는 직육면체 형태이다.
ndarray 는 고급 수치계산을 위해 list 기능을 확장한 것으로 볼 수있다.


'''
l = [1,2,3]
a=np.array(l)
print(l)
print(a)
print(type(a),len(a))
print(a.shape,a.dtype,a.ndim)  # .ndim 함수  몇 차원인지.

a2 = np.array([[1,2,3],[4,5,6]])
print(a2)
print(len(a2),a2.size)
print(a2.shape, a2.dtype, a2.ndim)

print([i+10 for i in l])
print(a+10)
print(np.sin(a))
print(a+a)
l = [[1,2,3],[11,12,13]]
a=np.array(l)
print(a[:,0])
print([i[0] for i in l])# [] 안하면 이렇게 출력 됨 <generator object <genexpr> at 0x000001F43771DAC0>
print("----")
print(a[a.mean(axis=1)>10]+[1,2,3])


print("----")
print(np.argsort([3,1,2,4])[::-1]) # dtype = int 64 [3 0 2 1]
print("----")
print(np.argsort([3,1,2,4])[::-2]) # dtype = int 64 [3 2]
print("----")
print(np.argsort([3,1,2,4])[::]) # dtype = int 64 [3 2] [1 2 0 3]

print("----")
c=np.array([1,2,3],dtype=float)
print(c.dtype)

print("----")
a=np.array(['a','b','c']) # dtype = str
print(a)
print("----")
a=np.array([1,2,3],dtype='str') # array([‘1’, ‘2’, ‘3’], dtype=‘<U1’) #중요
print(a)
print("----")
a=np.array(['1','2','3'],dtype=int) # array([1, 2, 3])
print(a)
print("----")
a=np.array([0,1,2],dtype=bool) # array([False, True, True])
print(a)
print("----")
a = np.array([[1], [2, 3]])
print(a,a.dtype,type(a))

''' 
어레이는 list,tuple,range로 부터 생성항 수 있다 
tuple 도 가능하나 일반적으로 list 사용!!!!!!
dict 와 set 은 사용 xxxxxx

'''
''' 기존 array 변경'''
q = np.array([1,2,3])
print(q,q.dtype)
w = np.array(q,dtype = float)
print(w,w.dtype)
e = np.array(q,dtype = str)
print(e,e.dtype)
q = np.array(q, e.dtype)
print(q,q.dtype)
print("=====")


print(np.zeros(10))   # 실수형!!
# print(np.zeros(2,3))  #에서!!
print(np.zeros((2,3)))
print(np.zeros([2,3,4])) #
# print(np.zeros(2,3))  #에서!!
print(np.ones([2,3,4])) #
h=np.ones([2,3,4]) # 실수형
print(h.dtype)
print(np.eye(3)) # 단위행렬 대각 행렬

print("=====")
'''어레이 생성 함수 '''

print(np.arange(10))
print("=====")
print(np.arange(1,10,2))
print("=====")
print(np.arange(1,10,2,dtype=float)) #[1. 3. 5. 7. 9.]
print("=====")
print(np.arange(8).reshape(2,2,2))
print("=====")
print(np.arange(8).reshape(2,-1))  # -1로 지정하면 자동할당됨

'''어레이 생성 함수 3'''
print(np.random.rand(3,3)) # 0~1 사이의 실수
print(np.random.randn(2,5)) # 평균 0 , 표준편차 1인 정규 분포
print(np.random.randint(10,size=(3,3)))  # 0~9사이의 정수
print("======")
print(np.random.uniform(100,101,size=(2,2))) # 100~101 사이의 실수
print("======")
print(np.random.normal(10,2,size=(2,2))) # 평균 10 , 표준편차 2인 정규분포
print("======")
print(np.random.choice([0,1],size=10))  # 0 과 1 에서 10개를 골라낸다
print(np.random.choice(['dog','cat'],size=(2,5))) # 문자열도 가능

'''연습문제 4'''
print("======")
a=np.random.rand(3,2)
print(a , a.shape)
print(np.arange(100).reshape(10,10))
a1 = np.array([1,2,3])
print(a1.shape) # shape 는 (3,) 3열 또는 x 축으로 3 개
a2 = np.array([[1,1],[2,2],[3,3]])
'''
1  2  3
1  2  3 
'''
print(a2.shape) # (3,1) axis=0
print(a2.sum(axis=0)) #[6 6]
print(a2.sum(axis=1)) # [2 4 6]


'''numpy 기본 연산 '''
# numpy는 list와 다르게 항목별로 계산한다.
print("=======")
c = np.arange(4).reshape(2,2)
d = np.arange(10,14).reshape(2,2)
print(c+d)
print(c%d)
print("=======")
print(c+1)
print(1/(c+1))
print(c**2)
print(np.zeros([2,2])+5,(np.zeros([2,2])+5).dtype)
print("=======")
print(np.sqrt(c))
print(np.exp(c))  # e ^ (항목별)
print(np.sin(c))  # rad 안 기준
print(np.ceil(np.sin(c)))  # ceil 올림
print("=======")
e = np.random.randn(3,4)
print(e)
print(np.sign(e))  # sign 부호 계산!
print(np.floor(e)) # 소수 첫번째 자리에서 내림 !!
print(np.abs(e))

'''색인과 슬라이싱 '''
print("=======")
f = np.arange(10)
print(f)
print(f[0], f[1], f[2], f[-1], f[-2], f[-3])
print("=======")
print(f[:-1])
print("=======")
print(f[3:7])
print("=======")
# 값 할당 가능
f[3:6]=-1
print(f)
print("=======")
g = np.arange(10)[::-1]
print(g)
h=g[::2] # h 는 g 일부분의 뷰
print(h)
'''
2 일때  [9 7 5 3 1] 
3 일때  [9 6 3 0] 
4 일떄  [9 5 1]
-2 일때 [0 2 4 6 8]
'''
h[0] = 999 # 값 할당 가능
print(g)
o = np.arange(12).reshape(3,4)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
print(o)
print(o[0])
print(o[-1])
print(o[:2,:2])
print(o[:,1::2]) # 부분만 꺼내는 연습 해야겠다
print("=======")
'''색인에서 항목별 선택 '''
# 이런 경우를 팬시 색인이라고 한다.
# 원하는 항목을 리스트에 색인에 넣으면 된다.
# 뒷장에 나오는 불리언 색인과 결합하여, 고급 계산에서 자주 사용됨
p = np.arange(12).reshape(3,4)
print(p)
print(p[[0,2]]) # 첫번쨰, 세번째 행  [[ ]] 주의
print(p[:,[0,2]]) # 첫번째 세번째 열
print(p[1])
print(p[[1]]) # 원 구조를 유지하면서 골라냄
print("=======")
print(p[[1,1]])
print("=======")



k = np.eye(3)
pprint(k)
l = [2, 0, 1, 0, 1, 2, 1]

pprint(k[l]) # One-Hot 인코딩 사례임?  왜 다르지?
qw = np.arange(12).reshape(4,3)
print("=======")
pprint(qw[ [1, 3] , [0, 2] ] )
pprint(qw[[1,3]] [:,[0,2]])
# pprint(qw[[1,3]] [:,[0,2]])

'''Numpy  불리언 색인'''
# 사칙연산을 하듯이, 조건식을 적용하면 어떻게 될까?
# 각 항목별로 True, False 가 할당 된다.
# 어레이 간의 조건식에는 , "|" or , 와  & and 가 사용된다,
ad = np.random.seed(1) # 랜덤 values 고정 # seed 마다 각 랜덤 values 를 저장되는?
ad = np.random.randn(3,3)
pprint(ad)
print("=======")
pprint(ad>0)  # 맞는기 안맞는지만 나옴
print("=======")
pprint((ad>1)|(ad<-1))
pprint((ad>-0.5)&(ad<0.5))
pprint(ad[ [True, False, True]]) # 첫번째 세번째 행
pprint(ad[:, [True, False, True]]) # 모든행에 1열(index=0) 3열(index=2)만
pprint(ad[ad>0]) # ad array에 0 보다 큰 것만
ad[ad>0] = 0
pprint(ad)

''' 행 골라내기 '''
#불리언 색인을 이용하여 SQL문 처럼 원하는 행을 골라내 보자
np.random.seed(55)
af = np.random.randn(100,3)
df = af.mean(axis=1) # mean 평균값
pprint(df.shape)
pprint(af[df>0])
cf = np.random.choice(['one', 'two', 'three'],100)
pprint(cf.shape)
print("=======")
pprint(af[cf=='one'])
pprint(af[cf=='one'].shape)


'''축 바꾸기'''
print("=======")
ae = np.arange(10)
pprint(ae.T)
ae = ae.reshape(2,5)
pprint(ae.T) # 전치 행렬

pprint(ae.reshape(5,2))

ar = np.arange(12).reshape(3,4)
pprint(ar.shape)
pprint(ar)
arb = ar.T
pprint(arb.shape)
pprint(arb)


''' 통계함수 '''
# 평균 표준편차, 최소, 최대 등을 구해보자~~
# 다차원 어레이인 경우 axis 옵션을 적용한다!!
print("=======")
ea = np.arange(12).reshape(3,4)
pprint(ea)
'''
       아래로 axis= 0 x축  # 오른쪽으로 axis =  y축 [ [ 두번쨰 부터 x1 에 대한 y 로 쌓여지는
array([[ 0,  1,  2,  3], y=0       
       [ 4,  5,  6,  7], y=1 
       [ 8,  9, 10, 11]])y=2     
        x=0 x=1 x=2 x=3         
'''

pprint(ea.sum()) # None 각 요소들을 다 더함
pprint(ea.sum(axis=0)) # 열간의 합 각 행의 열간  x 가 같은값끼리 더함
pprint(ea.sum(axis=1)) # 행간의 합 각 열의 행간  y 가 같은값끼리 더함

print("=======")
pprint(ea.std(axis=0)) # 표준편차
pprint(ea.min())
pprint(ea.min(axis=0))
pprint(ea.min(axis=1))
print("=======")
pprint(ea.max())
pprint(ea.max(axis=0))
pprint(ea.max(axis=1))
print("=======")
pprint(ea.cumsum())
pprint(ea.cumsum(axis=0))  # 누적합 x 축 방향으로
'''array([[ 0,  1,  2,  3],
       [ 4,  6,  8, 10],
       [12, 15, 18, 21]], dtype=int32)
'''
pprint(ea.cumsum(axis=1)) # 누적합 y 축 방향으로
'''array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]], dtype=int32)
       '''
print("=======")
# 전체 성분의 최소값, 최대값이 위치한 인덱스를 반환(argmin, argmax)
pprint(ea.argmin(axis=0))
pprint(ea.argmin(axis=1))
pprint(ea.argmax(axis=0))
pprint(ea.argmax(axis=1))

print("=======")
np.random.seed(10)
ew = np.random.rand(10,3)
rt = ew.mean(axis=1)>0.5
pprint(rt)
#배열의 데이터 비교방법 any,all,where, isnan,argmin,argmax
pprint(np.where(rt)) # where  조건의 맞는 인덱스 출력
pprint(np.where(rt)[0])
pprint(ew[ew.mean(axis=1)>0.5])
pprint((ew>0.5).any(axis=1)) # 1개 이상? any 조건 함수
pprint((ew>0.5).all(axis=1)) # all 조건 배열의 모든 데이터가 조건이 맞으면 true 하나라도 다르면 false

''' 정렬 '''

print("=======")
np.random.seed(1)
data1 = np.random.randint(10,size=(3,4))
pprint(data1)
'''array([[5, 8, 9, 5],
       [0, 0, 1, 7],
       [6, 9, 2, 4]])d
'''
ty=data1.copy()
ty.sort()
pprint(ty) # 각 행간만 정렬
print("=======")
ty1=data1.copy()
ty1.sort(axis=0)
pprint(ty1)
print("=======")
ty2=data1.copy()
ty2.sort(axis=1)
pprint(ty2)
pprint(data1)
# 첫번쨰 값으로 정렬하자?
pprint(np.argsort([5,0,6])) # 위치를 반환???
sorter = np.argsort(data1[:,0])# array([1, 0, 2], dtype=int64)
pprint(sorter) # array([1, 0, 2], dtype=int64)
pprint(data1[sorter])

'''np.save() np.load()  어레이 저장과 불러오기'''
print("=======")
a = np.random.randn(1000, 100)
pprint(a.shape)
pprint(a[0,0])
np.save('1.npy',a)  # 파일 이름 , 어레이
aa=np.load('1.npy')
pprint(aa.shape)
pprint(a[0,0])

a3=np.random.randn(10,10)
b3=np.random.randn(10,10)
np.save('2.npy',[a3,b3])
aa,bb = np.load('2.npy')
pprint(aa) # pprint(aa, bb) 는 에러남
pprint(bb)
print("=========")






'''axis 이해'''

# arr = np.arange(0, 32)
# print(len(arr))
# print("-----")
# print(arr)
# print("-----")
# v = arr.reshape([4,2,4])
# print(v)
# print("-----")
# print(v.ndim)
# print("-----")
# print(v.sum())# axis = none
# print("-----")
# res01 = v.sum(axis=0)
# print(res01.shape)
# print(res01)
# print("-----")
# res02 = v.sum(axis=1)
# res02_1=v.sum(axis=0)
# print(res02.shape)
# print(res02)
# print(res02_1.shape)
# print(res02_1)
#
#
# print("-----")
#
# res03 = v.sum(axis=2)
# print(res03.shape)
# print(res03)