import numpy as np
from pprint import *
'''텍스트 파일 불러오기 1'''
f = open('iris.csv')

line = f.readline() # 레이블 이름인 첫번째 줄 읽기
header = line.strip().split(',')
header = [i.strip('"') for i in header]
data = []
label = []

for line in f: # 나머지 라인
    l = line.strip().split(',')
    dl=[float(i) for i in l[:4]] # 4번째 칼럼 까지 읽기

    data.append(dl)
    label.append(l[4].strip('"')) # 마지막 칼럼 읽기?
f.close()

X=np.array(data); pprint(X.shape)
y=np.array(label); pprint(y.shape)


''' text 파일 불러오기 2  '''
print("========")
a=np.loadtxt('iris.csv',delimiter=',',skiprows=1,usecols=(0,1,2,3))
pprint(a.shape)
pprint(a[0])

name={'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2} # 5번째 칼럼을 실수로 변환한다.
a=np.loadtxt('iris.csv',delimiter=',',skiprows=1,converters={4: lambda  x: name[x.decode().strip('"')]})
pprint(a.shape)
X=a[:,:-1]
pprint(X.shape)
Y=a[:,-1]
pprint(Y.shape)


'''     '''
# header = line.strip().split(',')
# header = [i.strip('"') for i in header]
# X=a[:,:-1]
# Y=a[:,-1]
f = open('iris2.csv','w')
f.write(','.join(header)+'\n')
for i,j in zip(X,Y):
    f.write('%f,%f,%f,%f,%s\n' % (i[0],i[1],i[2],i[3],j))
f.close()

'''어레이 합치기 '''
a1=np.arange(4).reshape(2,2)
b1=np.arange(4).reshape(2,2)
b1=np.arange(10,14).reshape(2,2)
pprint(a1)
pprint(b1)
print("=======")
pprint(np.hstack([a1,b1])) # [a1, b1] 와 같이 [ ] 로 묶어야함! h 는 행으로
print("=======")
pprint(np.vstack([a1,b1])) # v 는 열로
print("=======")
pprint(np.c_[a1,b1]) # col이 추가되는 (행이 같으면)
pprint(np.r_[a1,b1]) # row이 추가되는 (열이 같으면)

'''브로드 캐스팅 1'''
print("=======")
np.random.seed(11)
a=np.random.randint(10,size=(3,2))
pprint(a)
b=a.mean(axis=0)
pprint(b)
pprint(a-b)
print("=======")
c=np.arange(4).reshape(2,2)
pprint(c)
pprint(c-[1,2])  # 각 행에 - [1,2]
pprint(c-[[1,2]]) # 각 행에 - [1,2]
pprint(c-[[1],[2]]) # 1행에 각 성분마다 - [1] 2행에 각 성분마다 - [2]

# pprint( a-a.mean(axis=1) ) # 1차원 배열은 아래로 확장됨을 알 수 있다.
#ValueError: operands could not be broadcast together with shapes (3,2) (3,)
pprint(a-a.mean(axis=1).reshape(3,1))

'''브로드 캐스팅 2'''
print("=======")
a=np.arange(3).reshape(3,1)
pprint(a)
pprint(a+[1,2,3]) # 행 벡터와 열벡터 간의 계산 shape 바뀌는거 주의

print("=======")
a = np.arange(16).reshape(4,4)
pprint(a)
#값 할당
a[1:-1,1:-1] = [99,101]
pprint(a)
print("=======")
# 행 별로 (옆으로) 정규화 예제
np.random.seed(1)
a = np.random.randint(5,size=(3,5))
pprint(a)
a_mean = a.mean(axis=1).reshape(3,1)
pprint(a_mean)
print("=======")
a_std = a.std(axis=1).reshape(3,1)
pprint(a_std)
pprint((a-a_mean)/a_std)

'''기타기능'''
print("=======")
a= np.arange(10).reshape(2,5)
pprint(a)
pprint(a.ravel())# a.ravel() 펼처준다. 즉 1행 어레이로
a=np.linspace(0,1,10) # 시작값 끝값 나누는 갯수
pprint(a)
a=np.linspace(0,1,11) # 11열
pprint(a)


'''numpy 심화학습!!'''
# 나중에

