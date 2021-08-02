import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import *
from mpl_toolkits.mplot3d import Axes3D

'''직선 그래프 '''
# X= [1,2,3,4,5,6,7]#x축
# X= ["Mon","Tue","Wed","Thur","Fri","Sat","Sun"]#x축
# Y1 = [15.6,14.2,16.3,18.2,17.1,20.2,22.4]
# Y2 = [20.1,23.1,23.8,25.9,23.4,25.1,26.3]
# plt.plot(X,Y1,label="Seoul")
# plt.plot(X,Y2,label="Busan")
# plt.xlabel("day")
# plt.xlabel("temperature")
# plt.legend(loc="upper left")
# plt.title("Temperatures of Cities")
# plt.show()
'''점 그래프'''
# plt.plot([15.6,14.2,16.3,18.2,17.1,20.2,22.4],"sm")
# plt.show()
'''막대 그래프'''
# X = ["Mon","Tue","Wed","Thur","Fri","Sat","Sun"]
# Y = [15.6,14.2,16.3,18.2,17.1,20.2,22.4]
# plt.bar(X,Y) ## .bar 막대그래프
# plt.show()

'''3차원 그래프'''
# 3차원 축 (axis)을 얻는다.
# axis = plt.axes(projection='3d')
# Z=np.linspace(0,1,100)
# X=Z*np.sin(30*Z)
# Y=Z*np.cos(30*Z)
# axis.plot3D(X,Y,Z)
# plt.show()

'''넘파이 배열'''
ftemp = [63,73,80,86,84,78,66,54,45,63]
F=np.array(ftemp)
# pprint(F)
'''화씨 섭씨'''
# 브로드 캐스팅
# pprint((F-32)*5/9) # 배열의 모든 요소에 이 연산이 적용된단
# plt.plot(F)
# plt.show()

'''배열 간 연산'''
A = np.array([1,2,3,4])
B = np.array([5,6,7,8])
result = A+B
pprint(A+B)
'''모든연산자 가능 '''
a = np.array([0,9,21,3])
pprint(a<10)

'''2차원 배열'''
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
pprint(b)
pprint(b[0][2])

'''lab BMI 계산하기'''
heights = [ 1.83, 1.76, 1.69, 1.86, 1.77, 1.73 ]
weights = [ 86, 74, 59, 95, 80, 68 ]
np_heights = np.array(heights)
np_weights = np.array(weights)

bmi = np_weights/(np_heights**2)
pprint(bmi)

'''넘파이 데이터 생성함수'''
# A=np.arange(1,10,2)
# A= np.linspace(0,10,100)
# pprint(A)
#
# plt.plot(A)
# plt.show()

'''균일 분포 난수 생성'''
np.random.seed(100)
# 시드가 설정되면 다음과 같은 문장을 수행 5개의 난수를 얻는다.
#난수는 0.0 에서 1.0사이의 값으로 생성된다,
np.random.rand(5)
pprint(np.random.rand(5))
print("=======")
pprint(np.random.rand(5,3))
print("=======")
pprint(np.random.randn(5))
print("=======")
pprint(np.random.randn(5,4))
print("=======")

# 위의 정규 분포는 평균값이 0이고 표준편차가 1.0이다.
# 만약 평균값과 표준편차를 다르게 할려면 다음과 같이 하면된다.
'''정규 분포 난수 생성'''

m , sigma = 10,2
pprint(m+sigma*np.random.randn(5))
print("=======")

mu , sigma = 0, 0.1 # 평균과 표준편차
C=np.random.normal(mu,sigma,5)
pprint(C) # 그래프 따로

''' 잡음이 들어간 직선 그리기'''
pure = np.linspace(1,10,100) # 1~10 까지 100데이터 생성
noise = np.random.normal(0,1,100)# 평균 0 표준편차 1인 100개 난수

signal =pure + noise
# plt.plot(signal)
# plt.show()

'''넘파이 내장 함수'''
# A = np.array([0,1,2,3])
# pprint(10*np.sin(A))
print("===========")
scores = np.array([[99,93,60],[98,82,93],[93,65,81],[78,82,81]])
pprint(scores.sum())
print("===========")
pprint(scores.min())

print("===========")
pprint(scores.max())
print("===========")
pprint(scores.mean())
print("===========")
print(scores.std())
print("===========")
pprint(scores.var())
print("===========")
pprint(scores.mean(axis=0))

'''히스토그램'''

# numbers=np.random.normal(size=10000)
# plt.hist(numbers)
# plt.xlabel('value')
# plt.ylabel('freq')
# plt.show()

'''정규 분포 그래프 2개 그리기'''
# m,sigma=10,2
# Y1=np.random.randn(10000)
# Y2=m+sigma*np.random.randn(10000)
#
# plt.figure(figsize=(10,6))
# plt.hist(Y1,bins=20)
# plt.hist(Y2,bins=20)
# plt.show()

'''싸인 함수 그리기'''
# -2파이에서 +2파이까지 100개의 데이터를 균일하게 생성한다.
# X=np.linspace(-2*np.pi,2*np.pi,100)
#
# # 넘파이 배열에 sin()함수 를 적용한다.
# Y1=np.sin(X)
# Y2=3*np.sin(X)
# plt.plot(X,Y1,X,Y2)
# plt.show()

'''MSE 오차 계산하기'''
print("=======")
#회귀 문제나 분류문제에서 실제 출력과 우리가 원하는 출력간의 오차를
#계산하기 위하여 MSE를 많이 계산한다.
# ypred =np.array([1,0,0,0,0])
# y = np.array([0,1,0,0,0])
# n=5
# MSE= (1/n)*np.sum(np.square(ypred-y))
# print(MSE)

'''인덱싱과 슬라이싱   '''
grades = np.array([88,72,93,94])
pprint(grades[1:3])
pprint(grades[:2])

'''논리적인 인덱싱'''
print("=======")
ages = np.array([18,19,25,30,28])
y= ages>20
pprint(y)
#논리저인 인덱심
pprint(ages[ages>20])

'''2차원 배열의 슬라이싱'''
print("=======")
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
pprint(a[0:2,1:3])
pprint(a>5)
pprint(a[a>5])

''' 직원들의 월급 인상하기'''
print("=======")
salary = np.array([220,250,230])
salary = salary *2
pprint(salary)
pprint(salary>450)

'''그래프 그리기'''
print("=======")
# X=np.arange(0,10)
# Y1=np.ones(10)
# Y2=X
# Y3=X**2
# plt.plot(X,Y1,X,Y2,X,Y3)
# plt.show()
'''전치행렬'''
print("=======")
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
pprint(x.transpose())
pprint(x.T)

'''역행렬 계산하기'''
print("=======")
x=np.array([[1,2],[3,4]])
y=np.linalg.inv(x)
pprint(y)
pprint(np.dot(x,y)) # 내적 계산

'''역행렬 계산하기'''
print("=======")
a = np.array([[3,1],[1,2]])
b = np.array([9,8])
x=np.linalg.solve(a,b)
pprint(x)