import numpy as np
import math
# 4x4 행렬 4row  4 col
#   4x4 2x4

'''
1 
2
3
4
 크기
 010100001110 저장이 
 1 크기 
 2 크기
 int 8 16 32 64 
 
'''
# dtype  자료형 확인하는 함수
'''
int   부호가 있는 정수 
uint  부호가 없는 정수 
float  실수
boolean True False
complex 복소수
string 문자열 
object 

'''




# t = math.inf
ls = [
    [1, 2, 3, 4],  #0  # ls[1][1] = 6
    [5, 6, 7, 8],  #1
    [9, 10,11,12], #2
    [13,14,15,16]  #3
]   # 0 1 2 3
# ls2 = [ # row 1 col 1 기준
#     [1,1],
#     [1,1],
#     [1,1],
#     [1,1]
#
# 둘중 1요소만 가지고 있어야 한다.
bbb = np.array(ls)
g = bbb[[1,1],[2,2],[3,1],[2,2]]
# ccc=np.array(ls2)
#       row  col         012345   0.511111 6789
# 행 1 ,2     열  모든 열
#  2행 3행    0열 1열

# 2,3    15 ,16 목표 설정
#  2 3
a = bbb[  : 1 , 1 : 3] # 2 3
b = bbb[ 3 :  , 2 :  ] # 15 16
# a = bbb[ :1, :3]
c = np.concatenate((a,b),axis= 1)
d = bbb[ : 2 ,: 1]
e = bbb[1:3 , 3 :]
# f = np.concatenate((d,e), axis= 1)
print(a)
print(b)
print(c)
print(g)
# print(f)
# print(bb b+ccc)
# print(ccc+bbb)
#[[1 2]
#  [4 5]]
'''  기준
 0.51     1    1.5
 1.51     2    2.5
 2.5     3    3.49
          4
'''# round 0.5 를 버린다.  0.51 반올림
                        # 0.5 내림

#
# c= np.round(3.5)
# print(c)

#
# print(np.isinf(bbb))
# print('fsf76')
# 똑같은 크기면 각 요소마다 마스크 씌워진다는거 아야
# names = np.array(['Beomwoo','Beomwoo','Kim', 'Jone', 'Lee','Beomwoo','Park','Beomwoo'])
# names = np.array(['Beomwoo','Beomwoo','Kim', 'Jo'])
# print(names)    #    0          1       2       3       4
# print(names.shape)
# # 마스크 대한 col or data 크기  ==  data의 row
# data3 = np.random.randn(8,4)
# print(data3)
# print(data3.shape)
#
# #요소가 범우인 항목에 대한 마스크 생성
#
# names_Beonmwoo_mask = (names == 'Beomwoo')
# print(names_Beonmwoo_mask) # [ True  True False False False  True False  True]
# print('fsf')
# # data3에서 0,1,5,7, 행의 요소를 꺼내 와야 한다.
# #이를 위해 요소가 범우인 것에 대한 boolean 값을 가지는 mask를 만들었고
# # 마스크룰 인덱싱에 응용하여 data의 0157행 을 꺼냈다.
# print(data3[names_Beonmwoo_mask,:])
# print('fsf')
# # 요소가 Kim 인 행의 데이터만 내기
# print(data3[names == 'Kim',:])
# # 논리연산을 응용하여 요소가 kim 또는 park인 행의 데이터만 꺼내기
# print('fsf')
# print(data3[(names == 'Kim') | (names == 'Park'),:])
# print('fsf')
#
# # data array 자체적으로도 마스크를 만들고, 이를 응용하여 인덱싱이 가능하다.
# # data array에서 0번쨰 열의 값이 0보다 작은 행을 구해보자
#
# # 먼저 마스크 만들기
# # data array에서 0 번째 열이 0보다 작은 요소의 boolean 값은 다음과 같다.