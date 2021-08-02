'''서울시 구별 CCTV 현황 분석하기'''
'''
서울시 각 구별 CCTV수를 파악하고, 인구대비 CCTV 비율을 파악해서 순위 비교
인구대비 CCTV 의 평균치를 확인하고 그로부터 CCTV가 과하게 부족한 구를 확인


'''

import  pandas as pd
import numpy as np
import os
from pprint import *
import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
os.getcwd()
print("======7======")
CCTV_Seoul = pd.read_csv('./data/01. CCTV_in_Seoul.csv', encoding='utf-8')
print(CCTV_Seoul.head())

print("======8======")
print(CCTV_Seoul.columns)
# Index(['기관명', '소계', '2013년도 이전', '2014년', '2015년', '2016년'], dtype='object')
# 컬럼의 이름이 반환
print("======9======")
print(CCTV_Seoul.columns[0]) # 1행 이 나타내는 것이 무엇인가

print("======10======")
CCTV_Seoul.rename(columns={CCTV_Seoul.columns[0] : '구별'}, inplace=True)
# 이름 변경
print(CCTV_Seoul.head()) # 5행만 보여주시오

print("======13======")
# pop_Seoul = pd.read_excel('./data/01. population_in_Seoul.xls',encoding='utf-8')
pop_Seoul = pd.read_excel('data/01. population_in_Seoul.xls')
# print(pop_Seoul.head())

print("======16======")
# pop_Seoul = pd.read_excel('data/01. population_in_Seoul.xls',header= 2,usecols= 'B, D, G, J, N',encoding='utf-8')
pop_Seoul = pd.read_excel('data/01. population_in_Seoul.xls',header= 2,usecols= 'B, D, G, J, N')
#                                                             #  2행부터    사용하는 컬럼 2 ,4, 7, ...
# print(pop_Seoul.head())
print("======17======")
pop_Seoul.rename(columns = { pop_Seoul.columns[0] : '구별',
                            pop_Seoul.columns[1] : '인구수',
                            pop_Seoul.columns[2] : '한국인',
                            pop_Seoul.columns[3] : '외국인',
                            pop_Seoul.columns[4] : '고령자'},inplace = True )
print(pop_Seoul.head())

print("======19======")
s = pd.Series([1,3,5,np.nan,6.8])
pprint(s)

print("======20======")
dates = pd.date_range('20130101', periods=6)
pprint(dates)
print("======21======")
## 얘는 이해 안감
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A','B','C','D'])
pprint(df)
print("======23======")
pprint(df.head(3))
print("======24======")
pprint(df.index)

print("======25======")
pprint(df.columns)
print("======26======")
pprint(df.values)
print("======27======")
pprint(df.info())
print("======28======")
pprint(df.describe())

print("======29======")
pprint(df.sort_values(by='B',ascending=False)) # B행 기준으로  ascending=False 내림차순 True 오름차순

print("======30======")
pprint(df)

print("======31======")
pprint(df['A'])
pprint(df['B'])
print("======32======")
pprint(df[0:3])
print("======33======")
pprint(df['20130102':'20130104'])
print("======34======")

pprint(df.loc[dates[0]])

print("======35======")
pprint(df.loc[:,['A','B']])

print("======36======")
pprint(df.loc['20130102':'20130104',['A','B']])

print("======37======")
pprint(df.loc['20130102',['A','B']])

print("======38======")
pprint(df.loc[dates[0],'A'])

print("======39======")
pprint(df.iloc[3]) # 4행 내용
print("======40======")
pprint(df.iloc[3:5,0:2])
print("======41======")
pprint(df.iloc[[1,2,4],[0,2]])
print("======42======")
pprint(df.iloc[1:3,:])
print("======43======")
pprint(df.iloc[:,1:3])

print("======44======")

pprint(df)
print("======45======")
pprint(df[df.A > 0])
print("======46======")
pprint(df[df > 0]) # 0미만 value 들 nan으로
print("======47======")
df2 = df.copy()
print("======48======")
df2['E']=['one','one','two','three','four','three']
pprint(df2)
print("======49======")
pprint(df2['E'].isin(['two','four']))   #지정한 행 또는 열에서 입력한 값이 있는지 확인하기
print("======50======")
pprint(df2[df2['E'].isin(['two','four'])])
print("======51======")
pprint(df)
print("======52======")
pprint(df.apply(np.cumsum)) #
print("======53======")
pprint(df.apply(lambda  x : x.max() - x.min()))

print("======54======")
'''CCTV 데이터 파악하기 '''
pprint(CCTV_Seoul.head())

print("======55======")
pprint(CCTV_Seoul.sort_values(by='소계',ascending=True).head(5))

print("======56======")
pprint(CCTV_Seoul.sort_values(by='소계',ascending=False).head(5))

print("======57======")
CCTV_Seoul['최근증가율'] = ( CCTV_Seoul['2016년']+CCTV_Seoul['2015년']+CCTV_Seoul['2014년'])/CCTV_Seoul['2013년도 이전']*100
pprint(CCTV_Seoul.sort_values(by='최근증가율',ascending=False).head(5))

print("======58======")
pprint(pop_Seoul.head())
print("======59======")
pop_Seoul.drop([0], inplace=True)
pprint(pop_Seoul.head())
print("======61======")
pprint(pop_Seoul['구별'].unique())
print("======62======")
# pprint(pop_Seoul['구별'].isnull())
pprint(pop_Seoul[pop_Seoul['구별'].isnull()])
print("======63======")
pop_Seoul['외국인비율'] = pop_Seoul['외국인']/pop_Seoul['인구수']*100
pop_Seoul['고령자비율'] = pop_Seoul['고령자']/pop_Seoul['인구수']*100
pprint(pop_Seoul.head())
print("======64======")
pprint(pop_Seoul.sort_values(by='인구수',ascending=False).head(5))

print("======65======")
pprint(pop_Seoul.sort_values(by='외국인',ascending=False).head(5))

print("======66======")
pprint(pop_Seoul.sort_values(by='외국인비율',ascending=False).head(5))
print("======67======")
pprint(pop_Seoul.sort_values(by='고령자',ascending=False).head(5))
print("======68======")
pprint(pop_Seoul.sort_values(by='고령자비율',ascending=False).head(5))

#
# '''pandas 고급 두 데이터 프레임 병합하기'''
#
# print("======69======")
#
# df1 = pd.DataFrame({'A':['A0','A1','A2','A3'],
#                     'B':['B0','B1','B2','B3'],
#                     'C':['C0','C1','C2','C3'],
#                     'D':['D0','D1','D2','D4']},index=[0,1,2,3])
# df2 = pd.DataFrame({'A':['A4','A5','A6','A7'],
#                     'B':['B4','B5','B6','B7'],
#                     'C':['C4','C5','C6','C7'],
#                     'D':['D4','D5','D6','D7']},index=[4,5,6,7])
# df3 = pd.DataFrame({'A':['A8','A9','A10','A11'],
#                     'B':['B8','B9','B10','B11'],
#                     'C':['C8','C9','C10','C11'],
#                     'D':['D8','D9','D10','D11']},index=[8,9,10,11])
# # DataFrame 은 설정하는거
# print("======70======")
# pprint(df1)
# print("======71======")
# pprint(df2)
# print("======72======")
# pprint(df3)
# print("======73======")
# result = pd.concat([df1,df2,df3]) # concat 핲쳐주는 순서대로 인덱스는 자기가 설정
# pprint(result)
# print("======74======")
# result = pd.concat([df1,df2,df3],keys=['x','y','z']) #키값 따로
# pprint(result)
# print("======75======")
# pprint(result.index)
# print("======76======")
# pprint(result.index.get_level_values(0))
# print("======77======")
# pprint(result.index.get_level_values(1))
# print("======78======")
# pprint(result)
# print("======79======")
# df4 = pd.DataFrame({'B':['B2','B3','B6','B7'],
#                     'D':['D2','D3','D6','D7'],
#                     'F':['F2','F3','F6','F7']},index=[2,3,6,7])
# result = pd.concat([df1, df4], axis=1)
# print("======80======")
# pprint(df1)
# print("======81======")
# pprint(df4)
# print("======82======")
# pprint(result) # 빈자리 nan
# print("======83======")
# result = pd.concat([df1, df4], axis=1, join='inner') #  join='inner' nan값 제외하고 반환
# pprint(result)
print("======84======")
# result = pd.concat([df1, df4], axis=1, join_axes=[df1.index]) #
# pprint(result)
# #TypeError: concat() got an unexpected keyword argument 'join_axes'
print("======85======")
# result = pd.concat([df1, df4], ignore_index=True)
# pprint(result)

print("======86======")
# left = pd.DataFrame({#'key':['KO','K4','K2','K3'],
#                      'A':['AO','A1','A2','A3'],
#                      'B':['BO','B1','B2','B3'],
#                      'key':['KO','K4','K2','K3']})
# right = pd.DataFrame({'key':['KO','K1','K2','K3'],
#                      'C':['CO','C1','C2','C3'],
#                      'D':['DO','D1','D2','D3']
#                      })#key':['KO','K1','K2','K3']
# print("======87======")
# pprint(left)
# print("======81======")
# pprint(right)
# print("======82======")
# pprint(pd.merge(left, right ,on='key')) # key 같은 값끼리
# print("======83======")
# pprint(pd.merge(left,right,how='left',on='key')) # key 값의 어떤 기준 left! key
# print("======84======")
# pprint(pd.merge(left,right,how='right',on='key'))
# print("======85======")
# pprint(pd.merge(left,right,how='outer',on='key')) # 모두다.
# print("======86======")
# pprint(pd.merge(left,right,how='inner',on='key')) # 공통된 부분만
#
# '''7. CCTV 데이터와 인구 데이터 합치고 분석하기'''
# print("======87======")
data_result = pd.merge(CCTV_Seoul, pop_Seoul, on= '구별')
pprint(data_result.head())
print("======88======")
del data_result['2013년도 이전']
del data_result['2014년']
del data_result['2015년']
del data_result['2016년']
pprint(data_result.head())
#
print("======89======")
data_result.set_index('구별', inplace=True) # .set_index('구별', inplace=True)면 index 제거
# data_result.set_index('구별', inplace=False) # False 면 나오게 기본값
pprint(data_result.head())
print("======90======")
pprint(np.corrcoef(data_result['고령자비율'],data_result['소계']))
#np.corrcoef() 피어슨 상관계수 값을 계산해준다.
print("======91======")
pprint(np.corrcoef(data_result['외국인비율'],data_result['소계']))
print("======92======")
pprint(np.corrcoef(data_result['인구수'],data_result['소계']))
print("======93======")
pprint(data_result.sort_values(by='소계',ascending=False).head(5))
print("======94======")
pprint(data_result.sort_values(by='인구수',ascending=False).head(5))

'''그래프 그리기 기초 - matplotlib'''
print("======96======")
# plt.figure()
# plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0])
# plt.show()
print("======97======")
t = np.arange(0, 12, 0.01)
y = np.sin(t)
# plt.figure(figsize=(10,6))
# plt.plot(t,y)
# plt.show()

print("======99======")
# plt.figure(figsize=(10,6))
# plt.plot(t,y)
# plt.grid()
# plt.show()
print("======100======")
# plt.figure(figsize=(10,6))
# plt.plot(t,y)
# plt.grid()
# plt.xlabel('time')
# plt.xlabel('Amplitude')
# plt.title('Example of sinewave')
# plt.show()
# print("======101======")
# plt.figure(figsize=(10,6))
# plt.plot(t, np.sin(t),label='sin')
# plt.plot(t, np.cos(t),label='cos')
# plt.grid()
# plt.xlabel('time')
# plt.xlabel('Amplitude')
# plt.title('Example of sinewave')
# plt.show()

print("======106======")
# plt.figure(figsize=(10,6))
# plt.plot(t, np.sin(t),lw=3,label='sin')
# plt.plot(t, np.cos(t),'r',label='cos')
# plt.grid()
# plt.xlabel('time')
# plt.xlabel('Amplitude')
# plt.title('Example of sinewave')
# plt.ylim(-1.2, 1.2)
# plt.xlim(0, np.pi)
# plt.show()

print("======106======")
# t = np.arange(0, 5, 0.5)
# plt.figure(figsize=(10,6))
# plt.plot(t,t, 'r--')
# plt.plot(t,t**2, 'bs')
# plt.plot(t,t**3, 'g^')
#
# plt.show()

print("======107======")
# t = np.arange(0, 5, 0.5)
# plt.figure(figsize=(10,6))
# pl1 = plt.plot(t,t**2, 'bs')
# plt.figure(figsize=(10,6))
# pl2 = plt.plot(t,t**3, 'g^')
#
# plt.show()

print("======108======")
t = [0,1,2,3,4,5,6]
y = [1,4,5,8,9,5,3]
# plt.figure(figsize=(10,6))
# plt.plot(t,y,color='green')
# plt.show()
print("======109======")
# plt.figure(figsize=(10,6))
# plt.plot(t,y,color='green', linestyle='dashed')
# plt.show()

print("======109======")
# plt.figure(figsize=(10,6))
# plt.plot(t,y,color='green', linestyle='dashed',marker='o',markerfacecolor = 'blue')
# plt.show()

print("======112======")
# plt.figure(figsize=(10,6))
# plt.plot(t,y,color='green', linestyle='dashed',marker='o',markerfacecolor = 'blue',markersize=12)
# plt.xlim([-0.5, 6.5])
# plt.ylim([0.5, 9.5])
# plt.show()

t1 = np.array([0,1,2,3,4,5,6,7,8,9])
y1 = np.array([9,8,7,9,8,3,2,4,3,4])
print("======113,114======")
# plt.figure(figsize=(10,6))
# plt.scatter(t1,y1)
# plt.show()
print("======115======")

# plt.figure(figsize=(10,6))
# plt.scatter(t1,y1,marker='>')
# plt.show()

# print("======115======")
#
# plt.figure(figsize=(10,6))
# plt.scatter(t1,y1,marker='>')
# plt.show()
print("======116,117======")
# colormap = t1 # colormap은 그 숫자마다 색을 표현해주는 듯함
# plt.figure(figsize=(10,6))
# plt.scatter(t1, y1, s=50, c = colormap, marker='>')
# plt.colorbar()
# plt.show()

print("======118,119======")
#                    loc y축 위치 그 값
s1 = np.random.normal(loc= 0 , scale=1 , size=1000) # scale 표준편차
s2 = np.random.normal(loc= 5 , scale=0.5 , size=1000)
s3 = np.random.normal(loc= 10 , scale=2 , size=1000)

# plt.figure(figsize=(10,6))
# plt.plot(s1,label = 's1')
# plt.plot(s2,label = 's2')
# plt.plot(s3,label = 's3')
# plt.legend()
# plt.show()

print("======120======")
# plt.figure(figsize=(10,6))
# plt.boxplot((s1,s2,s3)) # 각각(1,2,3,...)의 평균 위치 표준편차 위치, 그 값의 최대 최소
# plt.grid()
# plt.show()

print("======121======")
# 여기 다시 볼것
# plt.figure(figsize=(10,6))
# plt.subplot(221)
# plt.subplot(222)
# plt.subplot(212)
# plt.show()


print("======122======")
# 여기 다시 볼것
 #subplot(nrows, ncols, index, **kwargs)
 #nrows 행의 수 , ncols열의 수 , index 1부터 시작 순서는 위부터 오른쪽 위쪽에서 아래쪾방향으로 설정?
# , **kwargs
#subplot(pos, **kwargs)
#subplot(**kwargs)
# subplot(ax)
#subplot(axes)
#axes 객체를 바로 입력 받는 방법
# 기존의 axes의 정보를 받아온다.

# plt.figure(figsize=(10,6))
# plt.subplot(411)
# plt.subplot(423)
# plt.subplot(424)
# plt.subplot(413)
# plt.subplot(414)
# plt.show()

print("======123======")
# # 안뜸 제껴
# t = np.arange(0.5,0.01)
# plt.figure(figsize=(10,12))
# plt.subplot(411)
# plt.plot(t, np.sqrt(t))
# plt.grid()
#
# plt.subplot(423)
# plt.plot(t, t**2)
# plt.grid()
#
# plt.subplot(424)
# plt.plot(t, t**3)
# plt.grid()
#
# plt.subplot(413)
# plt.plot(t, np.sin(t))
# plt.grid()
#
# plt.subplot(414)
# plt.plot(t, np.cos(t))
# plt.grid()
#
# plt.show()

print("======124======")
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    rc('font',family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family = font_name)
else:
    print('Unknown system...sorry~~')
pprint(data_result.head())

print("======126======")
# 자료와 같이 색이 안뜨틑 이유는 무엇?
# plt.figure()
# data_result['소계'].plot(kind='barh',grid=True,figsize=(10,10))
# plt.show()

print("======127======")
# # plt.figure()
# data_result['소계'].sort_values().plot(kind='barh',grid=True,figsize=(10,10))
# plt.show()

print("======128======")
# data_result['CCTV비율'] = data_result['소계']/data_result['인구수'] * 100
# data_result['CCTV비율'].sort_values().plot(kind='barh',grid=True,figsize=(10,10))
# plt.show()
print("======129======")
# plt.figure(figsize=(6,6))
# plt.scatter(data_result['인구수'],data_result['소계'],s=50)
# plt.xlabel('인구수')
# plt.xlabel('CCTV')
# plt.grid()
# plt.show()
print("======130======")
# poly함수 무엇인가??
fp1 = np.polyfit(data_result['인구수'], data_result['소계'],1)
pprint(fp1)
print("======131======")
f1 = np.poly1d(fp1)
fx = np.linspace(100000,700000,100)
print("======132======")
# plt.figure(figsize=(10,10))
# plt.scatter(data_result['인구수'],data_result['소계'],s=50)
# plt.plot(fx,f1(fx), ls='dashed',lw=3,color = 'g')
# plt.xlabel('인구수')
# plt.ylabel('CCTV')
# plt.grid()
# plt.show()
print("======133======")
# data_result['오차'] = np.abs(data_result['소계']-f1(data_result['인구수']))
# df_sort = data_result.sort_values(by='오차',ascending=False)
# pprint(df_sort.head())
# print("======134======")
# plt.figure(figsize=(14,10))
# plt.scatter(data_result['인구수'],data_result['소계'],
#             c=data_result['오차'],s=50)
# plt.plot(fx,f1(fx), ls='dashed',lw=3,color='g')
#
# for n in range(10):
#     plt.text(df_sort['인구수'][n]*1.02, df_sort['소계'][n]*0.98,df_sort.index[n], fontsize=15)
#
# plt.xlabel('인구수')
# plt.ylabel('인구당비율')
# plt.colorbar()
# plt.grid()
# plt.show()