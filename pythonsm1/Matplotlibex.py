import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import *
'''
 line plot 그리기

  
'''
# 사용할 데이터를 생성한다.
#                        cumsum 누적 합계 자료 좀 더 보기
# s = pd.Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))

#



# print(s)
# s.plot()
# plt.show()
'''
line plot : 독립변수 x가 변함에 따라 종족변수 y 가 어떻게 변화하는지를 나타낸다.
특히 line plot 은 연속적인 x 의 변화에 따른 y 의 변화는 살펴보는 데 효과적

series s 의 인덱스가 x축 / 각 성분이 y축을 구성한다.
라인플롯을 부분별로 살펴보자.
x축 인덱스 / y축 성분 
그래프 우측 상단의 파란색 전원버튼을 누르기 전까지 
우리는 해당 그래프를 interactive 하게 조작할 수 있다.
그리고 다음 플롯을 생성할 때까지는 꺼야한다.
4번째 사각형을 클릭 한 뒤, plot에 범위를 지정하면 확대할 수 있다.
디스켓을 누르면 그림 형태로 저장할 수 있다.

이제 DataFrame을 만들고 , 라인 플롯을 생성해보자/
이때 10x4 array에 cumsum(axis=0)을 주면 열=행방향 (아래)으로 누적합계
Series일 때는 축을 안정해 주어도 된다. 

'''

print("=======30======")
# DataFrame 을 통한 line plot 그리기
df = pd.DataFrame(np.random.randn(10, 4).cumsum(axis=0),
                  columns=["A", "B", "C", "D"],
                  index= np.arange(0, 100, 10))
print(df)
                 #컬럼수=라인 플롯 수
                 # 만약 여러개의 열을 가진 df에서 특정열에 대한 1개의 라인플롯을 그리고싶다면,
                 #원하는 열을 Series의 형태로 뽑아지는 열 인덱싱 하여 , plot()을 호출하면 된다.
print("=======31======")
# df.plot()
# plt.show()
print("=====6=====")
# df['B'].plot()
# plt.show()

''' bar plot 그리기'''
#라인플롯은 연속적인 x 에 따른 y의 변화 볼때 굿
# 바 플롯은 연속적이지 않은 x 에 따른 변화

# 먼저 데이터를 만드는데 , 시지르의 링덱스에 들어가 list("연속된 문자열")함수는 연속되어
# 붙어있는 문자열을 하나씩 분리해서 리스트로 만들어 준다. 그 list 성분등이 index로 들어가 a,b,c,d,...p까지 되는것이다.
# Series에 들어가는 1차원 array가 16개 이므로 , 문자열에도 연속해서 16개를 주었다.
print("=====33=====")
s2 = pd.Series(np.random.rand(16), index=list("abcdefghijklmnop"))
pprint(s2)
print("=====39=====")
# s2.plot(kind='bar') # bar 수평으로 바꿀려면 barh 하고 그래프 비교
# plt.show()
print("=====40=====")
# s2.plot(kind='barh')
# plt.show()

print("=====41=====")
df2 = pd.DataFrame(np.random.rand(6,4),
                   index=['one','two','three','four','five','six'],
                   columns=pd.Index(["A","B","C","D"],name="Genus"))
pprint(df2)
print("=====11=====")
# df2.plot(kind="bar")
# plt.show()
# DataFrame 의 수평바 플롯 (kind='barh")를 추가로 인자에 stacked=True로 주게 되면,
# 인덱스 (x축)에 대한 바 플롯이 여러개 나타나는 것이 아니라
# 한 인덱스에 모든 바 플롯이 한 줄로 나타나게 되어 -> 하나의 index애 대한 각 플롯들(열들)의 성분 비율]을
#확인할 수 있을 것이다.
print("=====42=====")
# df2.plot(kind="barh",stacked=True)
# plt.show()

'''히스토그램 그리기'''
#하나의 변수 x만을 가지고 그릴 수 있다.
#히스토그램 그릴 경우 index가 따로 필요하지않음  Series로 구성
# 히스토그램의 x 축에는 bin이라는 하는 구간이 y축에는 그 구간 (bin)에 해당하는 x 의 갯수가 찍힌다,
#인덱스 없는 시리즈를 만들 떄 numpy에서 제공하는 정규분포를 의미하는 normal함수를 사용
# 이 normal함수는 평균과 표준편차를 지정 거기서 추출한 샘픙 갯수를 지정해서 뽑아낼 수 있다.
#만약 2차원으로 만들고 싶다면 size=(200,1)

print("=====43=====")
# 히스토그램 인덱스 필요 x
s3 = pd.Series(np.random.normal(0,1,size=200))
# pprint(s3)

print("=====156=====")
#기본적으로 10칸의 bin을 히스토그램이 차지하고 있다.
# 그래프를 보면 -0.5bin에 해당하는 값의 갯수는 50개가 약간 안되는 것을 알 수 있다.
# bin의 갯수를 s3.hist(bins=50)처럼 bins= 구간의 갯수로 직접 지정해줄수있다.
# bins= 갯수가 많아지면 그만큼 x가 들어가는 구간도 좁아지면서 더 세밀하게 관찰 할 수 있더ㅏ.,

# s3.hist()
# plt.show()
print("=====256=====")

# s3.hist(bins=50) # normaled=True 넣으면 에러뜸
# plt.show()
print("=====257=====")
# 또 bins의 갯수뿐만 아니라, normal인자 = True로 주면 이전까지 구간에 속한 x의 개수였지만,
#<각 bin에 속하는 갯수를 전체갯수로 나눈 비율, 즉 정규화된 값>을 bar의 높이로 사용하게된다.
# 애초에 200개의 샘플을 정규분포에 추출하였기 때문에.
#정규화된 결과는 정규분포를 의미하는 종모양과 유사하게 나타난다.
# s3.hist(bins=100)
# plt.show()


'''산점도(scatter plot) 그리기'''
'''
산점도는 서로다른 2개의 독립변수 x1,x2의 관계를 알아보기 위해 사용
2차원 평면상 점 형태로 나타내는 것
데이터로 np에서 제공하는 지정된 평균/표준편차 의 정규분포에서 가가 100개를 2차원 형태인
100x1로 추출, x1,x2의 array를 만들고 두 독립 변수 (100x1,100x1)을 np에서 제공하는 
concatenate()함수를 두 독립변수를 열방향(axis=1)으로 붙혀서 연결한다.
(100x1,100x1)=100x2 이때 각 array를 만들 때 series=1차원이 size100이 아니라 (100,1)의
2차원 array로 만들어서-> conctenate()로 연결할 수 있게 된다.
'''
print("=====54=====")
x1=np.random.normal(1,1,size=(100,1))
x2=np.random.normal(-2,4,size=(100,1))
X=np.concatenate((x1,x2),axis=1) #concatenate() 함수 두 어레이를 합치는 붙이는?
pprint(X)


print("=====128=====")
#이제 , 각 정규분포에서 뽑아낸 100x1 array 2개를 concatenate()로 붙힌
#100x2 array를 DataFrame으로 만들고, 각 컬럼명을 x1, x2로 준다.
df3 = pd.DataFrame(X,columns=["x1","y1"])
pprint(df3)
print("=====129=====")
#df3라는 DataFrame의 두 컬럼x1과 y1의 시각적인 상관관계(산점도)를 알아보기 위해
# pyplot에서 제공하는 plt.scatter()함수를 호출해야 한다.
# 이때 인자로 DataFrame의 각 열을 인덱싱 해서 넣어주면 된다.

# plt.scatter(df3['x1'],df3['y1']) # x1이 x축 , y1이 y축
# plt.show()

#산점도 상의 각 점들을 DataFrame의 각 행에 있는 x1,y1 성분의 상관관계를 의미한다.
# 즉,  x축이 x1 , y축이 y1의 값이다.

'''plot모양 변형하기'''
# figure -> subplot으로 세분화해서 그릴 준비 하기
# pyplot에서 제공하는 .figure()함수를 호출해서 내부가 텅빈 figure를 만들 수 있다.
print("=====7=====")
fig=plt.figure()
# plt.show()
# 이제 생성한 figure fig에 subplot을 추가해야한다.
#add_subplot(,,)을 호출하여 좌표평면을 나타내는 변수 axes에 인자가 들어간다.
# 앞에 2개인자는 fig 내에 몇행x몇열로 subplot을 만들것인지 지정한다.
# 마지막인자는, 행열의 몇번째 성분에 그릴 것인지를 지정하는 것이다.
# 2x2라면, 1,2/3,4 시작위치를 지정할 수 있다.
# ax1=fig.add_subplot(2,2,1)라는 명령어를 입력하면,
# 만들었던 figure의 2x2중 1번째 위치에서 빈 좌표평면을 (ax1)이 그려지는 것을 알 수 있다.

print("=====8=====")
# subplot 추가하기, add_subplot에는 총 3개 인자 들어감

ax1 = fig.add_subplot(2,2,1)

#첫번째 두번쨰 우리가 figure를 어떤 크기로 나눌지 대한 값
#세번째 figure에서 좌측상단으로 우측 방향으로 숫자가 붙는다. 이떄 우리가 add하고자하는
#즉
# 1 2
# 3 4
# 이떄 우리는 1위치에 subplot을 추가하고 해당 subplot을 ax1이라는 변수로 반환 받는다.
print("=====9=====")
pprint(ax1)

print("=====10=====")
ax2 = fig.add_subplot(2,2,2)
pprint(ax2)
print("=====11=====")
ax3 = fig.add_subplot(2,2,3)
pprint(ax3)

print("=====13=====")
# plt.figure() 생성한 fig에 .add_subplot() 을 통해 생성한 최펴평면? ax1,ax2ax3가 있
#pyplot 제공하는 plt.plot()함수의 인자에 1차원 랜덤 데이터 50개의 누적합 데이터를 주고
#좌표평면을 지정하지 않고 그린다면, 그 전인 2번쨰 좌표평명 (ax2)에
#그리고  그 다음에는 1번째 좌표평면(ax1)에 그려지게 된다.
#즉 좌표평면을 지정하지 않고 figure의 subplot에 그림을 그리면 ,
#마지막 부터 역순으로 plot에 채워진다.
# plt.plot(np.random.randn(50).cumsum())
# 위치를 지정하지 않고 plot하면 맨 마지막에 그림그려짐
#맨 마지막에 위치한 곳에 그려지는 것이 아니라 제일 마지막에 추가한 subplot에 그려짐
#2->3->1 순으로 subplot을 추가하여 테스트 해보면 1번 요소에 그려진다.

# plt.show()
print("=====14=====")
# plt.plot(np.random.randn(200).cumsum())
# plt.show()
# 한번 더 위치지정 없이 그리면 그 전에 요소에 그려진다고 했는데,
# 실제로 진행해보면 그냥 위의 것과 똑같이 제일 마지막에 추가한subplot에 중복되서 그려짐
print("=====15=====")
'''
axes를 지정해줘서 plot을 그려보자
첫번째로 ax1에다가 히스토그램(x값만 있으면 그려지는)을 한번 그려보자
밑에내용
아래 첫번째 좌표평면에 그려지게 된다.
(fig-subplot-ax가 없었을땐, Series.hist(bins=n,normed=True))
'''

#그럼 우리가 원하는 위치에 그림을 그리기 위해성는?
#위에서 add_subplot 을 할 때 변수명을 지정하여 반환값을 받았다.
# 해당 변수를 통해 plot을 그리면 된다.
ax1.hist(np.random.randn(100),bins= 20)
pprint(ax1.hist(np.random.randn(100),bins= 20)) # bins는 x축 bar의 개수 #랜덤정수 100개가 속하는 구간을 20개 구간으로 나타낸다


print("=====16=====")
'''
ax2 좌표평면에다가도 산점도를 그려보자 (산점도는 독립변수 2개 필요)
비교하는 독립변수에는 0부터 30전까지 1차원 array와 동일 array+3*30ro 랜덤 데이터를 대입하자
# (fig-subplot-ax가 없었을땐, plt.scatter(열인덱싱,열인덱싱))????
# '''
ax2.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30))
pprint(ax2.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30)))

# 특정 axes를 나타내는 변수 plot() .hist() .scatter()함수를 사용해서
# fig > subplot > axes 좌표평면에다가 그릴 수 있었다.

'''figure와 subplot을 좀 더 직관적으로 그리기'''
#pyplot에서 제공하는 plt.subplots(m,n) or (m)을 통해서 fig와 axes를 동시에 반환 받을수 있다.
#다시 해보자
fig, axes = plt.subplots(2,3)
# ax1.plot()
# plt.plot(ax1)
# plt.plot(ax2)
# plt.show()

# pprint(axes)
#위와 같이 만들면 2x3 subplot들을 가지는 figure를 만드는 것
# 이때 반환되는 값은 2개 로써 figure 자체와 축



print("=====19=====")
plt.plot(np.random.randn(30),color = 'g',marker='o',linestyle='--')
print("=====19=====")
plt.plot(np.random.randn(30),'k.-')
# plt.show()

fig,axes = plt.subplots(2,1)
# plt.show()
print("=====22=====")
data = pd.Series(np.random.rand(16),index=list('abcdefghijklmnop'))
print("=====23=====")
data.plot(kind='bar', ax=axes[0],color='k',alpha=0.7)
# plot함수를 그릴떄, figure 에서 원하는 위치를 지정하기위해 ax속성을
print("=====24=====")
data.plot(kind='barh', ax=axes[1],color='g',alpha=0.3)
# plt.show()

'''우리가 만들 plot의 눈금, 레이블, 범례 등을 지정 및 수정할 수 있다.'''
'''
눈금, 레이블, 범례를 조작하기 위해서는 먼저 figure를 만들어야 한다.
fig = plt.figure()
만든 fig에 subplot을 만들어 준다. 간단하게 1x1서브 플롯의 1번째 axes를 지정해주자.
ax=fig.add_subplot(1,1,1)
'''
print("=====340=====")
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum())
# 이떄 그래프에서 나타내는 눈금을 tick
# 즉 위의 그래프의 x tick은 200 , y tick은 10이다.
ax.set_xticks([0,250,500,750,1000])
# 눈금 문자로 하기 위해서
labels = ax.set_xticklabels(['one','two','three','four','five'],rotation=30,fontsize='small')
ax.set_title('random walk plot')
ax.set_xlabel('Stages')
ax.set_ylabel('Values')

print("=====357=====")
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum(),'k',label='one')
ax.plot(np.random.randn(1000).cumsum(),'b--',label='two')
ax.plot(np.random.randn(1000).cumsum(),'r.',label='three')
ax.legend(loc='best') # loc는 범례 위치할 곳
ax.set_xlim([100,900])
ax.set_ylim([-100,100])
plt.show()

''' Matplotlib을 이용한 데이터 시각화 맛보기 는 csv파일 필요 이후에 '''