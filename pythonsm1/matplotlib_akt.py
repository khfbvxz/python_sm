import numpy as np
import matplotlib.pyplot as plt

'''맛보기 1'''
# data1 = np.random.normal(1,1,size=100) # 평균 1 표준편차 1 size 100
# data2 = np.random.normal(3,1,size=100)
# plt.plot(data1,'ro-')  # data 1 을 ro(원) -실선으로
# plt.plot(data2,'bs--') # data 2 을 bs(사각) --점선으로
# plt.title('TEST') # 타이틀
# plt.legend(['data1','data2']) # 범례
# plt.xlabel('time') #
# plt.ylabel('value')
# plt.show()

'''맛보기 2 '''

# data11=np.random.normal(1,1,size=1000)
# data22=np.random.normal(3,1,size=1000)
# plt.scatter(data11,data22) # scatter 흩어지게 하다. 산점도!
# plt.title('TEST2')
# plt.xlabel('data11')
# plt.ylabel('data22')
# plt.show()

'''맛보기 3'''
# data=[ [np.sin(i**2+j**2) for j in np.arange(-2,2,0.05)]
# for i in np.arange(-2,2,0.05) ] # 시작 -2 , 끝 2 , 간격 0.05
# plt.imshow(data)
# plt.colorbar()
# plt.show()

''' iris 데이터 '''
# iris 데이터 읽어와 그래프 그리기~

# #f = open('iris.csv')
columns=["SepalLength","SepalWidth","PetalLength","PetalWidth"]
name={'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
X=np.loadtxt('iris.csv', delimiter=',', skiprows=1, converters={4: lambda x:
name[x.strip('"')]}, encoding='utf-8')
X1=X[X[:,-1]==0]
X2=X[X[:,-1]==1]
X3=X[X[:,-1]==2]
# plot
# plt.title('IRIS – sepal.length')
# plt.plot(X1[:,0],'ro-')  # r빨강 o 원 실선
# plt.plot(X2[:,0],'gs:') # g초록 s 네모  . . .  선으로
# plt.plot(X3[:,0],'b^--') # b파랑 ^ 세모 - - - 선
# plt.legend(name.keys())
# plt.xlabel('samples')
# plt.ylabel('values')
# plt.show()

'''Matplotlib  plot()'''
# plt.plot(range(10,40), X1[10:40,0], color='red', marker='o', linestyle='solid',
# linewidth=2, markersize=20, label='Iris-setosa')
# plt.show()
# # 위와 똑같은데 색 마커 라인스타일이 단축됨
# plt.plot(range(10, 40), X1[10:40, 0], 'ro-', linewidth = 2, markersize = 20, label ='Iris-setosa')
# plt.show()
# # 그냥 띄웠을 떄
# plt.plot(X2[:,1],'gs')
# plt.show()

'''matplotlob 설정 기능들'''
# # 축, 그리드, 레이블 등
# plt.title('TEST')
# plt.plot(range(-10,10),[np.sin(i) for i in range(-10,10)], label='sin(x)')
# plt.xlabel('x')
# plt.ylabel('sin(x)')
# plt.xticks(range(-10,10,2)) # x축 ticks 간격
# plt.axis('equal') # plt.axis('equal') ?
# # axis 함수는 x,y 축 범위를 설정 할 수 있게 하는 것과 동시에 여러 옵션을을 설정할 수 있는 함수
# plt.grid() #
# plt.legend() # ? # 범례
# plt.show()

'''scatter()'''
# 2차원 점들로 표시 # 이놈 다시 봐
# iris 데이터임
# plt.scatter(X[:,0], X[:,1], c=X[:,1], s=X[:,2]*100, alpha=0.2)
# plt.show() # c: color, s: size

'''hist 히스토그램  '''
# 히스토그램 그릴떄 bin 갯수 설정 중요 적으면 너무 뭉뚱그려짐
# 많으면 이빨빠지 빗처럼 이상해짐 대개 경우 default 세팅
# bin 히스토그램 너비? alpha 0~1값 히스토그램 진하기?
# 이 값들은 해보면서 보기좋게 설정 one노트 블로그 참고
# plt.title('IRIS - petal.length')
# plt.hist(X1[:,2],bins=20,alpha=0.5,label='Setosa')
# plt.hist(X2[:,2],bins=20,alpha=0.5,label='Versicolor')
# plt.hist(X3[:,2],bins=20,alpha=0.5,label='Virginica')
# plt.xlabel('petal.length')
# plt.ylabel('count')
# plt.legend()
# plt.show()

'''서브플롯 여러개 plot을 한장에~'''
# fig=plt.figure()
# fig.suptitle('IRIS')
# for col in range(4):
#     plt.subplot(2,2,col+1) # 1 ~ 4
#     plt.hist(X1[:,col],bins=20,alpha=0.5,label='Setosa')
#     plt.hist(X2[:,col],bins=20,alpha=0.5,label='Versicolor')
#     plt.hist(X3[:,col],bins=20,alpha=0.5,label='Virginica')
#     plt.xlabel(columns[col])
#     plt.ylabel('count')
#     plt.legend()
# plt.show()

'''서브플롯 2'''
# fig,axes = plt.subplots(2,2,sharex=True,sharey=True)
# fig.suptitle('IRIS')
# for col in range(4):
#     ax=axes[col//2,col%2] # axes.shape==(2,2) # 객체
#     ax.hist(X1[:,col],bins=20,alpha=0.5,label='Setosa')
#     ax.hist(X2[:,col],bins=20,alpha=0.5,label='Versicolor')
#     ax.hist(X3[:,col],bins=20,alpha=0.5,label='Virginica')
#     ax.set_xlabel(columns[col])
#     ax.set_ylabel('count')
#     ax.legend()
# plt.show()

'''기타 text(), annotate()'''
# plt.scatter(X[:,0],X[:,1],c=X[:,-1],s=X[:,2]*100, alpha=0.2)
# plt.text(6,3.2,'sepal width\n/\nsepal length', fontsize=30, alpha=0.5,
# ha='center', va='center') # 배경에 글 위치나 폰트 진하기
# plt.annotate('Setosa',xy=(6,4),xytext=(6.5,4.2),arrowprops=dict(facecolor='black'))
# plt.show()   # 쓸 내용   좌표 화살표는 여기서 시작해서 텍스트 좌표 에서 끝같음  화살표 컬러
