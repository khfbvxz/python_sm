from sympy import *
'''
수식 출력을 LaTex 수식으로 보이게 한다. init_printing()
Rational( 분자, 분모 ) 함수는 유리수를 분수로 나타낸다
N() 함수는 수치값으로 계산한다 
evalf() 메소드로 수치값을 계산할 수도 있다
symbols 함수는 한 번에 여러개의 기호변수를 선언할 수 있어서 더 편리하다.
subs() 메소드로 기호변수에 특정한 값을 대입하거나, 다른 기호변수로 대체할 수 있다.
x  를  y  로 바꾸려면 expr.subs(x,y)
_ 는 바로 이전의 출력을 가리킨다. _.subs(y,2)
여러개의 변수에 값을 대입할 수 있다.
expr = x**2 + 2*y*z + 1
expr.subs( [ (x,1), (y,2), (z,3) ] )
simplify() 함수는 수식을 간단히 만들어준다 
simplify( cos(x)**2 + sin(x)**2 ) 1 
simplify(expr)
sympify() 함수는 문자열을 수식으로 바꾸어 준다. 사용자가 입력한 수식을 처리하는데 사용할 수 있다.
expand() 함수는 수식을 전개한다.expand( (x-1)*(x-2) )
factor() 함수는 수식을 인수분해 한다 factor( x**3 - 8 ) 
collect() 함수는 수식을 특정 변수의 다항식으로 정리한다. collect(expr, x) x3+x2(−z+2)+x(y+1)−3
   
coeff() 함수는 수식의 특정항의 계수를 반환한다. 
expr.coeff(x,2)       # x**2 항의 계수

cancel() 함수는 분수의 분자와 분모를 약분하여 간단한 형태로 만든다.
cancel(expr) (x+1)/x
또한, 분수식들을 통분하여, 하나의 분수식으로 만들어준다.

apart() 함수는 분수식을 부분 분수들로 쪼개어 준다.

방정식은 Eq( 왼쪽, 오른쪽 ) 으로 표현한다

미분 diff() 함수로 도함수를 구한다.
diff( cos(x), x ) -sin(x)
2차 도함수 diff( x**2, x, x )  # 2 
마지막 인자에 2 를 기입하여도 같은 결과를 얻는다.
diff( x**2, x, 2 ) # 2

Derivative() 함수는 도함수를 표현하는 용도에 사용된다.
deriv = Derivative( exp(-x**2), x )  # 얘는 직접해 
doit() 메소드로 연산을 수행하여 결과를 얻을 수 있다.
deriv.doit()
Eq( deriv, deriv.doit() )
diff() 함수로 편도함수를 구할 수 있다.
편도함수는 대부분의 경우에 미분의 순서에 관계 없다.
적분
integrate() 함수로 부정적분을 수행한다.
integrate( 1/x, x ) #log(x)
# 적분 상수는 따로 출력 되지 않는다!!
integrate() 함수로 정적분을 구할 수 있다. 적분 범위는 두번째 인자로 설정한다.
integrate( x**2, (x,0,1) )
무한대는 소문자 o 를 두개 사용하여 oo 로 나타낸다.

integrate( exp(-x**2), (x,0,oo) )

이중적분도 구할 수 있다.
Integral() 함수는 적분을 수학적으로 표현하는데 사용된다.
integ = Integral( log(x) )
integ
doit() 메소드로 실행 결과를 얻는다.

limit 함수로 극한을 구한다.
limit( sin(x)/x, x, 0 )

solve() 함수로 방정식의 해를 구한다.
solve( eqn, x )
이차방정식의 근의 공식
a, b, c = symbols('a, b, c')
eqn = Eq( a*x**2 + b*x + c, 0 )
solve( eqn, x )
기호변수는 암묵적으로 복소수로 취급되므로, 복소근도 구해진다.
연립 방정식의 해
방정식과 변수를 리스트로 묶어서 인수를 전달한다. 튜플도 가능 결과 딕셔너리 형태로 
solve( [ x-y+2, x+y-3 ], [x,y] )

Taylor 급수
series( sin(x), x )
x의 차수를 원하는 값으로 설정할 수 있다. 다음 예와 같이 10차 이전까지의 급수를 나타내면
series( sin(x), x, n=10 )
급수를 계산에 활용하기 위해서는, 마지막 차수항 O(x) 를 제거해야 한다. 이를 위해 removeO 메소드를 사용한다.
Sin = series( sin(x), x, n=10 ).removeO()
사인 함수를 9차의 다항식으로 근사하였다. 이 식에  x=π/2  를 대입하여 보면
Sin.subs( x, pi/2 ).evalf()

행렬 Matrix() 함수로 행렬을 만든다.
A = Matrix( [[1,2],[3,4]] )

det() 함수로 행렬식(determinant)을 구한다.
A.det() ad-bc

inv() 함수로 역행렬을 구한다
A.inv() 행렬의 -1 승으로 역행렬을 구할 수도 있다. A**-1

행렬의 원소를 기호로 표기하여 수식적으로 다룰 수 있다.
a11, a12, a21, a22 = symbols('a11, a12, a21, a22')
A = Matrix( [[a11, a12],[a21, a22]] )

연립 방정식의 해
A = Matrix( [[1,-1],[1,1]] )
b = Matrix( [-2,3] )
방정식의 해는 x=A−1b 


'''
# sympy
print("====2====")
init_printing() # 그리기 시작?
x, y, z = symbols('x y z')
a, b, c, t = symbols('a b c t')

#matplotlib inline
print("====3, 4 ====")
sin(pi/3)  # xlim  ylim x, y 범위
# plot(sin(x),xlim=(-6.28,6.28), ylim=(-2,2))
print("====6====")
# plot(cos(x),xlim=(-6.28,6.28), ylim=(-2,2))

print("====7====")
expand_trig(tan(a+b))
print(expand_trig(tan(a+b)))
print("====8====")
trigsimp(sin(t)+cos(t))
print(trigsimp(sin(t)+cos(t)))

print("====9====")
fx = x**2 -4
print(fx)

print("====19====")
gx = x - 2
print(gx)
print("====20====")
soln = solve( Eq(fx,gx),x) # 느낌이 fx-gx(x**2-x-2) 의 근
print(Eq(fx,gx))
print(soln) # 밑에 보면 만나는 지점
print("====21====")
# plot(fx,gx,xlim=(-6,6), ylim=(-10,10))
print("====22====")
# integrate 함수 적분
print(integrate(gx-fx , (x,soln[0],soln[1])))

print("====23====")
fx = integrate(x/(x**2+1), x)
print(fx)

print("====24====")
plot(fx, xlim=(-6,6), ylim=(-10,10))
