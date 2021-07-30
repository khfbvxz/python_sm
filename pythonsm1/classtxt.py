# 객체는 속성과 동작을 가진다.
# 자동차는 메이커나 모델 색상 이 속성
#         주행하기 방향바꾸기 정치하기 동작
import  math
import  random
# 설계도를 클래스 라고 한다. 클래스란 특정한 종류의 객체들을 찍어내는 형틀(template)또는 청사진(blueprint)
# 클래스로부터 만들어지는 객체를 인스턴스 라한다.

class Counter:
    def __init__(self, initValue=0):  # 생성자 정의  # 사용자가 값을 전달하지 않았으면 0으로 생각한다.
        self.count = 0  # 인스턴스 변수 생성
    def increment(self): #메서드를 정의
        self.count += 1

# a1 = Counter() #Counter() 객체의 인스턴스 생성  # 객체생성?
# a1.increment()
# print("카운터의 값=", a1.count)
# #카운터의 값= 1
# a = Counter(100)  # 카운터 초기값은 100이 된다.
# b = Counter()     # 카운터의 초기 값은 0이 된다.
'''
class Television:
    def __init__(self, channel, volume, on):
        self.channel = channel
        self.volume = volume
        self.on = on

    def show(self):
        print(self.channel, self.volume, self.on)

    def setChannel(self, channel):
        self.channel = channel

    def getChannel(self):
        return self.channel
'''
# t = Television(9,10,True)
# t.show()
#
# t.setChannel(11)
# t.show()

'''
9 10 True
11 10 True

'''

# Circle 클래스를 정의한다.

class Circle:
    def __init__(self, radius = 0):
        self.radius = radius

    def getArea(self):
        return math.pi * self.radius

    def getPerimeter(self):
        return 2 * math.pi * self.radius

# Circle 객체
# c = Circle(10)
# print("원의 면적", c.getArea())
# print("원의 면적", c.getPerimeter())

class Car:
    def __init__(self,speed,color,model):
        self.speed = speed
        self.color = color
        self.model = model

    def drive(self):
        self.speed = 60
'''

myCar1 = Car(0, "blue", "E-class")
myCar2 = Car(40,"red","m5")
print("자동차 객체 생성")
print("벤츠속도는 ",myCar1.speed)
print("비엠속도는 ",myCar2.speed)
print("벤츠 색는 ",myCar1.color)
print("비엠 색는 ",myCar2.color)
print("벤츠 모델는 ",myCar1.model)
print("비엠 모델는 ",myCar2.model)

myCar1.drive()
myCar2.drive()
print("벤츠속도는 ",myCar1.speed)
print("비엠속도는 ",myCar2.speed)
'''
# 정보 은닉
# 파이썬에서 인스턴스 변수를 private으로 정의하려면 변수 이름 앞에 __ 을 붙이면 된다.
# private이 붙은 인스턴스 변수는 클래스 내부에서만 접근 될 수 있다.
# 인스턴스 변수값을 반화하는 접근자 getters
# 인스턴스 변수값을 설정하는 설정자 setters
class Student:
    def __init__(self,name=None, age=0):
        self.__name = name
        self.__age = age

    def getAge(self):
        return  self.__age

    def getName(self):
        return  self.__name

    def setAge(self,age):
        self.__age=age
    def setName(self,name):
        self.__name = name
# obj = Student()
# print(obj.__age)
#AttributeError: 'Student' object has no attribute '__age'
#
# obj1 = Student("Hong",20)
# print(obj1.getName())
# print(obj1.getAge())

class BankAcount:
    def __init__(self):
        self.__balance = 0

    def withdraw(self, amount):
        self.__balance += amount
        print("통장에", amount, "가 입금되었습니다.")
        return self.__balance
    def deposit(self, amount):
        self.__balance -= amount
        print("통장에", amount,"가 출금되었습니다")
        return self.__balance
# a = BankAcount()
# a.deposit(100)
# a.withdraw(10)
# # 통장에 100 가 출금되었습니다
# # 통장에 10 가 입금되었습니다.


# 객체 참조
class Television:
    def __init__(self,channel,volume,on):
        self.channel=channel
        self.volume=volume
        self.on=on

    def setChannel(self,channel):
        self.channel = channel
#참조 공유
t = Television(11,10,True)
s = t
s.channel = 9

#동일한 객체를 참조하고 있는지를 검사하는 연산자
# is is not

if s is t :
    print("동일한 객체를 참조하고 있음")

if s is not t:
    print("다른 객체를 참조하고 있음")
# 변수가 현재 아무것도 가리키고 있지 않다면 None으로 설정하는 것이 좋다
# None은 아무것도 참조하고 있지 않다는 것을 나타내는 특별한 값이다.

# myTv = None
# if myTv is None:
#     print("현재 tv업슴")
#
# 텔레비젼을 클래스로 정의 한다.

# 클래스 변수
# 모든 객체를 통틀어서 하나만 생성되고 모든 객체가 이것을 공유하는
# 변수를 클래스 멤버라고 한다.
class Television:
    serialNumber = 0 # 클래스 변수 모든 객체가 이것을 공유

    def __init__(self,channel,volume,on): #생성자 메서드
        self.channel = channel  #인스턴스 변수 선언 및 초기화
        self.volume = volume
        self.on = on
        Television.serialNumber += 1 # 클래스 변수를 하나 증가하낟.
        self.number = Television.serialNumber
        # 클래스 변수의 값을 객체의 실
    def show(self): #보여주는 show 메서드 생성
        print(self.channel,self.volume,self.on)
#전달 받은 텔레비전의 음량을 줄인다.
def setSilentMode(t):
    t.volume = 2

# setSilentMode()
myTV = Television(11,10,True)
myTV.show()
setSilentMode(myTV)
myTV.show()
'''
# 상수 정의
class Monster:
    # 상수 값 정의
    WEAK = 0
    NORMAL = 10
    STRONG = 20
    VERYSTRONG = 30
    def __init__(self):
        self.health = Monster.NORMAL

    def eat(self):
        self.__health = Monster.STRONG

    def attack(self):
        self.__health = Monster.WEAK
'''
class Dog:
    kind = "Bulldog"   # 클래스 변수
    def __init__(self,name,age):   #생성자 메서드
        self.name = name      # 각 인스턴스에 유일한 인스턴스 변수
        self.age = age        # 각 인스턴스에 유일한 인스턴스 변수
        self.pum = Dog.kind
    def show(self):

        print(self.name,"의 품종",self.pum)

na = Dog('미미',8)
na.show()

class Dice():
    def __init__(self,x,y):
        self._x = x
        self._y = y
        self._size = 30
        self._value = 1

    def read_dice(self):
        return self._value
    def print_dice(self):
        print("주사위의 값=",self._value)
    def roll_dice(self):
        self._value = random.randint(1,6)

d = Dice(100,100)
d.roll_dice()
d.print_dice() 