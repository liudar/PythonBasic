
# 类和对象

class Person:
    def agree(self):
        print("OK")

    def setName(self,name):
        self.name = name

    def getName(self):
        return self.name

# 继承，多个父类用 逗号 隔开。
class HenanPerson(Person):
    pass


class BeijingPerson(Person):
    def __init__(self,name,age):
        self.name = name
        self.age = age

    # def __init__(self):
    #     pass


if __name__ == '__main__':
    echo = HenanPerson()

    # 两者等价
    echo.agree()
    HenanPerson.agree(echo)

    # 调用有参方法
    echo.setName('echo')
    print(echo.getName())

    #
    liudr = BeijingPerson('liudr',18)
    print(liudr.name)




