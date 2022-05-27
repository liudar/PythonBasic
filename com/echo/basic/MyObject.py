
# 类和对象 定义： 可执行的代码块
# 只能有一个构造函数
# 多个构造函数： 可以使用多参数，然后默认值代替。
# python 中没有接口的定义，全靠自觉

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

        # 调用私有方法
        self.__changeName('hhh')

    # 私有方法： 前面加 '__'
    def __changeName(self,name):
        self.name = name

# 对象的属性或方法，相当于java的反射
def attr():
    echo = BeijingPerson("echo",18)
    # 是否有name属性或方法
    hasName = hasattr(echo,'name')
    print(hasName)

    # 获取属性的值
    print(getattr(echo,'name'))

    #设置属性的值
    setattr(echo,'name','echo')
    print(echo.getName())


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

    # 外部调用私有方法,写此行代码的时候没有提示。
    liudr._BeijingPerson__changeName('eee')
    print(liudr.name)


    # 判断前者 是否是 后者的子类
    issub = issubclass(BeijingPerson,Person)
    print(issub)

    # 是否实例
    isinst = isinstance(echo,BeijingPerson)
    print(isinst)

    attr()





