from random import choice

def mergeFunc(f1, f2):
    def newFunc(self):
        f1()
        f2()
    return newFunc
"""
类的拼接，处理，增加方法等
__add__      +
__lshift__   <<
__rshift__   >>
__dict__ 如果是类调用，则查看类的方法和属性， 如果是对象调用，查看对象的属性
__dir__  类似上面那个，这个只有名字，没有对应的类型
"""

class MyMeta(type):
    def __add__(self, other: "MyMeta"):
        mergeName = self.__name__ + other.__name__
        mergeDict = self.__dict__ | other.__dict__

        for k, v in self.__dict__.items():
            for ko, vo in other.__dict__.items():
                if type(v) == type(vo) == staticmethod:
                    mergeDict[k + ko] = mergeFunc(v, vo)
                else:
                    mergeDict[k + ko] = choice([v, vo])

        res = type(mergeName, (), mergeDict)
        return res

    def __lshift__(self, other): # <<
        print(self.__dict__)
        print(other.__dict__)
        for k, v in other.__dict__.items():
            setattr(self, k, v)
        return self

class Person(metaclass=MyMeta):
    a = 1
    @staticmethod
    def say():
        print(111111)

class Person2(metaclass=MyMeta):
    b = 1
    @staticmethod
    def say2():
        print(222222)


# 使用type直接生成类。
def person_init(self, name):
    self.name = name

def say(self):
    print(f"{self.name},{self.count}")

def create_clazz():
    L = type("Person", (), {"__init__": person_init, "count": 0, "say": say})
    l = L("123")
    l.say()


if __name__ == '__main__':
    # create_clazz()

    # L = Person + int
    # print(L.__dict__)
    # print(Person.__name__)

    person3 = Person + Person2
    for k, v in person3.__dict__.items():
        print(f"{k},{v}")
    print(person3().saysay2())


