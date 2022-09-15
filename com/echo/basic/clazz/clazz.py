from dataclasses import dataclass

@dataclass()
class Vector:
    x:int
    y:int

class Person:
    def __init__(self, name):
        self.name = name

    # _不提示方法， __私有方法
    def __age(self):
        return "18"
    # function
    def say(self):
        print(f"my name is {self.name}")

    # toString()
    def __str__(self):
        return f"{self.name}"

    # 打印列表中的Person， 会调用这个， 而不是显示内存地址
    def __repr__(self):
        return f"{self.name}"

    # classmethod
    @classmethod #静态方法， cls代表类本身
    def create_person(cls):
        return cls("name")

    # with 必须有这两两个方法, as 是给enter返回值重命名
    def __enter__(self):
        print("enter")
        return self

    def __len__(self):
        return len(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 在这里处理异常
        print("exit")
        print(exc_tb)
        print(exc_type)
        print(exc_val)

        # 返回true， 则代表在这里处理异常， 异常不会抛出
        return True

    # person + person
    def __add__(self, other):
        self.name = self.name + other.name
        return self

    def calc(self):
        100 / 0

def aaa():
    a = 0

# 如果方法的返回值是有enter方法也是可以的
def create_person() -> Person:
    return Person.create_person()

if __name__ == '__main__':
    # p = Person.create_person()
    # print(hasattr(p, 'name'))
    # print(hasattr(p, 'age'))
    # print(getattr(p, "name"))

    # 是try catch finally的简化版，这里的异常都会被 __exit__ 处理掉。
    # aaa在下面还可以用，只处理方法块内的异常

    # with Person.create_person() as aaa:
    #     100 / 0
    #
    # print(len(aaa))
    #
    # with create_person() as p:
    #     100/0
    #
    # v = Vector(1, 2)
    # print(v)
    #
    # aaa = Person("aaa") + Person("bbb")
    # print(aaa.name)

    # 对象只打印类的属性， 而类会打印方法，静态方法等。
    # for k, v in Person.create_person().__dict__.items():
    #     print(f"{k} -> {type(k)},{v} -> {type(v)}")

    # print(Person.create_person().__dict__)

    # 打印所有属性和方法组成的列表， 没有对应的类型啊。
    print(Person.create_person().__dir__())

