
# 类和对象

class Person:
    def agree(self):
        print("OK")

class HenanPerson(Person):
    pass


if __name__ == '__main__':
    a = HenanPerson()
    a.agree()