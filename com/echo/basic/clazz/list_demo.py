
from clazz import Person

def open_file():
    with open("../citys", encoding="utf-8") as citys:
        for city in citys.readlines():
            print(str(city))

def list_demo():
    list = [1, 2, 3, 4, 5]
    a = [x*x for x in list if x > 3]
    print(a)

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # flat
    list1 = [x for row in matrix for x in row]
    print(list1)

    list2 = [[x for x in row] for row in matrix]
    print(list2)

    list = ['1']
    aa = list[0] or 0

    print(f"{aa!r}")

if __name__ == '__main__':
    list_demo()