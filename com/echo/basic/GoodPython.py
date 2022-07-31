import sys

if __name__ == '__main__':
    name = 'echo'
    # 使用f打印
    print(f"name = {name}")

    # 打印系统信息
    print(sys.version)

    # 使用列表下标.
    names = {1,2,3}
    for i, n in enumerate(names):
        # i 是从零开始的下标, enumerate 中也可以指定开始位置.
        print(i, n)

