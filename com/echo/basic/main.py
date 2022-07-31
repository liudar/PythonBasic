# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import progress
import sys
from ..tensorflow import demo

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.

    # 如何导入其他文件夹中的模块
    # 1. 所有sys.path 中的都会加载 ,所有可以sys.path.append("我们的文件夹")
    # 2. 将文件夹加入到环境变量中
    # 3. 添加到lib/site-packages
    print(sys.path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # progress.progressor()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
