
# 进度条效果
# \b 消除前一个字符, 在下次输出时将前面的退出去
import time

def __init__():
    print("init")

def progressor():
    size = 0

    for i in range(100):
        for s in range(size):
            print("\b", end='')

        msg = f"{i}/{100}"
        size = len(msg)
        print(msg, end='')

        time.sleep(0.1)

if __name__ == '__main__':
    print("\033[30;41m输出内容")
    # \033是固定转移字符
    # [ 表示开始定义颜色
    # 0 为默认的字体颜色
    # 32 定义文本为绿色
    # 40 表示背景为黑色
    #
    # 30 (黑色)、31 (红色)、32 (绿色)、33 (黄色)、34 (蓝色)、35 (紫红色)、36 (青色)和37 (白色)
    # 背景色相同为40, 41 ...

    # 光标定位
    print("\033[0;0m123123123")
    print("123123123", end='')
    print("\raaaa")
    print("\033[1A123", end='')

