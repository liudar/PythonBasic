
import time

# 30 (黑色)、31 (红色)、32 (绿色)、33 (黄色)、34 (蓝色)、35 (紫红色)、36 (青色)和37 (白色)
# # 背景色相同为40, 41 ...
# \033[0m 关闭所有属性
# \033[1m 设置高亮度
# \033[4m 下划线
# \033[5m 闪烁
# \033[7m 反显
# \033[8m 消隐
# \033[30m — \033[37m 设置前景色
# \033[40m — \033[47m 设置背景色
# \033[nA 光标上移n行  用这个都可以实现修改内容了
# \033[nB 光标下移n行
# \033[nC 光标右移n行
# \033[nD 光标左移n行
# \033[y;xH设置光标位置
# \033[2J 清屏
# \033[K 清除从光标到行尾的内容
# \033[s 保存光标位置
# \033[u 恢复光标位置
# \033[?25l 隐藏光标
# \033[?25h 显示光标
# \a 发出警告声；
# \b 删除前一个字符；
# \c 最后不加上换行符号；
# \f 换行但光标仍旧停留在原来的位置；
# \n 换行且光标移至行首；
# \r 光标移至行首，但不换行；
# \t 插入tab；
# \v 与\f相同；
# \ 插入\字符；

ERRORCOLOR = "\033[30;43m"
COMMONCOLOR = "\033[0;0m"

# 选择行

def error(msg):
    print(f"{ ERRORCOLOR }{ msg }")

def info(msg):
    print(f"{ COMMONCOLOR }{ msg }")

def pre(n, msg):
    print(f"\033[{n}A{msg}")



def shark():
    print("123")
    print("123")
    print("123")

    count = 0
    # 闪烁 你好 , 可以通过子线程绘制,然后主线程来接受 input, 进行控制
    while (True):
        print("\033[3A+++++++++++++++++++++++++++")
        print("+", end='')
        if count % 2 == 1:
            print("       你好              ", end='')
        else:
            print("                         ", end='')
        print("+")
        print("+++++++++++++++++++++++++++")
        time.sleep(0.5)
        count = count + 1



if __name__ == '__main__':
    error("123")
    shark()