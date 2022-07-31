import threading
import time
from pynput import keyboard

select = 1

def shark():
    print("123")
    print("123")
    print("123")

    count = 0
    # 闪烁 你好 , 可以通过子线程绘制,然后主线程来接受 input, 进行控制
    while True:
        print("\033[4A+++++++++++++++++++++++++++")
        print("+", end='')
        if select == 1 and count % 2 == 1:
            print("                         ", end='')
        else:
            print("       你好              ", end='')
        print("+")
        print("+", end='')
        if select == 2 and count % 2 == 1:
            print("                         ", end='')
        else:
            print("       你好              ", end='')
        print("+")
        print("+++++++++++++++++++++++++++")
        time.sleep(0.5)
        count = count + 1

def on_press(text):
    global select

    if (text == 'w'):
        select = select - 1
    elif (text == 's'):
        select = select + 1

if __name__ == '__main__':
    threading.Thread(target=shark).start()

    # 监听键盘输入
    with keyboard.Listener(on_press=on_press) as lsn:
        lsn.join()
