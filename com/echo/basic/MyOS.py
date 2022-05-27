import os
import sys
# sys.path.append('/com/echo/basic/')
# os.path.join()

if __name__ == '__main__':
    # 获得当前路径
    print(os.getcwd())

    # 创建文件夹 如果已存在报错
    try:
        os.mkdir('e:\\echo')
    except Exception as e:
        print("error")


    # 删除文件夹
    os.rmdir('e:\\echo')