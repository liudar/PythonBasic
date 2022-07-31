import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


if __name__ == '__main__':
    image_path = input("请输入图片路径: ")
    image_path = "C:/Users/liudr/Desktop/" + image_path + ".png"
    print(image_path)
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))
    img_arr = 255 - img_arr

    plt.imshow(img_arr, cmap='gray')
    plt.show()