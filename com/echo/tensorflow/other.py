import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


# 解决Sequential 无法搭建跳连的非顺序结构
class IrisModel(tf.Model):

    # 定义网络结构
    def __init__(self):
        super(IrisModel, self).__init__()
        # 定义网络结构
        self.d1 = layers.Dense(3)

    # 调用网络结构块, 实现前向传播
    def call(self, x):
        y = self.d1(x)
        return y

# 目前感觉就封装了一下
class MnistModel(tf.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128,activation='relu')
        self.d2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y

# 自治数据集, mnist
# 使用的时候先判断是否已经生成过
def generateds(path, txt):
    # txt 的格式 : 0_5.jpg  5

    f = open(txt , 'r') # 以只读格式打开txt
    contents = f.readlines() # 读取所有的行
    f.close()

    x, y_ = [], []
    for content in contents:
        value = content.split() # 以空格切割
        img_path = path + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L')) # 将图片转换为8位灰度值的np.array格式
        img = img / 255.
        x.append(img)
        y_.append(value[1])

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_

# 数据增强, 扩展数据集
def imageGen(x_train):
    image_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale = 0.1, #所有数据乘以该数值
        rotation_range = 45, #随机旋转角度范围 单位度
        width_shift_range = 0.15, #随机宽度偏移量
        height_shift_range= 0.15, #随机高度偏移量
        horizontal_flip = True, #是否随机水平旋转
        zoom_range = 0.5 # 随机缩放范围[1-n, 1+n], 缩放阈值为50%
    )
    # fit的时候需要四维数据, 最后一个通道是灰度值RGB
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    # 统计数据特征, 如均值,方差等
    image_gen.fit(x_train)

    # 具体使用时
    # model.fit(image_gen.flow(x_train, y_train, batch_size=32), ...)

# 断点续训, 可在以前训练的基础上继续训练
def load_weights(path):
    model = IrisModel()

    if os.path.exists(path + ".index"):
        model.load_weights(path)

def save_weights(path):
    model = IrisModel()
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(callbacks=[callback])

# 参数提取
def get_weight():
    mode = IrisModel()
    # 返回模型中可训练参数
    vars = mode.trainable_variables()

    # 不省略
    np.set_printoptions(threshold=np.inf)
    print(vars)

    # 写入到文件中
    file = open("./weights.txt", 'w')
    for var in vars:
        file.write(str(var.name) + '\n')
        file.write(str(var.shape) + '\n')
        file.write(str(var.numpy()) + '\n')
    file.close()

# acc/loss可视化
def show():
    model = IrisModel()
    history = model.fit()
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 3)
    plt.plot(acc, label='Train ACC')
    plt.plot(val_acc, lable='Validation ACC')
    plt.title("ACC")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train LOSS')
    plt.plot(val_loss, lable='Validation LOSS')
    plt.title("LOSS")
    plt.legend()
    plt.show()


# 应用程序
def app():
    model = MnistModel

    # 预处理图片
    image_path = input(":")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    # 因为我们的图是白底黑字, 要颜色取反
    img_arr = 255 - img_arr
    img_arr = img_arr/255.0
    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)
    predict = tf.argmax(result, axis=1)
    print(predict)

if __name__ == '__main__':
    # model = IrisModel()
    pass
