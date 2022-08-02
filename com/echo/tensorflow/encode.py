import tensorflow as tf
from sklearn import datasets
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import common
import matplotlib.pyplot as plt

"""
用途： 数据降噪， 降维处理, 压缩后用来比较搜索，
"""
# 数据存放位置： C:\Users\echo\.keras\datasets

def show1(autoencoder, x_test):
    # 查看压缩后的值
    autoencoder1 = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('compress').output)
    autoencoder2 = Model(inputs=autoencoder.get_layer('in2').input, outputs=autoencoder.get_layer('out').output)
    compress_imgs = autoencoder1.predict(x_test)
    decoded_imgs = autoencoder2.predict(compress_imgs)

    # number of example digits to show
    n = 5
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        # plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(compress_imgs[i].reshape((4, 4)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == int(n / 2):
            ax.set_title('Original Images')

        # plot reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape((28, 28)))
        # 压缩后的值
        # plt.imshow(decoded_imgs[i].reshape((4, 4)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == int(n / 2):
            ax.set_title('Reconstructed Images')
    plt.show()

def get_next_layers(model):
    model_layers = model.layers
    next_layers = {}
    for i in range(len(model_layers) - 1):
        if model_layers[i].name != '':
            next_layers[model_layers[i].name] = model_layers[i + 1]
    return next_layers

def get_layer_names(model):
    model_layers = model.layers
    layer_names = []
    for i in range(len(model_layers)):
        if model_layers[i].name != '':
            layer_names.append(model_layers[i].name)
    return layer_names

def show(autoencoder, x_test, count, *layer_names):
    plt.figure(figsize=(10, 10))
    layer_size = len(layer_names)

    # 如何为空则取 有名字的
    if layer_size == 0:
        layer_names = get_layer_names(autoencoder)
        layer_size = len(layer_names)

    # [<keras.layers.core.dense.Dense object at 0x0000018115086D60>,
    # <keras.layers.core.dense.Dense object at 0x0000018115468BE0>,
    # <keras.layers.core.dense.Dense object at 0x0000018113E0CAC0>,
    # <keras.layers.core.dense.Dense object at 0x000001811774DDC0>]
    next_layers = get_next_layers(autoencoder)

    # layer必须是顺序的, 可以只输入一个值
    pre = ''
    pre_data = x_test

    # if show_input: show原始数据
    for i in range(count):
        ax = plt.subplot(layer_size + 1, count, i + 1)
        image = pre_data[i]
        image_size = int(np.sqrt(len(image)))
        plt.imshow(image.reshape((image_size, image_size)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == int(count / 2):
            ax.set_title("input")

    for j in range(layer_size):
        layer = layer_names[j]
        if pre == '':
            model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer).output)
        else:
            model = Model(inputs=next_layers[pre].input, outputs=autoencoder.get_layer(layer).output)

        pre_data = model.predict(pre_data)
        for i in range(count):
            ax = plt.subplot(layer_size + 1, count, i + 1 + (j + 1) * count)
            image = pre_data[i]
            image_size = int(np.sqrt(len(image)))
            plt.imshow(image.reshape((image_size, image_size)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == int(count / 2):
                ax.set_title(layer)

        pre = layer
    plt.show()

# 压缩为2维后， 在坐标轴上查看
def show3(autoencoder, x_test, y_test):
    model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('compress').output)
    y_pre = model.predict(x_test)
    x_array = [ [],[],[],[],[],[],[],[],[],[] ]
    y_array = [ [],[],[],[],[],[],[],[],[],[] ]

    for i in range(len(y_test)):
        y = y_test[i]
        t = y_pre[i]

        xxx = x_array[y]
        yyy = y_array[y]

        xxx.append(t[0])
        yyy.append(t[1])

    for i in range(len(y_array)):
        plt.scatter(x_array[i], y_array[i], label=str(i))

    plt.legend()
    plt.show()
    
# 给数据增加噪声
def noise():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 生成带噪音的数据 随机生成一些数相加，然后提出小于0和大于1的数字。
    noise_factor = 0.4
    x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

    # 清理小于0与大于1的数据
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
    
def encode():
    # 64  32/16
    mnist = tf.keras.datasets.mnist
    fashion = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion.load_data()

    x_train = x_train_fashion.astype('float32') / 255.0
    x_test = x_test_fashion.astype('float32') / 255.0
    # x_test_fashion = x_train.astype('float32') / 255.0
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    # x_test_fashion = x_test_fashion.reshape(len(x_test_fashion), -1)

    # 输入的大小。
    # 因为输入的图片 shape 是(28, 28)，将它展开后的大小是 784
    input_size = 784
    # 隐藏层神经元的大小
    hidden_size = 64
    # 压缩向量长度为 32
    compression_size = 36
    # autoEncoder 网络定义
    autoEncoder = keras.models.Sequential([
        # Flatten(),  # 拉直层，数据已经展开了， 不需要这层
        Dense(hidden_size, activation='relu', name="in1"),  # 全连接层
        Dense(compression_size, activation='relu', name="compress"),
        Dense(hidden_size, activation='relu', name="in2"),
        Dense(input_size, activation='sigmoid', name='out'),
    ])
    # 查看各层
    #print(autoEncoder.layers)

    # 网络训练
    autoEncoder.compile(optimizer="adam",
                  loss='binary_crossentropy')

    autoEncoder.fit(x_train, x_train, batch_size=32, epochs=5, validation_data=(x_test, x_test), validation_freq=20,
              callbacks=[common.callback('./encode')])
    autoEncoder.summary()

    show(autoEncoder, x_test, 10)


if __name__ == '__main__':
    encode()
