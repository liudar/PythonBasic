
import tensorflow as tf
import tensorflow as tf
from sklearn import datasets
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import common
import matplotlib.pyplot as plt

"""
用途： 数据降噪， 降维处理
"""
# 数据存放位置： C:\Users\echo\.keras\datasets
def show(autoencoder, x_test, y_test):
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

def encode():
    mnist = tf.keras.datasets.mnist
    fashion = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_test_fashion = x_test_fashion.astype('float32') / 255.0
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    x_test_fashion = x_test_fashion.reshape(len(x_test_fashion), -1)

    # 输入的大小。
    # 因为输入的图片 shape 是(28, 28)，将它展开后的大小是 784
    input_size = 784
    # 隐藏层神经元的大小
    hidden_size = 64
    # 压缩向量长度为 32
    compression_size = 2
    # autoEncoder 网络定义
    autoEncoder = keras.models.Sequential([
        # Flatten(),  # 拉直层，数据已经展开了， 不需要这层
        Dense(hidden_size, activation='relu'),  # 全连接层
        Dense(compression_size, activation='relu',name='compress'),
        Dense(hidden_size, activation='relu'),
        Dense(input_size, activation='sigmoid'),
    ])

    # 网络训练
    autoEncoder.compile(optimizer="adam",
                  loss='binary_crossentropy')

    # autoEncoder.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005),
    #               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #               metrics=['sparse_categorical_accuracy'])

    autoEncoder.fit(x_train, x_train, batch_size=32, epochs=1, validation_data=(x_test, x_test), validation_freq=20,
              callbacks=[common.callback('./encode')])
    autoEncoder.summary()

    # xxx = x_test.reshape(len(x_test), 28, 28) * 255
    # result = autoEncoder.predict(x_test)
    # yyy = result.reshape(len(x_test), 28, 28) * 255

    # xxx = x_test_fashion.reshape(len(x_test_fashion), 28, 28) * 255
    # result = autoEncoder.predict(x_test_fashion)
    # yyy = result.reshape(len(x_test_fashion), 28, 28) * 255

    # for i in range(1, 100):
    #     plt.imshow(xxx[i], cmap='gray')
    #     plt.show()
    #     plt.imshow(yyy[i], cmap='gray')
    #     plt.show()
    # a = 0

    show(autoEncoder,x_test,y_test)


if __name__ == '__main__':
    encode()








def show2(autoencoder, x_test):
    decoded_imgs = autoencoder.predict(x_test)

    # number of example digits to show
    n = 5
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        # plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape((28, 28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == int(n / 2):
            ax.set_title('Original Images')

        # plot reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape((28, 28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == int(n / 2):
            ax.set_title('Reconstructed Images')