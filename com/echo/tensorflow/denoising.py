import tensorflow as tf
import tensorflow as tf
from sklearn import datasets
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import common
import matplotlib.pyplot as plt

# 和encode相比只是输入的数据不一样。
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

    # n = 5
    # plt.figure(figsize=(10, 4.5))
    # for i in range(n):
    #     # plot original image
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(x_test[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     if i == int(n / 2):
    #         ax.set_title('Original Images')
    #
    #     # plot noisy image
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(x_test_noisy[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     if i == int(n / 2):
    #         ax.set_title('Noisy Input')
    # plt.show()

    fun2(x_train_noisy, x_test_noisy, x_train, x_test)

def fun2(x_train_noisy,x_test_noisy, x_train, x_test):
    # 输入的大小
    input_size = 784
    # 隐藏层神经元的大小
    hidden_size = 64
    # 压缩向量长度为32
    compression_size = 32

    # denoising autoencoder与之前定义的一样
    input_img = Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation='relu')(input_img)
    compressed_vector = Dense(compression_size, activation='relu')(hidden_1)
    hidden_2 = Dense(hidden_size, activation='relu')(compressed_vector)
    output_img = Dense(input_size, activation='sigmoid')(hidden_2)

    autoencoder = Model(input_img, output_img)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train_noisy, x_train, epochs=3)
    fun3(autoencoder, x_test_noisy, x_test)

def fun3(autoencoder, x_test_noisy, x_test):
    n = 5
    # 可视化预测结果
    plt.figure(figsize=(10, 7))

    images = autoencoder.predict(x_test_noisy)

    for i in range(n):
        # plot original image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == int(n / 2):
            ax.set_title('Original Images')

        # plot noisy image
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == int(n / 2):
            ax.set_title('Noisy Input')

        # plot noisy image
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == int(n / 2):
            ax.set_title('Autoencoder Output')
    plt.show()

if __name__ == '__main__':
    fun1()