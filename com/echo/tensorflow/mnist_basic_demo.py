import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
import common

# C:\Users\echo\AppData\Roaming\Python\Python39\Scripts
if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    # mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # tf.keras.layers.Flatten()

    # 灰度图
    # plt.imshow(x_train[0], cmap='gray')
    # plt.show()

    x_train, x_test = x_train/255.0, x_test/255.0

    # 搭建网络结构, 逐层描述每层网络, Sequential容器,翻译为序列的，顺序的, 封装了网络结构
    model = keras.models.Sequential([
        layers.Flatten(), # 拉直层
        layers.Dense(128, activation='relu'), # 全连接层
        layers.Dense(10, activation='softmax')
    ])
    # 卷积层 layers.Conv2D()

    # 配置优化器, 损失函数, 评测指标
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    # 执行训练过程, 配置训练数据, 批次, epoch, 测试集, 每多少epoch测试一次.
    model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=20, callbacks=[common.callback('./mnist_basic')])
    # model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=20)

    # 打印网络结构和参数统计
    model.summary()



