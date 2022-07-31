from sklearn import datasets
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# 使用通用api搭建神经网络
if __name__ == '__main__':
    # 准备数据
    data = datasets.load_iris()
    x = data.data
    y = data.target

    # 数据集乱序
    np.random.seed(116)
    np.random.shuffle(x)
    np.random.seed(116)
    np.random.shuffle(y)
    tf.random.set_seed(116)

    # 拆分训练集和测试集
    x_train = x[:-30]
    y_train = y[:-30]
    x_test = x[-30:]
    y_test = y[-30:]

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    # 配对, 并32一个批次
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


    epoch = 500
    lr = 0.1
    train_loss_result = []
    test_acc = []
    loss_all = 0

    # 搭建网络
    # 因为是 x * w1 + b, 这里x有4个特征, 所以w1要有4列, 输出要有3个对应3种鸢尾花, b也要有3个.
    w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
    # 这里b的个数要和w的对应
    b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

    for epoch in range(epoch):
        for step, (x_train, y_train) in enumerate(train_db):
            with tf.GradientTape() as tape:
                y = tf.matmul(x_train, w1) + b1
                y = tf.nn.softmax(y)
                y_ = tf.one_hot(y_train, depth=3)
                loss = tf.reduce_mean(tf.square(y_ - y))
                loss_all += loss.numpy()
            grads = tape.gradient(loss, [w1, b1])
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])

        #print(f"epcho = { epoch }, loss = { loss_all / 4 }")
        train_loss_result.append(loss_all / 4)
        loss_all = 0

        # 测试效果
        total_corrent, total_number = 0, 0
        for x_test, y_test in test_db:
            y = tf.matmul(x_test, w1) + b1
            # 激活函数 softmax, relu, sigmoid
            y = tf.nn.sigmoid(y)
            pred = tf.argmax(y, axis=1)
            pred = tf.cast(pred, dtype=y_test.dtype)
            corrent = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
            corrent = tf.reduce_sum(corrent)
            total_corrent += int(corrent)
            total_number += x_test.shape[0]

        acc = total_corrent / total_number
        test_acc.append(acc)
        #print(f"test_acc = { test_acc }")


    # 可视化
    plt.title("Loss Function Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_loss_result, label="$Loss$")
    plt.legend()
    plt.show()

    plt.title("Acc Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(test_acc, label="$Acc$")
    plt.legend()
    plt.show()
