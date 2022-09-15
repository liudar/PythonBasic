import tensorflow as tf
from sklearn import datasets
import numpy as np
from tensorflow import keras

# C:\Users\echo\AppData\Roaming\Python\Python39\Scripts
# 使用keras 搭建神经网络
if __name__ == '__main__':
    data = datasets.load_iris()
    x = data.data
    y = data.target

    np.random.seed(116)
    np.random.shuffle(x)
    np.random.seed(116)
    np.random.shuffle(y)
    tf.random.set_seed(116)

    model = keras.models.Sequential([keras.layers.Dense(3, activation='softmax', kernel_regularizer=keras.regularizers.l2())])

    log_dir = './'
    callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    # (特征, 标签, 批次, epoch, 测试集占比, 每20epoch验证一次)
    model.fit(x, y, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20, callbacks=[callback])

    model.summary()




