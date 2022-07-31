import tensorflow as tf
from sklearn import datasets
import numpy as np
from tensorflow import keras

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

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    # (特征, 标签, 批次, epoch, 测试集占比, 每20epoch验证一次)
    model.fit(x, y, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

    model.summary()

    # 查看权重
    # model.weights, layer.weights
    # 如何输出中间结果
    # 1. 可以将中间层赋值给对象,查看中间层结果, 如layer1 = keras.layers.Dense(3...)
    # out1 = layer1(in0)
    # 2. 取某一层的输出为输出新建为model，采用函数模型, keras.layers.Dense(3..., name="Dense_1")
    # dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense_1').output)
    # #以这个model的预测值作为输出
    # dense1_output = dense1_layer_model.predict(data)




