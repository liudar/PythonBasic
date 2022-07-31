import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from PIL import Image
import numpy as np

def show(history):
    # 报错没有对应val_ 是因为fit的时候 validation_freq 设置的不对.
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train ACC')
    plt.plot(val_acc, label='Validation ACC')
    plt.title("ACC")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train LOSS')
    plt.plot(val_loss, label='Validation LOSS')
    plt.title("LOSS")
    plt.legend()
    plt.show()

def app(model):
    while True:

        image_path = input("请输入图片路径: ")
        image_path = "C:/Users/liudr/Desktop/" + image_path + ".png"
        print(image_path)
        img = Image.open(image_path)
        img = img.resize((28, 28), Image.ANTIALIAS)
        img_arr = np.array(img.convert('L'))

        # 因为我们的图是白底黑字, 要颜色取反
        img_arr = 255 - img_arr
        img_arr = img_arr/255.0
        x_predict = img_arr[tf.newaxis, ...]
        result = model.predict(x_predict)
        predict = tf.argmax(result, axis=1)
        print(result)
        tf.print(predict)
        print("--------------------")

if __name__ == '__main__':
    # mnist = tf.keras.datasets.mnist
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    tf.keras.layers.Flatten()

    # 灰度图
    # plt.imshow(x_train[0], cmap='gray')
    # plt.show()

    x_train, x_test = x_train/255.0, x_test/255.0

    model = keras.models.Sequential([
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

    model.summary()

    # show(history)

    app(model)



