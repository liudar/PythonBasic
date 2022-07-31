from tensorflow import keras

def callback(log_dir):
    return keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)