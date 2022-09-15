from tensorflow import keras
import numpy as np

def callback(log_dir):
    return keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


"""
    reshape(1,-1)转化成1行： 
    reshape(2,-1)转换成两行： 
    reshape(-1,1)转换成1列： 
    reshape(-1,2)转化成两列
    reshape(10000, 28, 28) # 
    reshape(10000, -1)     # 
"""
if __name__ == '__main__':
    a = [[1,2,3,4,5,6],[2,3,4,5,6,7]]
    a = np.array(a)
    b = a.reshape(2, 2, 3)
    b = b.reshape(2, -1)
    print(a.shape)
