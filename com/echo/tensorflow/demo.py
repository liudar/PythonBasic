import tensorflow as tf
import numpy as np

# pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple

def __init__():
    pass

def tensor():
    a = tf.constant([1,3], dtype=tf.int64)
    print(a)

    # 创建二行二列的, 用 0 填充
    a = tf.zeros([2, 2])
    print(a)

    # 创建三维的, 用 1 填充
    a = tf.ones([2, 2, 2])
    print(a)

    # 用 9 填充
    a = tf.fill([2, 2], 9)
    print(a)

    # numpy的数组转换, 他两个是互通的.
    a = np.array([1, 2, 3])
    a = tf.convert_to_tensor(a, dtype=tf.int64)
    print(a)

    # 正太分布, 维度, 均值, 标准差
    tf.random.normal([2, 2], mean=1, stddev=4)

    # 生成截断式正态分布的随机数, 不会生成在 2σ 之外的数据.
    tf.random.truncated_normal([2, 2], mean=1, stddev=4)

    # 生成均匀分布的随机数 维度, 最小值, 最大值
    tf.random.uniform([2, 2], minval=2, maxval=18)

def funs():
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(a)

    # 转换tensor的数据类型
    a = tf.cast(a, dtype=tf.float32)
    print(a)

    # 获取tensor中最小值
    print(f"min = { tf.reduce_min(a) }")
    # 获取最大值
    print(f"max = { tf.reduce_max(a) }")

    # 均值, axis=0 竖着的列, axis=1 横着的行, 默认是所有元素.
    a = tf.reduce_mean(a, axis=0)
    print(a)

    a = tf.reduce_sum(a, axis=0)
    print(a)

    # 四则运算 add, subtract, multiply, divide
    # 平方, 次方, 开放  square pow sqrt
    # 矩阵乘 matmul
    # 关联特征标签 tf.data.Dataset.from+tensor_slices((features,labels))
    # 求导 tf.GradientTape().gradient(函数, 对谁求导)
    # 独热码 tf.one_hot
    # tf.nn.softmax
    # 更新数值并返回,要Variable类型的 assign_sub
    # 指定维度最大的索引 argmax

def var():
    # 将变量标记为可训练的, 被标记的变量会在反向传播中更新梯度的值. 用来标记训练参数如w, b
    w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
    print(w)

if __name__ == '__main__':
    # var()
    print(tf.random.truncated_normal([4, 3], stddev=0.1, mean=1))


