from sklearn import neural_network
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def add():
    xs = [[1, 123], [2, 12], [3, 53], [4, 65], [5, 65], [6, 23], [7, 43]]
    ys = [[sum(x)] for x in xs]

    print(ys)

    xt = [[9, 98], [3, 1], [12, 23], [2, 43]]
    yt = [[sum(x)] for x in xt]

    nn = neural_network.MLPRegressor(hidden_layer_sizes=(2),
                                     activation="relu",
                                     learning_rate_init=0.001,
                                     max_iter=20000
                                     )

    nn.fit(xs, ys)

    pre = nn.predict(xt)

    plt.plot(yt, 'b-', linewidth='2', label='yt')
    # plt.plot(xs,ys)
    plt.plot(pre, label='pre')
    plt.grid()  # axix='x'
    plt.legend()  # 图例
    plt.show()

    print(yt)
def pow():
    xs = []
    ys = []

    for i in range(0, 1000, 3):
        xs.append([i])
        ys.append([i*i*i])

    # print(ys)
    xt = []
    yt = []
    for i in range(1, 1000, 3):
        xt.append([i])
        yt.append([i*i*i])

    # ['identity', 'logistic', 'relu', 'softmax', 'tanh']
    nn = neural_network.MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,100),
                                     activation="relu")

    nn.fit(xs, ys)
    print(nn.score(xt,yt))

    pre = nn.predict(xt)

    # bar 柱状图
    # scatter 点
    # plt.plot(xt,yt, 'bD-', linewidth='1', label='yt')
    plt.plot(xt,yt, 'b-', linewidth='2', label='yt')
    # plt.plot(xs,ys)
    plt.plot(xt,pre, label='pre')
    plt.grid() # axix='x'
    plt.legend() # 图例
    plt.show()

    print(f"均方误差： {metrics.mean_squared_error(pre, yt)}")
    print(f"R： {metrics.r2_score(pre, yt)}")
    # yt = nn.predict(xt)
    # print(yt)

if __name__ == '__main__':
    add()