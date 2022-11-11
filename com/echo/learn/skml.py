import sklearn.svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


# 保存：joblib.dump(rf, 'test.pkl')
# 加载：estimator = joblib.load('test.pkl')
# python 先翻译成字字节码文件， 然后在PVM(python的虚拟机)上执行。

def train(iris, estimator):
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=123)

    # std = StandardScaler()
    # x_train = std.fit_transform(x_train)
    # x_test = std.fit_transform(x_test)

    estimator.fit(x_train, y_train)
    print(estimator.score(x_test, y_test))

    # y_pred = model.predict(X_test)
    # y_pred_pro = model.predict_proba(X_test)
    #
    # print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
    # print(metrics.classification_report(y_test, y_pred))
    # skplt.metrics.plot_roc(y_test, y_pred_pro)
    # skplt.metrics.plot_precision_recall_curve(y_test, y_pred_pro)
    # skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

    # 将模型持久化
    joblib.dump(estimator, "test.pkl")
    return None


def train2(iris, estimator):
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=123)

    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    estimator.fit(x_train)
    print(y_test)
    print(estimator.predict(x_test))
    # 聚类的类别和我们的类别名字不一样，所以不用用这个方法进行比较
    # print(estimator.score(x_test, y_test))
    # 所有样本的平均轮廓系数
    print(silhouette_score(iris.data, iris.target))
    return None


def pca_demo():
    pca = PCA(0.95)
    data = [[2, 8, 4, 5],
            [6, 3, 0, 8],
            [5, 4, 9, 1]]
    print(pca.fit_transform(data))


def grid_demo():
    knn = KNeighborsClassifier()
    iris = load_iris()
    grid = GridSearchCV(knn, {"n_neighbors": [2, 3, 4, 5, 6, 7, 8]}, cv=2)
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=23)
    grid.fit(x_train, y_train)

    print(grid.score(x_test, y_test))
    print(grid.best_estimator_)
    print(grid.best_score_)


if __name__ == '__main__':
    iris = load_iris()
    # 贝叶斯
    # nb = MultinomialNB()

    # 近邻
    # knn = KNeighborsClassifier()

    # 决策树
    # tree = DecisionTreeClassifier()

    # 聚类
    # km = KMeans(n_clusters=3)

    # SVM
    svm = SVC(kernel='linear', probability=True,  C=0.7)

    train(iris, svm)
    # train(iris, nb)
    # train(iris, knn)
    # train(iris, tree)
    # train2(iris, km)

    # 加载持久化的模型
    # estimator = joblib.load("test.pkl")
    # print(estimator.score(iris.data, iris.target))

    # pca_demo()
    # grid_demo()

