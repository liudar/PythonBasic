from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


###
# 特征抽取（特征值化） 将一些非数字的元素（如文本）转换为数字
#   如果是中文要选择一个分词器
# 特征预处理 将特征进行归一化 或 标准化处理
#   MinMaxScaler
#   StandardScaler
# 特征降维
#   特征选择
#   主成分分析 将高维转换为低维数据， 例如一个二维的坐标，转换为一维的长度
#
# Boosting 提升
#   弱学习： 正确率 跟 错误率差不多的学习方法
# 通过损失函数，加强弱学习，使其变成强学习的算法
#
#
# ###
def dict_demo():
    data = [{'city': '北京', 'temp': 100}, {'city': '上海', 'temp': 60}]
    transform = DictVectorizer()  # 默认为稀疏矩阵
    data_new = transform.fit_transform(data)
    print(transform.get_feature_names())
    print(data_new.toarray())
    return None


'''
对于英文来说， 会过滤掉单个字母, 也可以通过stop_words参数过滤掉不重要的单词。
对于中文来说， 要用分词器，如
pip install jieba
import jieba
text = " ".join(list(jieba.cut(text)))
'''


def text_demo():
    text = ["i love china, china is good"]
    transform = CountVectorizer()

    data_new = transform.fit_transform(text)
    print(transform.get_feature_names())
    print(data_new.toarray())


def text_demo2():
    text = ["i love china, china is good"]
    transform = TfidfVectorizer()

    data_new = transform.fit_transform(text)
    print(transform.get_feature_names())
    print(data_new.toarray())


if __name__ == '__main__':
    # dict_demo()
    text_demo2()
