# coding=utf-8
# @Author : Eric

import matplotlib.pyplot as plt
import matplotlib
import data_process


def draw_length(docs):
    # 绘制每个问题的长度
    x = range(0, len(docs))
    y = [len(docs[i]) for i in x]
    plt.plot(x, y)
    plt.xlabel('问题编号')
    plt.ylabel('问题包含单词数目')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('Transformers问题长度图')
    plt.show()


def draw_num_of_length(docs):
    # 绘制每个数量对应的问题数量
    length = []
    for i in range(0, len(docs)):
        length.append(len(docs[i]))

    x = range(0, max(length))  # 取所有的
    # x = range(0, 100)  # 取所有长度小于100的
    y = [length.count(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('问题包含单词数目')
    plt.ylabel('对应问题数目')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('allennlp问题长度分布图')
    plt.show()


if __name__ == "__main__":
    docs = data_process.get_data()
    draw_length(docs)
    draw_num_of_length(docs)
