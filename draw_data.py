# coding=utf-8
# @Author : Eric

import matplotlib.pyplot as plt
import matplotlib
import pandas

import data_process
import settings


def draw_length(docs):
    # 绘制每个问题的长度
    x = range(0, len(docs))
    y = [len(docs[i]) for i in x]
    plt.plot(x, y)
    plt.xlabel('问题编号')
    plt.ylabel('问题包含单词数目')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('问题长度图')
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
    plt.title('问题长度分布图')
    plt.show()


def draw_num_of_length_2(docs):
    type1_count = 0
    type2_count = 0
    type3_count = 0
    type4_count = 0
    type5_count = 0
    type6_count = 0
    type7_count = 0
    for doc in docs:
        leng = len(doc)
        if leng < 25:
            type1_count += 1
        elif leng < 50:
            type2_count += 1
        elif leng < 75:
            type3_count += 1
        elif leng < 100:
            type4_count += 1
        elif leng < 125:
            type5_count += 1
        elif leng < 150:
            type6_count += 1
        else:
            type7_count += 1

    # 绘制每个数量对应的问题数量
    x = ['0-25', '25-50', '50-75', '75-100', '100-125', '125-150', '150+']
    # x = range(0, 100)  # 取所有长度小于100的
    y = [type1_count, type2_count, type3_count, type4_count, type5_count, type6_count, type7_count]
    plt.plot(x, y)
    plt.xlabel('问题包含单词数目')
    plt.ylabel('问题数量')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('Github Issue问题长度分布图')
    plt.show()


if __name__ == "__main__":
    docs = []
    for i in range(0, 7):
        nlp_api = settings.nlp_api[i]
        docs += data_process.get_data(nlp_api)
    # draw_length(docs)
    draw_num_of_length_2(docs)

    so_docs = []
    type1_count = 0
    type2_count = 0
    type3_count = 0
    type4_count = 0
    type5_count = 0
    type6_count = 0
    type7_count = 0
    for i in range(0, 7):
        nlp_api = settings.nlp_api[i]
        filepath = settings.stackoverflow_filepath[nlp_api]
        df = pandas.read_csv(filepath)
        df = df.fillna(0)
        for index, row in df.iterrows():
            body = row['Body']
            body = data_process.process_query(body)
            body = data_process.clean(body).split(' ')
            leng = len(body)
            if leng < 25:
                type1_count += 1
            elif leng < 50:
                type2_count += 1
            elif leng < 75:
                type3_count += 1
            elif leng < 100:
                type4_count += 1
            elif leng < 125:
                type5_count += 1
            elif leng < 150:
                type6_count += 1
            else:
                type7_count += 1

    # 绘制每个数量对应的问题数量
    x = ['0-25', '25-50', '50-75', '75-100', '100-125', '125-150', '150+']
    # x = range(0, 100)  # 取所有长度小于100的
    y = [type1_count, type2_count, type3_count, type4_count, type5_count, type6_count, type7_count]
    plt.plot(x, y)
    plt.xlabel('问题包含单词数目')
    plt.ylabel('问题数量')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('Stack Overflow问题长度分布图')
    plt.show()