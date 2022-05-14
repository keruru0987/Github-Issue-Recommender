# coding=utf-8
# @Author : Eric

import numpy as np
from collections import Counter


class BM25(object):
    def __init__(self, docs, titles, tags_list):
        self.docs = docs   # 传入的docs要求是已经分好词的list
        self.doc_num = len(docs)  # 文档数
        self.vocab = set([word for doc in self.docs for word in doc])  # 文档中所包含的所有词语
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.doc_num  # 所有文档的平均长度
        self.k1 = 1.5
        self.b = 0.75

    def idf(self, word):
        if word not in self.vocab:
            word_idf = 0
        else:
            qn = {}
            for doc in self.docs:
                if word in doc:
                    if word in qn:
                        qn[word] += 1
                    else:
                        qn[word] = 1
                else:
                    continue
            word_idf = np.log((self.doc_num - qn[word] + 0.5) / (qn[word] + 0.5))
        return word_idf

    def score(self, word):
        # 输入一个word，求他与每个doc的得分，最后将得分汇总后返回
        score_list = []
        for index, doc in enumerate(self.docs):
            word_count = Counter(doc)
            if word in word_count.keys():
                f = (word_count[word]+0.0) / len(doc)
            else:
                f = 0.0
            r_score = (f*(self.k1+1)) / (f+self.k1*(1-self.b+self.b*len(doc)/self.avgdl))
            score_list.append(self.idf(word) * r_score)
        return score_list

    def score_all(self, sequence, so_title_text, so_tags):
        # 将各个单词的得分相加
        score = []
        count = 0
        for word in sequence:
            score.append(self.score(word))
            # print('the round', count, '/', len(sequence), 'complete')
            count += 1
        sum_score = np.sum(score, axis=0)  # 纵轴求和
        # 得分除去query的长度
        if len(sequence) != 0:
            sum_score /= len(sequence)
        # 转换为列表形式返回
        return sum_score.tolist()
