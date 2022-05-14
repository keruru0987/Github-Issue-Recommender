# coding=utf-8
# @Author : Eric

import math
import pandas as pd


class VSM(object):
    def __init__(self, docs, titles, tags_list):
        self.docs = docs
        self.doc_num = len(docs)
        self.vocab = set([word for doc in self.docs for word in doc])

    def computeTF(self, vocab, doc):
        tf = dict.fromkeys(vocab, 0)
        for word in doc:
            tf[word] += 1
        return tf

    def computeIDF(self, tfList):
        idfDict = dict.fromkeys(tfList[0], 0)  # 词为key，初始值为0
        N = len(tfList)  # 总文档数量
        for tf in tfList:  # 遍历字典中每一篇文章
            for word, count in tf.items():  # 遍历当前文章的每一个词
                if count > 0:  # 当前遍历的词语在当前遍历到的文章中出现
                    idfDict[word] += 1  # 包含词项tj的文档的篇数df+1
        for word, Ni in idfDict.items():  # 利用公式将df替换为逆文档频率idf
            idfDict[word] = math.log10(N / Ni)  # N,Ni均不会为0
        return idfDict  # 返回逆文档频率IDF字典

    def computeTFIDF(self, tf, idfs):  # tf词频,idf逆文档频率
        tfidf = {}
        for word, tfval in tf.items():
            tfidf[word] = tfval * idfs[word]
        return tfidf

    def score_all(self, sequence, so_title_text, so_tags):
        tf_list = []
        for doc in self.docs:
            tf = self.computeTF(self.vocab, doc)
            tf_list.append(tf)
        idfs = self.computeIDF(tf_list)
        tf_idf_list = []
        for tf in tf_list:
            tf_idf = self.computeTFIDF(tf, idfs)
            tf_idf_list.append(tf_idf)
        Dvector = pd.DataFrame([tfidf for tfidf in tf_idf_list])  # 文档的向量

        query = []
        for word in sequence:
            if word in self.vocab:
                query.append(word)
            else:
                continue
        tf = self.computeTF(self.vocab, query)
        Q_tf_idf = self.computeTFIDF(tf, idfs)  # Query的向量

        scores = []
        for vector in Dvector.to_dict(orient='records'):
            score = 0
            for k in Q_tf_idf:
                if k in vector:
                    score += Q_tf_idf[k]*vector[k]
            scores.append(score)
        return scores
