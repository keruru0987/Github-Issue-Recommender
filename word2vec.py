# coding=utf-8
# @Author : Eric

import numpy as np
from scipy.linalg import norm
import settings
import gensim

model_file = settings.word2vec_modelpath
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)


class Word2Vec(object):
    def __init__(self, docs):
        self.docs = docs
        self.doc_num = len(docs)
        self.vocab = set([word for doc in self.docs for word in doc])

    # 将一个问题中所有词的词向量求平均，作为该问题的问题向量
    def sentence_vector(self, doc):
        v = np.zeros(300)
        count = 0
        for word in doc:
            # 判断该词在不在word2vec模型里面，后续可以用自己的语料来训练模型，避免这个问题
            if model.has_index_for(word):
                v += model[word]
                count += 1
        if count!=0:
            v /= count
        return v

    def score_cos(self,v1,v2):
        if norm(v1)*norm(v2) != 0:
            return np.dot(v1, v2) / (norm(v1) * norm(v2))
        else:
            return 0

    def score_all(self, sequence):
        scores = []
        v_query = self.sentence_vector(sequence)
        for doc in self.docs:
            # 如果doc为空，直接记为0分
            if len(doc)>0:
                v_doc = self.sentence_vector(doc)
                scores.append(self.score_cos(v_query,v_doc))
            else:
                scores.append(0)
        return scores
