# coding=utf-8
# @Author : Eric

import numpy as np
import data_process
from scipy.linalg import norm
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


class Sentence2Vec(object):
    # 传入未清洗的数据
    def __init__(self, docs):
        self.docs = docs
        self.doc_num = len(docs)
        self.vocab = set([word for doc in self.docs for word in doc])

    def score_cos(self, v1, v2):
        if norm(v1)*norm(v2) != 0:
            return np.dot(v1, v2) / (norm(v1) * norm(v2))
        else:
            return 0

    def word2sentence(self, wordlist):
        str_res = ''
        for word in wordlist:
            str_res += word
        return str_res

    def score_all(self, sequence):
        scores = []
        # query = self.word2sentence(sequence)
        v_query = model.encode(sequence)
        for doc in self.docs:
            # 如果doc为空，直接记为0分
            if len(doc) > 0:
                # doc_sentence = self.word2sentence(doc)
                v_doc = model.encode(doc)
                scores.append(self.score_cos(v_query, v_doc))
            else:
                scores.append(0)
            # print('1')
        return scores


if __name__ == '__main__':
    docs = data_process.get_raw_data('TextBlob')
    query = data_process.get_query()
    sentence2vec = Sentence2Vec(docs)
    scores = sentence2vec.score_all(query)
    print(scores)