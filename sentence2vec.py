# coding=utf-8
# @Author : Eric

import numpy as np
import data_process
from scipy.linalg import norm
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
import settings

# model = SentenceTransformer('all-mpnet-base-v2')
# model = SentenceTransformer('bert-base-nli-mean-tokens')
# model = SentenceTransformer('all-distilroberta-v1')
# model = SentenceTransformer('all-MiniLM-L12-v2')
# model = SentenceTransformer('multi-qa-distilbert-cos-v1')
# model = SentenceTransformer('paraphrase-albert-small-v2')
# model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# 找十个推荐的，顺序，看顺序是否一样


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
    query = data_process.process_query(settings.stackoverflow_text['TextBlob'])
    sentence2vec = Sentence2Vec(docs)
    scores = sentence2vec.score_all(query)
    print(scores)