# coding=utf-8
# @Author : Eric

import settings
import heapq
import random
import matplotlib.pyplot as plt
import matplotlib
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel


class Analyzer(object):
    def __init__(self, docs, scores=None):
        self.docs = docs
        self.scores = scores

    def compute_coherence(self, num_topics, docs2, doc_term_matrix, dictionary):
        ldamodel = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=30, random_state=1)
        print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
        ldacm = CoherenceModel(model=ldamodel, texts=docs2, dictionary=dictionary, coherence='u_mass')
        print(ldacm.get_coherence())
        return ldacm.get_coherence()

    def analyze_lda(self):
        select_num = settings.select_num

        # 获取最大的select_num个分数和他们对应的编号
        print(heapq.nlargest(select_num, self.scores))
        sort_re = list(map(self.scores.index, heapq.nlargest(select_num, self.scores)))
        docs2 = []
        for i in sort_re:
            docs2.append(self.docs[i])
        dictionary = corpora.Dictionary(docs2)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs2]

        x = range(1, 10)
        y = [self.compute_coherence(i, docs2, doc_term_matrix, dictionary) for i in x]
        plt.plot(x, y)
        plt.xlabel('主题数目')
        plt.ylabel('coherence大小')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.title(settings.nlp_api[settings.nlp_choose] + ' bm25 主题-coherence变化情况')
        plt.show()

    def baseline_analyze_lda(self):
        select_num = settings.select_num
        doc_index_list = range(0, len(self.docs))
        random_re = random.sample(doc_index_list, select_num)

        docs2 = []
        for i in random_re:
            docs2.append(self.docs[i])

        dictionary = corpora.Dictionary(docs2)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs2]

        x = range(1, 10)
        y = [self.compute_coherence(i, docs2, doc_term_matrix, dictionary) for i in x]
        plt.plot(x, y)
        plt.xlabel('主题数目')
        plt.ylabel('coherence大小')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.title(settings.nlp_api[settings.nlp_choose] + ' bm25 主题-coherence变化情况')
        plt.show()