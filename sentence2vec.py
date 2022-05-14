# coding=utf-8
# @Author : Eric

import numpy as np
import data_process
from scipy.linalg import norm
from sentence_transformers import SentenceTransformer, util
import settings
import sys
import time
model = SentenceTransformer('all-MiniLM-L6-v2')
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
    def __init__(self, docs, titles, tags_list):
        self.docs = docs
        self.titles = titles
        self.tags_list = tags_list
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

    def score_all(self, so_body_text, so_title_text, so_tags):
        all_progress = len(self.docs)*4  # 进度条
        count_progress = 0
        scores1 = []
        # query = self.word2sentence(sequence)
        v_query_body = model.encode(so_body_text)
        for doc in self.docs:
            # 如果doc为空，直接记为0分
            if len(doc) > 0:
                # doc_sentence = self.word2sentence(doc)
                v_doc = model.encode(doc)
                scores1.append(self.score_cos(v_query_body, v_doc))
            else:
                scores1.append(0)
            count_progress += 1
            progress = count_progress/all_progress*100
            progress = round(progress, 1)
            print("\r", end="")
            print('进度：{}%'.format(progress), "▋" * (int(round(progress)) // 2), end="")
            sys.stdout.flush()
            time.sleep(0.00001)

        scores2 = []
        v_query_title = model.encode(so_title_text)
        for title in self.titles:
            if len(title) > 0:
                v_gi_title = model.encode(title)
                scores2.append(self.score_cos(v_query_title, v_gi_title))
            else:
                scores2.append(0)
            count_progress += 1
            progress = count_progress / all_progress * 100
            progress = round(progress, 1)
            print("\r", end="")
            print('进度：{}%'.format(progress),"▋" * (int(round(progress)) // 2), end="")
            sys.stdout.flush()
            time.sleep(0.00001)

        tag_counts = []
        for tags in self.tags_list:
            count = 0
            for tag in tags:
                if tag.lower() in so_tags.lower():
                    count += 1
            tag_counts.append(count)
            count_progress += 1
            progress = count_progress / all_progress * 100
            progress = round(progress, 1)
            print("\r", end="")
            print('进度：{}%'.format(progress), "▋" * (int(round(progress)) // 2), end="")
            sys.stdout.flush()
            time.sleep(0.00001)


        max_tag_count = max(tag_counts)
        # 防止全为0的情况
        if max_tag_count == 0:
            max_tag_count = 1

        scores_final = []
        if len(scores1) != len(scores2):
            raise Exception('title body num not match')
        # 计算最终得分
        for i in range(len(scores1)):
            k = 1.5  # 标题系数
            t = 1    # tag系数
            score = (1 + t*tag_counts[i]/max_tag_count)/(1+t) * (scores2[i]*k + scores1[i])/(k+1)
            scores_final.append(score)
            count_progress += 1
            progress = count_progress / all_progress * 100
            progress = round(progress, 1)
            print("\r", end="")
            print('进度：{}%'.format(progress), "▋" * (int(round(progress)) // 2), end="")
            sys.stdout.flush()
            time.sleep(0.00001)
        print('')

        return scores_final


if __name__ == '__main__':
    docs = data_process.get_raw_data('TextBlob')
    titles = data_process.get_title('TextBlob')
    tags_list = data_process.get_labels('TextBlob')
    query = data_process.process_query(settings.stackoverflow_text['TextBlob'])

    sentence2vec = Sentence2Vec(docs, titles, tags_list)
    scores = sentence2vec.score_all(query, 'Polarity and subjectively from text', '<python><dataframe><enhancement><textblob>')
    print(scores)