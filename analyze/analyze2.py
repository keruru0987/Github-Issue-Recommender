# coding=utf-8
# @Author : Eric
# coding=utf-8
# @Author : Eric
import heapq
import pandas as pd
import settings
import tagged_data_process2
from sentence2vec import Sentence2Vec
from bm25 import BM25
from vsm import VSM
from word2vec import Word2Vec
import time
import allennlp_question
import gensim_question
import nltk_question
import spacy_question
import stanfordnlp_question
import textblob_question
import transformers_question


def get_rel_list(model_name, api, so_body, so_title, so_tags, cur_rel):
    docs = tagged_data_process2.get_tagged_raw_data(api)
    titles = tagged_data_process2.get_tagged_title(api)
    tag_list = tagged_data_process2.get_tagged_labels(api)

    scores = [0 for i in range(0, settings.new_select_num)]

    if model_name == 'bm25':
        docs = tagged_data_process2.get_tagged_data(api)
        select_model = BM25(docs, titles, tag_list)
        scores = select_model.score_all(tagged_data_process2.process_query_2(so_body), so_title, so_tags)
    elif model_name == 'vsm':
        docs = tagged_data_process2.get_tagged_data(api)
        select_model = VSM(docs, titles, tag_list)
        scores = select_model.score_all(tagged_data_process2.process_query_2(so_body), so_title, so_tags)
    elif model_name == 'word2vec':
        docs = tagged_data_process2.get_tagged_data(api)
        select_model = Word2Vec(docs, titles, tag_list)
        scores = select_model.score_all(tagged_data_process2.process_query_2(so_body), so_title, so_tags)
    elif model_name == 'sentence2vec':
        select_model = Sentence2Vec(docs, titles, tag_list)
        scores = select_model.score_all(tagged_data_process2.process_query(so_body), so_title, so_tags)  # 不做数据清洗
        # scores = select_model.score_all(tagged_data_process.clean(tagged_data_process.process_query(so_body)), so_title, so_tags)  # 做数据清洗
    else:
        raise Exception('no such model found')
    # scores = select_model.score_all(tagged_data_process.process_query(so_body), so_title, so_tags)
    select_num = settings.new_select_num  # N
    # sort_re = list(map(scores.index, heapq.nlargest(select_num, scores)))  # 当有重复的数的时候不准
    sort_re_all = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    sort_re = sort_re_all[0:select_num]
    # print(sort_re)
    print(sort_re)

    fpath = settings.new_tagged_github_filepath[api]
    gi_df = pd.read_csv(fpath)
    gi_df = gi_df.fillna('')
    rel_list = []

    for index in sort_re:
        rel_list.append(gi_df.loc[index][cur_rel])
    return rel_list


def calculate_Pk(rel_list, k):
    # 前k个推荐中的正确率
    rel_num = 0
    for i in range(0, k):
        if rel_list[i] > 0:
            rel_num += 1
    Pk = rel_num/k
    return Pk


# 需要修改
def calculate_AP(rel_list, api, cur_rel):
    # 计算Average Precision
    N = settings.new_select_num  # 推荐的个数
    m = 0  # 总的相关个数
    num_of2 = 0
    fpath = settings.new_tagged_github_filepath[api]
    gi_df = pd.read_csv(fpath)
    gi_df = gi_df.fillna('')

    # 修改处
    for index, row in gi_df.iterrows():
        if row[cur_rel] > 0:
            m += 1
            if row[cur_rel] > 1:
                num_of2 += 1
    # print(m)
    score = 0
    for k in range(1, N+1):
        score += calculate_Pk(rel_list, k) * rel_list[k-1]
    if m < N:
        min_mn = m
    else:
        min_mn = N
    score = score/(min_mn + num_of2)
    return score


if __name__ == '__main__':
    test_list = [0 for i in range(0, settings.select_num)]


    # api和model双层循环
    # api_data_list = [allennlp_data1, gensim_data1, nltk_data1, spaCy_data1, stanford_nlp_data1, TextBlob_data1, Transformers_data1]

    # api_data_list = [allennlp_question.allennlp_data1, allennlp_question.allennlp_data2, allennlp_question.allennlp_data3,
    #                  allennlp_question.allennlp_data4, allennlp_question.allennlp_data5, allennlp_question.allennlp_data6,
    #                  allennlp_question.allennlp_data7, allennlp_question.allennlp_data8, allennlp_question.allennlp_data9,
    #                  allennlp_question.allennlp_data10]

    # api_data_list = [gensim_question.gensim_data1, gensim_question.gensim_data2, gensim_question.gensim_data3,
    #                  gensim_question.gensim_data4, gensim_question.gensim_data5, gensim_question.gensim_data6,
    #                  gensim_question.gensim_data7, gensim_question.gensim_data8, gensim_question.gensim_data9,
    #                  gensim_question.gensim_data10 ]

    # api_data_list = [nltk_question.nltk_data1, nltk_question.nltk_data2, nltk_question.nltk_data3,
    #                  nltk_question.nltk_data4, nltk_question.nltk_data5, nltk_question.nltk_data6,
    #                  nltk_question.nltk_data7, nltk_question.nltk_data8, nltk_question.nltk_data9,
    #                  nltk_question.nltk_data10]

    # api_data_list = [spacy_question.spaCy_data1, spacy_question.spaCy_data2, spacy_question.spaCy_data3,
    #                  spacy_question.spaCy_data4, spacy_question.spaCy_data5, spacy_question.spaCy_data6,
    #                  spacy_question.spaCy_data7, spacy_question.spaCy_data8, spacy_question.spaCy_data9,
    #                  spacy_question.spaCy_data10]

    # api_data_list = [stanfordnlp_question.stanford_nlp_data1, stanfordnlp_question.stanford_nlp_data2, stanfordnlp_question.stanford_nlp_data3,
    #                  stanfordnlp_question.stanford_nlp_data4, stanfordnlp_question.stanford_nlp_data5, stanfordnlp_question.stanford_nlp_data6,
    #                  stanfordnlp_question.stanford_nlp_data7, stanfordnlp_question.stanford_nlp_data8, stanfordnlp_question.stanford_nlp_data9,
    #                  stanfordnlp_question.stanford_nlp_data10]

    # api_data_list = [textblob_question.TextBlob_data1, textblob_question.TextBlob_data2, textblob_question.TextBlob_data3,
    #                  textblob_question.TextBlob_data4, textblob_question.TextBlob_data5, textblob_question.TextBlob_data6,
    #                  textblob_question.TextBlob_data7, textblob_question.TextBlob_data8, textblob_question.TextBlob_data9,
    #                  textblob_question.TextBlob_data10]

    api_data_list = [transformers_question.Transformers_data1, transformers_question.Transformers_data2, transformers_question.Transformers_data3,
                     transformers_question.Transformers_data4, transformers_question.Transformers_data5, transformers_question.Transformers_data6,
                     transformers_question.Transformers_data7, transformers_question.Transformers_data8, transformers_question.Transformers_data9,
                     transformers_question.Transformers_data10]

    i = 1
    scores = []
    for sel_api in api_data_list:
        cur_rel = 'rel' + str(i)
        print(cur_rel)
        cur_api = sel_api['api']
        cur_so_body = sel_api['so_body']
        cur_so_title = sel_api['so_title']
        cur_so_tags = sel_api['so_tags']
        cur_rel_num = sel_api['rel_num']
        # model_list = ['bm25', 'vsm', 'word2vec', 'sentence2vec']

        model_list = ['sentence2vec']
        # model_list = ['word2vec']
        # model_list = ['vsm']
        # model_list = ['bm25']
        for model in model_list:
            print("当前i: ", i)
            print('当前model： ' + model)
            print('当前api： ' + cur_api)
            time_start = time.time()
            rel_list = get_rel_list(model, cur_api, cur_so_body, cur_so_title, cur_so_tags, cur_rel)
            print('rel_list: ', rel_list)
            time_end = time.time()
            time_sum = time_end - time_start
            score = calculate_AP(rel_list, cur_api, cur_rel)
            scores.append(score)
            print("分数: " + str(score))
            print("run time:" + str(time_sum))
            print('-------------------------------------------------------------------')
        i += 1

    sum = 0
    for score in scores:
        sum += score
    average = sum/10
    print('average: ', average)
