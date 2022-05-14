# coding=utf-8
# @Author : Eric
import heapq
import pandas as pd

import settings
import tagged_data_process
from sentence2vec import Sentence2Vec
from bm25 import BM25
from vsm import VSM
from word2vec import Word2Vec


def get_rel_list(model_name, api, so_body, so_title, so_tags):
    docs = tagged_data_process.get_tagged_raw_data(api)
    titles = tagged_data_process.get_tagged_title(api)
    tag_list = tagged_data_process.get_tagged_labels(api)

    if model_name == 'bm25':
        select_model = BM25(docs, titles, tag_list)
    elif model_name == 'vsm':
        select_model = VSM(docs, titles, tag_list)
    elif model_name == 'word2vec':
        select_model = Word2Vec(docs, titles, tag_list)
    elif model_name == 'sentence2vec':
        select_model = Sentence2Vec(docs, titles, tag_list)
    else:
        raise Exception('no such model found')
    scores = select_model.score_all(tagged_data_process.process_query(so_body), so_title, so_tags)

    select_num = settings.select_num  # N
    sort_re = list(map(scores.index, heapq.nlargest(select_num, scores)))
    # print(sort_re)

    fpath = settings.tagged_github_filepath[api]
    gi_df = pd.read_csv(fpath)
    gi_df = gi_df.fillna('')
    rel_list = []

    for index in sort_re:
        rel_list.append(gi_df.loc[index]['rel'])
    return rel_list


def calculate_Pk(rel_list, k):
    rel_num = 0
    for i in range(0, k):
        if rel_list[i] > 0:
            rel_num += 1
    Pk = rel_num/k
    return Pk


def calculate_AP(rel_list, api):
    N = settings.select_num
    m = 0
    fpath = settings.tagged_github_filepath[api]
    gi_df = pd.read_csv(fpath)
    gi_df = gi_df.fillna('')
    for index, row in gi_df.iterrows():
        if row['rel'] > 0:
            m += 1
    # print(m)
    score = 0
    for k in range(1, N+1):
        score += calculate_Pk(rel_list, k) * rel_list[k-1]
    if m < N:
        min_mn = m
    else:
        min_mn = N
    score = score/min_mn
    return score


if __name__ == '__main__':
    TextBlob_data = {'api': "TextBlob",
                     'so_body': "<p>I want to analyze sentiment of texts that are written in German. "
                                "I found a lot of tutorials on how to do this with English, "
                                "but I found none on how to apply it to different languages.</p>",
                     'so_title': "Sentiment analysis of non-English texts",
                     'so_tags': "<python><machine-learning><nlp><sentiment-analysis><textblob>"}

    model_list = ['bm25', 'vsm', 'word2vec', 'sentence2vec']
    for model in model_list:
        rel_list = get_rel_list(model, TextBlob_data['api'], TextBlob_data['so_body'], TextBlob_data['so_title'], TextBlob_data['so_tags'])
        score = calculate_AP(rel_list, TextBlob_data['api'])
        print(model + ": " + str(score))


