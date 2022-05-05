# coding=utf-8
# @Author : Eric
import heapq

import settings
import pandas as pd
import data_process
from word2vec import Word2Vec
from sentence2vec import Sentence2Vec


class GIRecommend(object):
    def __init__(self, api, title, body, tags):
        self.api = api
        self.so_title = title
        self.so_body = body
        self.so_tags = tags

    def recommend(self):
        docs = data_process.get_data(self.api)
        s2v = Sentence2Vec(docs)
        scores = s2v.score_all(self.so_body)
        # print(scores)

        select_num = settings.select_num
        sort_re = list(map(scores.index, heapq.nlargest(select_num, scores)))
        # print(sort_re)
        fpath = settings.github_filepath[self.api]
        gi_df = pd.read_json(fpath)
        gi_df = gi_df.fillna('')

        # issue所在的索引
        change_list = []
        for index, row in gi_df.iterrows():
            if row['pull_request'] == '':
                change_list.append(index)

        result_gi = []

        for index in sort_re:
            link = settings.api_prelink[self.api] + str(gi_df.loc[change_list[index]]['number'])
            information = [link, gi_df.loc[change_list[index]]['title'], gi_df.loc[change_list[index]]['body'], gi_df.loc[change_list[index]]['number']]
            result_gi.append(information)

        return result_gi


if __name__ == '__main__':
    gi_recommend = GIRecommend('allennlp','hello','hello world!','<allennlp>')
    gi = gi_recommend.recommend()
    for inf in gi:
        print(inf)