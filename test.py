# coding=utf-8
# @Author : Eric
import pandas
import settings


def gi_num():
    for i in range(0, 7):
        nlp_api = settings.nlp_api[i]
        filepath = 'data/new_issue/' + nlp_api + '.csv'
        df = pandas.read_csv(filepath)
        '''
        for index, row in df.iterrows():
            if row['number'] == 3911:
                print(row['title']+row['body'])
        '''
        print(nlp_api + ' 数据个数：' + str(df.shape[0]))


'''
allennlp     数据个数：2272
gensim       数据个数：952
nltk         数据个数：882
spaCy        数据个数：2748
stanford-nlp 数据个数：968
TextBlob     数据个数：158
Transformers 数据个数：4196
'''


def so_num():
    for i in range(0, 7):
        nlp_api = settings.nlp_api[i]
        filepath = settings.stackoverflow_filepath[nlp_api]
        df = pandas.read_csv(filepath)
        print(nlp_api + ' 数据个数：' + str(df.shape[0]))


'''
allennlp 数据个数：170
gensim 数据个数：2210
nltk 数据个数：6666
spaCy 数据个数：2904
stanford-nlp 数据个数：3261
TextBlob 数据个数：328
Transformers 数据个数：1114
'''


if __name__ == '__main__':
    gi_num()