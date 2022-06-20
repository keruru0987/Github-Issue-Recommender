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

# 数据accept/close比例


def so_accept():
    for i in range(0, 7):
        count = 0
        nlp_api = settings.nlp_api[i]
        filepath = settings.stackoverflow_filepath[nlp_api]
        df = pandas.read_csv(filepath)
        df = df.fillna(0)
        for index, row in df.iterrows():
            if row['AcceptedAnswerId'] != 0:
                count += 1
        print(nlp_api + ' 解决数据个数：' + str(count))
'''
allennlp 解决数据个数：52
gensim 解决数据个数：976
nltk 解决数据个数：3075
spaCy 解决数据个数：1246
stanford-nlp 解决数据个数：1302
TextBlob 解决数据个数：112
Transformers 解决数据个数：353
'''


def gi_close():
    for i in range(0, 7):
        count = 0
        nlp_api = settings.nlp_api[i]
        filepath = 'data/new_issue/' + nlp_api + '.csv'
        df = pandas.read_csv(filepath)

        for index, row in df.iterrows():
            if row['state'] == 'closed':
                count += 1

        print(nlp_api + ' 解决的数据个数：' + str(count))
'''
allennlp 解决的数据个数：2187
gensim 解决的数据个数：684
nltk 解决的数据个数：675
spaCy 解决的数据个数：2675
stanford-nlp 解决的数据个数：774
TextBlob 解决的数据个数：76
Transformers 解决的数据个数：3872
'''


if __name__ == '__main__':
    # so_accept()
    # gi_close()
    # filepath = 'data/new_issue/allissue.csv'
    filepath = 'data/stackoverflow/allso.csv'
    df = pandas.read_csv(filepath)
    print(' 数据个数：' + str(df.shape[0]))