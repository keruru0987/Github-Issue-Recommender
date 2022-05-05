# coding=utf-8
# @Author : Eric
# 数据的获取以及处理

import settings
import string
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    # 数据清洗
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def get_data(api=''):
    # 如果指定了api，则按照指定的api来，否则选择setting中默认的api
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    if api != '':
        nlp_api = api
    print('当前api为：' + nlp_api)
    # 数据的获取以及处理
    fpath = settings.github_filepath[nlp_api]

    df = pd.read_json(fpath)
    df = df.fillna('')
    texts = []

    # 去掉其中的pull,保留issue
    for index, row in df.iterrows():
        if row['pull_request'] == '':
            texts.append(row['body'])

    docs = []
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    for sentence in texts:
        non_sentence = sentence.replace('\r\n', ' ')
        nonn_sentence = non_sentence.replace('\n', ' ')
        sentence_words = clean(nonn_sentence).split(' ')  # 进行数据清洗

        tokens = []
        for word in sentence_words:
            wordd = re.sub(r, '', word)
            if not wordd.isalpha():
                continue
            if len(wordd) > 10 or wordd == '' or len(word) < 2:
                continue
            else:
                tokens.append(wordd)
        docs.append(tokens)
    return docs


def get_query():
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    print('当前api为：' + nlp_api)
    stack_text = settings.stackoverflow_text[nlp_api]
    query = []
    for word in clean(stack_text).split(' '):
        # if word in stopwords:
        # continue
        if not word.isalpha():
            continue
        if len(word) > 10 or len(word) < 2:
            continue
        else:
            query.append(word)
    return query


if __name__ == '__main__':
    # print(get_data())
    a = get_data('allennlp')
    print(a)
