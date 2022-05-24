# coding=utf-8
# @Author : Eric
import settings
import string
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
from markdown import markdown


def get_tagged_title(api=''):
    # 获取标题数据
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    if api != '':
        nlp_api = api
    # print('当前api为：' + nlp_api)

    # 数据的获取以及处理
    fpath = settings.tagged_github_filepath[nlp_api]

    df = pd.read_csv(fpath)
    df = df.fillna('')
    title_texts = []

    # 去掉其中的pull,保留issue
    for index, row in df.iterrows():
        title_texts.append(row['title'])
    return title_texts


def get_tagged_raw_data(api=''):
    # 直接将没有清洗的字符串数组返回，s2v用
    # 如果指定了api，则按照指定的api来，否则选择setting中默认的api
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    if api != '':
        nlp_api = api
    # print('获取tagged数据，当前api为：' + nlp_api)

    # 数据的获取以及处理
    fpath = settings.tagged_github_filepath[nlp_api]

    df = pd.read_csv(fpath)
    df = df.fillna('')
    texts = []

    for index, row in df.iterrows():
        texts.append(row['body'])

    non_code_texts = []
    for i in range(0, len(texts)):
        # 消除代码块
        pattern = r'```(.*\n)*?```'
        text = re.sub(pattern, '', texts[i])
        # 消除命令行指令
        pattern2 = r'> > > .*'
        text = re.sub(pattern2, '', text)
        # 消除报错 一般两行
        pattern3 = r'File .*\n.*'
        text = re.sub(pattern3, '', text)
        # 消除说明
        pattern4 = r'<!--.*-->'
        text = re.sub(pattern4, '', text)
        html = markdown(text)
        non_code_texts.append(BeautifulSoup(html, 'html.parser').get_text())

    return non_code_texts


def get_tagged_labels(api=''):
    # 获取label数据,返回的是一个二维数组
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    if api != '':
        nlp_api = api
    # print('当前api为：' + nlp_api)

    # 数据的获取以及处理
    fpath = settings.tagged_github_filepath[nlp_api]

    df = pd.read_csv(fpath)
    df = df.fillna('')
    labels = []

    for index, row in df.iterrows():
        cur_labels = []
        label_list = eval(row['labels'])
        for label_dict in label_list:
            cur_labels.append(label_dict['name'])
        labels.append(cur_labels)
    return labels


def process_query(query):
    # 删代码
    pattern = r'<code>(.*\n)*?</code>'
    text = re.sub(pattern, '', query)
    # 删图片
    pattern2 = r'<img.*?>'
    text = re.sub(pattern2, '', text)
    # 删链接
    re_query = BeautifulSoup(text, 'html.parser').get_text()
    return re_query


def process_query_2(old_query):
    # 三个老算法的query处理
    query = []
    for word in clean(old_query).split(' '):
        # if word in stopwords:
        # continue
        if not word.isalpha():
            continue
        if len(word) > 10 or len(word) < 2:
            continue
        else:
            query.append(word)
    return query


def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    # 数据清洗
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def get_tagged_data(api=''):
    # 如果指定了api，则按照指定的api来，否则选择setting中默认的api
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    if api != '':
        nlp_api = api
    # print('当前api为：' + nlp_api)

    # 数据的获取以及处理
    fpath = settings.tagged_github_filepath[nlp_api]

    df = pd.read_csv(fpath)
    df = df.fillna('')
    texts = []

    # 去掉其中的pull,保留issue
    for index, row in df.iterrows():
        texts.append(row['body'])

    non_code_texts = []
    for i in range(0, len(texts)):
        # 消除代码块
        pattern = r'```(.*\n)*?```'
        text = re.sub(pattern, '', texts[i])
        # 消除命令行指令
        pattern2 = r'> > > .*'
        text = re.sub(pattern2, '', text)
        # 消除报错 一般两行
        pattern3 = r'File .*\n.*'
        text = re.sub(pattern3, '', text)
        # 消除说明
        pattern4 = r'<!--.*-->'
        text = re.sub(pattern4, '', text)
        html = markdown(text)
        non_code_texts.append(BeautifulSoup(html, 'html.parser').get_text())

    docs = []
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    for sentence in non_code_texts:
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