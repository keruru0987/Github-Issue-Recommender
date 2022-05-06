# coding=utf-8
# @Author : Eric
# 数据的获取以及处理

import settings
import string
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
from markdown import markdown


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
    fpath = settings.new_github_filepath[nlp_api]

    df = pd.read_csv(fpath)
    df = df.fillna('')
    texts = []

    # 去掉其中的pull,保留issue
    for index, row in df.iterrows():
        texts.append(row['body'])

    non_code_texts = []
    for i in range(0, len(texts)):
        # 消除代码块
        pattern = r'```(.*\n)*```'
        text = re.sub(pattern, '', texts[i])
        # 消除命令行指令
        pattern2 = r'> > > .*'
        text = re.sub(pattern2, '', text)
        # 消除报错 一般两行
        pattern3 = r'File .*\n.*'
        text = re.sub(pattern3, '', text)
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


def get_raw_data(api=''):
    # 直接将没有清洗的字符串数组返回，s2v用
    # 如果指定了api，则按照指定的api来，否则选择setting中默认的api
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    if api != '':
        nlp_api = api
    print('当前api为：' + nlp_api)

    # 数据的获取以及处理
    fpath = settings.new_github_filepath[nlp_api]

    df = pd.read_csv(fpath)
    df = df.fillna('')
    texts = []

    # 去掉其中的pull,保留issue
    for index, row in df.iterrows():
        texts.append(row['body'])

    non_code_texts = []
    for i in range(0, len(texts)):
        # 消除代码块
        pattern = r'```(.*\n)*```'
        text = re.sub(pattern, '', texts[i])
        # 消除命令行指令
        pattern2 = r'> > > .*'
        text = re.sub(pattern2, '', text)
        # 消除报错 一般两行
        pattern3 = r'File .*\n.*'
        text = re.sub(pattern3, '', text)
        html = markdown(text)
        non_code_texts.append(BeautifulSoup(html, 'html.parser').get_text())

    return non_code_texts


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


def process_query(query):
    pattern = r'<code>(.*\n)*</code>'
    text = re.sub(pattern, '', query)
    re_query = BeautifulSoup(text, 'html.parser').get_text()
    return re_query


if __name__ == '__main__':
    t = '''<p>I am trying to analyze open-ended questions with Polarity and subjectivity. So what I want to achieve is to upload the CSV file, then add new columns one for polarity, subjectively, negative or positive column and here what I did:</p>
<pre><code>from textblob import TextBlob
import pandas as pd 
import numpy as np

# Load the data
from google.colab import files
uploaded = files.upload()


text = open(uploaded) // *this did not work so I just replaced uploaded with the name of the file and the path... this is not what I want. I hoped to get the file name here once uploaded in the first step and refer it to the file name in this line.* //
text = text.read()
blob = TextBlob(text)

with open('text.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        data = data.append(row[row], blob.polarity,blob.subjectivity)
        print(data) 
</code></pre>
<p>And I want to print the data in an external file. but could not figure that out. how can I do it, and thank you in advance.</p>'''
    b = process_query(t)
    # print(get_data())
    a = get_data('TextBlob')
    print(a)
