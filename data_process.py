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
        pattern = r'```(.*\n)*?```'
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


def get_title(api=''):
    # 获取标题数据
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    if api != '':
        nlp_api = api
    print('当前api为：' + nlp_api)

    # 数据的获取以及处理
    fpath = settings.new_github_filepath[nlp_api]

    df = pd.read_csv(fpath)
    df = df.fillna('')
    title_texts = []

    # 去掉其中的pull,保留issue
    for index, row in df.iterrows():
        title_texts.append(row['title'])
    return title_texts


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
        html = markdown(text)
        non_code_texts.append(BeautifulSoup(html, 'html.parser').get_text())

    return non_code_texts


def get_gi_code(api=''):
    # 有问题
    # 顺便检查其他正则表达式
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

    for index, row in df.iterrows():
        texts.append(row['body'])

    code_list = []
    for i in range(0, len(texts)):
        pattern = r'```(.*\n)*?```'
        text = texts[i]
        codes = re.findall(pattern, text)
        code_list.append(codes)

    return code_list


def get_labels(api=''):
    # 获取label数据,返回的是一个二维数组
    nlp_choose = settings.nlp_choose  # 选择要研究的NLP库
    nlp_api = settings.nlp_api[nlp_choose]
    if api != '':
        nlp_api = api
    print('当前api为：' + nlp_api)

    # 数据的获取以及处理
    fpath = settings.new_github_filepath[nlp_api]

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
    # 删代码
    pattern = r'<code>(.*\n)*?</code>'
    text = re.sub(pattern, '', query)
    # 删图片
    pattern2 = r'<img.*?>'
    text = re.sub(pattern2, '', text)
    re_query = BeautifulSoup(text, 'html.parser').get_text()
    return re_query


def get_query_code(query):
    code_pattern = r'<code>(.*\n)*?</code>'
    codes = re.findall(code_pattern, query)
    return codes


def alter_pic_size(query):
    # 修改显示图片长度为1000px
    pattern = r'<img'
    text = re.sub(pattern, "<img style='width: 1000px'", query)
    return text


if __name__ == '__main__':

#     jyt = '''I love TextBlob, thank you so much for making this awesome Python tool :+1:
#
# I am wondering if there is a solution to a tokenization issue I'm seeing.  Here's some example code with an excerpt from Game of Thrones to demonstrate the issue:
#
# ```
# In [1]: from textblob import TextBlob
# In [2]: text = TextBlob('“We should start back,” Gared urged as the woods began to grow dark around them. “The wildlings are dead.” “Do the dead frighten you?” Ser Waymar Royce asked with just the hint of a smile. Gared did not rise to the bait. He was an old man, past fifty, and he had seen the lordlings come and go. “Dead is dead,” he said. “We have no business with the dead.” “Are they dead?” Royce asked softly.')
# In [3]: text.sentences
# ```
#
# And here's the ouput when I call the sentences attribute:
#
# ```
# Out[3]:
# [Sentence("“We should start back,” Gared urged as the woods began to grow dark around them."),
#  Sentence("“The wildlings are dead.” “Do the dead frighten you?” Ser Waymar Royce asked with just the hint of a smile."),
#  Sentence("Gared did not rise to the bait."),
#  Sentence("He was an old man, past fifty, and he had seen the lordlings come and go."),
#  Sentence("“Dead is dead,” he said."),
#  Sentence("“We have no business with the dead.” “Are they dead?” Royce asked softly.")]
# ```
#
# The issue here is that TextBlob is tokenizing sentences that run together with quotations as a single sentence. The second "sentence" above demonstrates this:
#
# ```
# Sentence("“The wildlings are dead.” “Do the dead frighten you?” Ser Waymar Royce asked with just the hint of a smile.")
# ```
#
# should instead be:
#
# ```
# Sentence("“The wildlings are dead.”)
# Sentence(“Do the dead frighten you?” Ser Waymar Royce asked with just the hint of a smile.")
# ```
#
# The same is the case for the last example sentence I've shown:
#
# ```
# Sentence("“We have no business with the dead.” “Are they dead?” Royce asked softly.")
# ```
#
# It seems that TextBlob does not Tokenize a sentence if it appears in quotes.  In other words, `“We have no business with the dead.”` is its own sentence, but TextBlob tokenizes the sentence such that it also includes the phrases that follow: `“Are they dead?” Royce asked softly."`
#
# Is there a way to avoid this, that is to force TextBlob to treat an occurrence of `."` as the end of a sentence?
# '''
#     pattern = r'```(.*\n)*?```'
#     hh = re.sub(pattern, 'aaa', jyt)
#     codes = re.findall(pattern, jyt)

    tt = get_raw_data('TextBlob')
    mm = get_gi_code('TextBlob')
    # m = get_labels('TextBlob')
    t = '''<p>I am experiencing some problems using the TextBlob library. I'm trying to run a very simple piece of code like this:</p>
<pre><code>from textblob import TextBlob
text = 'this is just a test'
blob = TextBlob(text)
blob.detect_language()
</code></pre>
<p>And it continually gives me this error:</p>
<pre><code>/usr/lib/python3.7/urllib/request.py in http_error_default(self, req, fp, code, msg, hdrs)
    647 class HTTPDefaultErrorHandler(BaseHandler):
    648     def http_error_default(self, req, fp, code, msg, hdrs):
--&gt; 649         raise HTTPError(req.full_url, code, msg, hdrs, fp)
    650 
    651 class HTTPRedirectHandler(BaseHandler):

HTTPError: HTTP Error 404: Not Found
</code></pre>
<p>What is the problem? I have tried it on several devices and it gives me the same error everytime.</p>
<p>Thanks!</p>'''
    c = get_query_code(t)
    b = process_query(t)
    # print(get_data())
    a = get_data('TextBlob')
    print(a)
