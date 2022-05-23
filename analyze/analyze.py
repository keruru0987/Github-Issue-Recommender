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

    scores = [0 for i in range(0, settings.select_num)]

    if model_name == 'bm25':
        docs = tagged_data_process.get_tagged_data(api)
        select_model = BM25(docs, titles, tag_list)
        scores = select_model.score_all(tagged_data_process.process_query_2(so_body), so_title, so_tags)
    elif model_name == 'vsm':
        docs = tagged_data_process.get_tagged_data(api)
        select_model = VSM(docs, titles, tag_list)
        scores = select_model.score_all(tagged_data_process.process_query_2(so_body), so_title, so_tags)
    elif model_name == 'word2vec':
        docs = tagged_data_process.get_tagged_data(api)
        select_model = Word2Vec(docs, titles, tag_list)
        scores = select_model.score_all(tagged_data_process.process_query_2(so_body), so_title, so_tags)
    elif model_name == 'sentence2vec':
        select_model = Sentence2Vec(docs, titles, tag_list)
        scores = select_model.score_all(tagged_data_process.process_query(so_body), so_title, so_tags)  # 不做数据清洗
        # scores = select_model.score_all(tagged_data_process.clean(tagged_data_process.process_query(so_body)), so_title, so_tags)  # 做数据清洗
    else:
        raise Exception('no such model found')
    # scores = select_model.score_all(tagged_data_process.process_query(so_body), so_title, so_tags)
    select_num = settings.select_num  # N
    # sort_re = list(map(scores.index, heapq.nlargest(select_num, scores)))  # 当有重复的数的时候不准
    sort_re_all = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    sort_re = sort_re_all[0:select_num]
    # print(sort_re)

    fpath = settings.tagged_github_filepath[api]
    gi_df = pd.read_csv(fpath)
    gi_df = gi_df.fillna('')
    rel_list = []

    for index in sort_re:
        rel_list.append(gi_df.loc[index]['rel'])
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
def calculate_AP(rel_list, api):
    # 计算Average Precision
    N = settings.select_num  # 推荐的个数
    m = 0  # 总的相关个数
    num_of2 = 0
    fpath = settings.tagged_github_filepath[api]
    gi_df = pd.read_csv(fpath)
    gi_df = gi_df.fillna('')
    for index, row in gi_df.iterrows():
        if row['rel'] > 0:
            m += 1
            if row['rel'] > 1:
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


    '''
    gensim_data = {'api': '',
                   'so_body': '',
                   'so_title': '',
                   'so_tags': ''}'''

    allennlp_data = {'api': 'allennlp',
                     'so_body': '''<p>I was trying to install a library (<code>allennlp</code>) via <code>pip3</code>. But it complained about the PyTorch version. While <code>allennlp</code> requires <code>torch=0.4.0</code> I have <code>torch=0.4.1</code>:</p>

<pre><code>...
Collecting torch==0.4.0 (from allennlp)
  Could not find a version that satisfies the requirement torch==0.4.0 (from allennlp) (from versions: 0.1.2, 0.1.2.post1, 0.4.1)
No matching distribution found for torch==0.4.0 (from allennlp)
</code></pre>

<p><em>Also manually install:</em></p>

<pre><code>pip3 install torch==0.4.0
</code></pre>

<p><em>Doesn't work either:</em></p>

<pre><code>  Could not find a version that satisfies the requirement torch==0.4.0 (from versions: 0.1.2, 0.1.2.post1, 0.4.1)
No matching distribution found for torch==0.4.0
</code></pre>

<p>Same for other versions.</p>

<p>Python is version <code>Python 3.7.0</code> installed via <code>brew</code> on Mac OS.</p>

<p>I remember that some time ago I was able to switch between version <code>0.4.0</code> and <code>0.3.1</code> by using <code>pip3 install torch==0.X.X</code>.</p>

<p>How do I solve this?</p>
                     ''',
                     'so_title': "pip - Installing specific package version does not work",
                     'so_tags': '<python-3.x><pip><homebrew><pytorch><allennlp>'}

    gensim_data = {'api': 'gensim',
                   'so_body': '''<p>According to the <a href="http://radimrehurek.com/gensim/models/word2vec.html" rel="noreferrer">Gensim Word2Vec</a>, I can use the word2vec model in gensim package to calculate the similarity between 2 words.</p>

    <p>e.g.</p>

    <pre><code>trained_model.similarity('woman', 'man') 
    0.73723527
    </code></pre>

    <p>However, the word2vec model fails to predict the sentence similarity. I find out the LSI model with sentence similarity in gensim, but, which doesn't seem that can be combined with word2vec model. The length of corpus of each sentence I have is not very long (shorter than 10 words).  So, are there any simple ways to achieve the goal?</p>''',
                   'so_title': 'How to calculate the sentence similarity using word2vec model of gensim with python',
                   'so_tags': '<python><gensim><word2vec>'}

    nltk_data = {'api': 'nltk',
                 'so_body': "<p>I'm just starting to use NLTK and I don't quite understand how to get a list of words from text."
                            " If I use <code>nltk.word_tokenize()</code>, I get a list of words and punctuation. "
                            "I need only the words instead. How can I get rid of punctuation? "
                            "Also <code>word_tokenize</code> doesn't work with multiple sentences: dots are added to the last word.</p>",
                 'so_title': 'How to get rid of punctuation using NLTK tokenizer?',
                 'so_tags': '<python><nlp><tokenize><nltk>'}

    TextBlob_data = {'api': "TextBlob",
                     'so_body': "<p>I want to analyze sentiment of texts that are written in German. "
                                "I found a lot of tutorials on how to do this with English, "
                                "but I found none on how to apply it to different languages.</p>",
                     'so_title': "Sentiment analysis of non-English texts",
                     'so_tags': "<python><machine-learning><nlp><sentiment-analysis><textblob>"}

    cur_api = nltk_data['api']
    cur_so_body = nltk_data['so_body']
    cur_so_title = nltk_data['so_title']
    cur_so_tags = nltk_data['so_tags']

    model_list = ['bm25', 'vsm', 'word2vec', 'sentence2vec']
    for model in model_list:
        print('当前model： ' + model)
        print('当前api： ' + cur_api)
        rel_list = get_rel_list(model, cur_api, cur_so_body, cur_so_title, cur_so_tags)
        score = calculate_AP(rel_list, cur_api)
        print(model + "分数: " + str(score))
        print('-------------------------------------------------------------------')



