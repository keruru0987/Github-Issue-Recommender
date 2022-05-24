# coding=utf-8
# @Author : Eric
import heapq
import pandas as pd
import matplotlib.pyplot as plt
import settings
import tagged_data_process
from sentence2vec import Sentence2Vec
from bm25 import BM25
from vsm import VSM
from word2vec import Word2Vec
import time


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


def draw_piri(rel_list, total_rel,model):
    for i in range(0, len(rel_list)):
        if rel_list[i] == 2:
            rel_list[i] = 1

    NUM_ACTUAL_ADDED_ACCT = total_rel
    precs = []
    recalls = []

    for indx, rec in enumerate(rel_list):
        precs.append(sum(rel_list[:indx + 1]) / (indx + 1))
        recalls.append(sum(rel_list[:indx + 1]) / NUM_ACTUAL_ADDED_ACCT)

    plt.plot(recalls, precs, label=model, marker='^', markersize=3)


if __name__ == '__main__':
    test_list = [0 for i in range(0, settings.select_num)]

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
                     'so_tags': '<python-3.x><pip><homebrew><pytorch><allennlp>',
                     'rel_num': 24}

    gensim_data = {'api': 'gensim',
                   'so_body': '''<p>According to the <a href="http://radimrehurek.com/gensim/models/word2vec.html" rel="noreferrer">Gensim Word2Vec</a>, I can use the word2vec model in gensim package to calculate the similarity between 2 words.</p>

    <p>e.g.</p>

    <pre><code>trained_model.similarity('woman', 'man') 
    0.73723527
    </code></pre>

    <p>However, the word2vec model fails to predict the sentence similarity. I find out the LSI model with sentence similarity in gensim, but, which doesn't seem that can be combined with word2vec model. The length of corpus of each sentence I have is not very long (shorter than 10 words).  So, are there any simple ways to achieve the goal?</p>''',
                   'so_title': 'How to calculate the sentence similarity using word2vec model of gensim with python',
                   'so_tags': '<python><gensim><word2vec>',
                   'rel_num': 26}

    nltk_data = {'api': 'nltk',
                 'so_body': "<p>I'm just starting to use NLTK and I don't quite understand how to get a list of words from text."
                            " If I use <code>nltk.word_tokenize()</code>, I get a list of words and punctuation. "
                            "I need only the words instead. How can I get rid of punctuation? "
                            "Also <code>word_tokenize</code> doesn't work with multiple sentences: dots are added to the last word.</p>",
                 'so_title': 'How to get rid of punctuation using NLTK tokenizer?',
                 'so_tags': '<python><nlp><tokenize><nltk>',
                 'rel_num': 26}

    spaCy_data = {'api': 'spaCy',
                  'so_body': '''<p>what is difference between <code>spacy.load('en_core_web_sm')</code> and <code>spacy.load('en')</code>? <a href="https://stackoverflow.com/questions/50487495/what-is-difference-between-en-core-web-sm-en-core-web-mdand-en-core-web-lg-mod">This link</a> explains different model sizes. But i am still not clear how <code>spacy.load('en_core_web_sm')</code> and <code>spacy.load('en')</code> differ</p>

<p><code>spacy.load('en')</code> runs fine for me. But the <code>spacy.load('en_core_web_sm')</code> throws error</p>

<p>i have installed <code>spacy</code>as below. when i go to jupyter notebook and run command <code>nlp = spacy.load('en_core_web_sm')</code> I get the below error </p>''',
                  'so_title': "spacy Can't find model 'en_core_web_sm' on windows 10 and Python 3.5.3 :: Anaconda custom (64-bit)",
                  'so_tags': '<python><python-3.x><nlp><spacy>',
                  'rel_num': 36}

    stanford_nlp_data = {'api': 'stanford-nlp',
                         'so_body': '''<p>How can I split a text or paragraph into sentences using <a href="http://nlp.stanford.edu/software/lex-parser.shtml" rel="noreferrer">Stanford parser</a>?</p>

<p>Is there any method that can extract sentences, such as <code>getSentencesFromString()</code> as it's provided for <a href="http://stanfordparser.rubyforge.org/" rel="noreferrer">Ruby</a>?</p>''',
                         'so_title': "How can I split a text into sentences using the Stanford parser?",
                         'so_tags': '<java><parsing><artificial-intelligence><nlp><stanford-nlp>',
                         'rel_num': 28}

    TextBlob_data = {'api': "TextBlob",
                     'so_body': "<p>I want to analyze sentiment of texts that are written in German. "
                                "I found a lot of tutorials on how to do this with English, "
                                "but I found none on how to apply it to different languages.</p>",
                     'so_title': "Sentiment analysis of non-English texts",
                     'so_tags': "<python><machine-learning><nlp><sentiment-analysis><textblob>",
                     'rel_num': 40}

    Transformers_data = {'api': "Transformers",
                         'so_body': '''<p>I fine-tuned a pretrained BERT model in Pytorch using huggingface transformer. All the training/validation is done on a GPU in cloud.</p>
<p>At the end of the training, I save the model and tokenizer like below:</p>
<pre><code>best_model.save_pretrained('./saved_model/')
tokenizer.save_pretrained('./saved_model/')
</code></pre>
<p>This creates below files in the <code>saved_model</code> directory:</p>
<pre><code>config.json
added_token.json
special_tokens_map.json
tokenizer_config.json
vocab.txt
pytorch_model.bin
</code></pre>
<p>Now, I download the <code>saved_model</code> directory in my computer and want to load the model and tokenizer. I can load the model like below</p>
<p><code>model = torch.load('./saved_model/pytorch_model.bin',map_location=torch.device('cpu'))</code></p>
<p>But how do I load the tokenizer? I am new to pytorch and not sure because there are multiple files. Probably I am not saving the model in the right way?</p>
''',
                         'so_title': "How to load the saved tokenizer from pretrained model",
                         'so_tags': "<machine-learning><pytorch><huggingface-transformers>",
                         'rel_num': 26}

    '''
    cur_api = Transformers_data['api']
    cur_so_body = Transformers_data['so_body']
    cur_so_title = Transformers_data['so_title']
    cur_so_tags = Transformers_data['so_tags']
    cur_rel_num = Transformers_data['rel_num']

    # model_list = ['bm25', 'vsm', 'word2vec', 'sentence2vec']
    model_list = ['sentence2vec']
    for model in model_list:
        print('当前model： ' + model)
        print('当前api： ' + cur_api)
        rel_list = get_rel_list(model, cur_api, cur_so_body, cur_so_title, cur_so_tags)
        score = calculate_AP(rel_list, cur_api)
        print(model + "分数: " + str(score))
        draw_piri(rel_list, cur_rel_num, model)
        print('-------------------------------------------------------------------')

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title(cur_api + ' P(i) vs. r(i) for Increasing $i$ for AP@20')
    plt.legend()
    plt.show()
    '''

    # api和model双层循环
    # api_data_list = [allennlp_data, gensim_data, nltk_data, spaCy_data, stanford_nlp_data, TextBlob_data, Transformers_data]
    api_data_list = [TextBlob_data]

    for sel_api in api_data_list:
        cur_api = sel_api['api']
        cur_so_body = sel_api['so_body']
        cur_so_title = sel_api['so_title']
        cur_so_tags = sel_api['so_tags']
        cur_rel_num = sel_api['rel_num']
        # model_list = ['bm25', 'vsm', 'word2vec', 'sentence2vec']
        model_list = ['sentence2vec']
        for model in model_list:
            print('当前model： ' + model)
            print('当前api： ' + cur_api)
            time_start = time.time()
            rel_list = get_rel_list(model, cur_api, cur_so_body, cur_so_title, cur_so_tags)
            time_end = time.time()
            time_sum = time_end - time_start
            score = calculate_AP(rel_list, cur_api)
            print("分数: " + str(score))
            print("run time:" + str(time_sum))
            draw_piri(rel_list, cur_rel_num, model)
            print('-------------------------------------------------------------------')

        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title(cur_api + ' P(i) vs. r(i) for Increasing $i$ for AP@20')
        plt.legend()
        # plt.show()


