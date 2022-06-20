# coding=utf-8
# @Author : Eric

allennlp_data1 = {'api': 'allennlp',
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

gensim_data1 = {'api': 'gensim',
                   'so_body': '''<p>According to the <a href="http://radimrehurek.com/gensim/models/word2vec.html" rel="noreferrer">Gensim Word2Vec</a>, I can use the word2vec model in gensim package to calculate the similarity between 2 words.</p>

    <p>e.g.</p>

    <pre><code>trained_model.similarity('woman', 'man') 
    0.73723527
    </code></pre>

    <p>However, the word2vec model fails to predict the sentence similarity. I find out the LSI model with sentence similarity in gensim, but, which doesn't seem that can be combined with word2vec model. The length of corpus of each sentence I have is not very long (shorter than 10 words).  So, are there any simple ways to achieve the goal?</p>''',
                   'so_title': 'How to calculate the sentence similarity using word2vec model of gensim with python',
                   'so_tags': '<python><gensim><word2vec>',
                   'rel_num': 26}

nltk_data1 = {'api': 'nltk',
                 'so_body': "<p>I'm just starting to use NLTK and I don't quite understand how to get a list of words from text."
                            " If I use <code>nltk.word_tokenize()</code>, I get a list of words and punctuation. "
                            "I need only the words instead. How can I get rid of punctuation? "
                            "Also <code>word_tokenize</code> doesn't work with multiple sentences: dots are added to the last word.</p>",
                 'so_title': 'How to get rid of punctuation using NLTK tokenizer?',
                 'so_tags': '<python><nlp><tokenize><nltk>',
                 'rel_num': 26}

spaCy_data1 = {'api': 'spaCy',
                  'so_body': '''<p>what is difference between <code>spacy.load('en_core_web_sm')</code> and <code>spacy.load('en')</code>? <a href="https://stackoverflow.com/questions/50487495/what-is-difference-between-en-core-web-sm-en-core-web-mdand-en-core-web-lg-mod">This link</a> explains different model sizes. But i am still not clear how <code>spacy.load('en_core_web_sm')</code> and <code>spacy.load('en')</code> differ</p>

<p><code>spacy.load('en')</code> runs fine for me. But the <code>spacy.load('en_core_web_sm')</code> throws error</p>

<p>i have installed <code>spacy</code>as below. when i go to jupyter notebook and run command <code>nlp = spacy.load('en_core_web_sm')</code> I get the below error </p>''',
                  'so_title': "spacy Can't find model 'en_core_web_sm' on windows 10 and Python 3.5.3 :: Anaconda custom (64-bit)",
                  'so_tags': '<python><python-3.x><nlp><spacy>',
                  'rel_num': 36}

stanford_nlp_data1 = {'api': 'stanford-nlp',
                         'so_body': '''<p>How can I split a text or paragraph into sentences using <a href="http://nlp.stanford.edu/software/lex-parser.shtml" rel="noreferrer">Stanford parser</a>?</p>

<p>Is there any method that can extract sentences, such as <code>getSentencesFromString()</code> as it's provided for <a href="http://stanfordparser.rubyforge.org/" rel="noreferrer">Ruby</a>?</p>''',
                         'so_title': "How can I split a text into sentences using the Stanford parser?",
                         'so_tags': '<java><parsing><artificial-intelligence><nlp><stanford-nlp>',
                         'rel_num': 28}

TextBlob_data1 = {'api': "TextBlob",
                     'so_body': "<p>I want to analyze sentiment of texts that are written in German. "
                                "I found a lot of tutorials on how to do this with English, "
                                "but I found none on how to apply it to different languages.</p>",
                     'so_title': "Sentiment analysis of non-English texts",
                     'so_tags': "<python><machine-learning><nlp><sentiment-analysis><textblob>",
                     'rel_num': 40}

Transformers_data1 = {'api': "Transformers",
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