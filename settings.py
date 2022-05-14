# coding=utf-8
# @Author : Eric

nlp_api = ['allennlp', 'gensim', 'nltk', 'spaCy', 'stanford-nlp', 'TextBlob', 'Transformers']
nlp_choose = 5  # 指定当前研究的API

api_label_prelink = {'allennlp':'https://github.com/allenai/allennlp/labels/', 'gensim':'https://github.com/RaRe-Technologies/gensim/labels/',
               'nltk':'https://github.com/nltk/nltk/labels/', 'spaCy': 'https://github.com/explosion/spaCy/labels/',
               'stanford-nlp':'https://github.com/stanfordnlp/CoreNLP/labels/', 'TextBlob':'https://github.com/sloria/TextBlob/labels/',
               'Transformers':'https://github.com/huggingface/transformers/labels/'}

api_prelink = {'allennlp':'https://github.com/allenai/allennlp/issues/', 'gensim':'https://github.com/RaRe-Technologies/gensim/issues/',
               'nltk':'https://github.com/nltk/nltk/issues/', 'spaCy': 'https://github.com/explosion/spaCy/issues/',
               'stanford-nlp':'https://github.com/stanfordnlp/CoreNLP/issues/', 'TextBlob':'https://github.com/sloria/TextBlob/issues/',
               'Transformers':'https://github.com/huggingface/transformers/issues/'}

github_filepath = {'gensim': 'data/issue/gensim.json', 'allennlp': 'data/issue/allennlp.json',
                   'nltk': 'data/issue/nltk.json', 'spaCy': 'data/issue/spaCy.json',
                   'stanford-nlp': 'data/issue/CoreNLP.json', 'TextBlob': 'data/issue/TextBlob.json',
                   'Transformers': 'data/issue/transformers.json'}

new_github_filepath = {'gensim': 'data/new_issue/gensim.csv', 'allennlp': 'data/new_issue/allennlp.csv',
                       'nltk': 'data/new_issue/nltk.csv', 'spaCy': 'data/new_issue/spaCy.csv',
                       'stanford-nlp': 'data/new_issue/CoreNLP.csv', 'TextBlob': 'data/new_issue/TextBlob.csv',
                       'Transformers': 'data/new_issue/transformers.csv'}

tagged_github_filepath = {'gensim': 'data/tagged_data/gensim.csv', 'allennlp': 'data/tagged_data/allennlp.csv',
                          'nltk': 'data/tagged_data/nltk.csv', 'spaCy': 'data/tagged_data/spaCy.csv',
                          'stanford-nlp': 'data/tagged_data/CoreNLP.csv', 'TextBlob': 'data/tagged_data/TextBlob.csv',
                          'Transformers': 'data/tagged_data/transformers.csv'}

stackoverflow_filepath = {'gensim':'D:/data/stack/gensim.csv', 'allennlp':'D:/data/stack/allennlp.csv',
                          'nltk':'D:/data/stack/nltk.csv', 'spaCy':'D:/data/stack/spacy.csv',
                          'stanford-nlp':'D:/data/stack/stanford-nlp.csv', 'TextBlob':'D:/data/stack/textblob.csv',
                          'Transformers':'D:/data/stack/transformers.csv'}

word2vec_modelpath = 'D:/model/GoogleNews-vectors-negative300.bin'

select_num = 20  # 选取的分数最高的issue数目

allen_text = '''
<p>I am new to the AllenNLP library. I am using the Pretrained Bidaf-elmo model for a reading comprehension task. My code looks like -</p>
<p><div class="snippet" data-lang="js" data-hide="false" data-console="true" data-babel="false">
<div class="snippet-code">
<pre class="snippet-code-html lang-html prettyprint-override"><code>from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
from allennlp_models import pretrained
from allennlp.training.util import evaluate
import allennlp.data.data_loaders.simple_data_loader

archive_file_path = "https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz"
input_path = "C:\\Users\\SHRIPRIYA\\sample_dataset.json"

data_load = simple_data_loader(input_path)
evaluate(model=archive_file_path, data_loader = data_load, output_file=output, predictions_output_file=pred_output_file, cuda_device=0)</code></pre>
</div>
</div>
</p>
<p>The line <code>simple_data_loader()</code> throws an error - <code>name 'simple_data_loader' is not defined</code>. I know this is a syntax error but I could not find any examples to load a JSON file using a Data Loader function from AllenNLP and evaluate it using a pre-trained model.</p>
<p>About my data:</p>
<ol>
<li>total sample passages = 10,000</li>
<li>total questions = 1000</li>
</ol>
<p>Each sample passage needs to be subjected to all the 1000 questions. My sample JSON input looks like -</p>
<pre><code>{
  &quot;passage&quot;: &quot;Venus is named after the Roman goddess of love and beauty. Venus is the second planet from the sun. Is the brightest object in the sky besides our Sun and the Moon. Venus has no moons. It is also known as the morning star because at sunrise it appears in the east. It is also known as the evening star as it appears at sunset when it is in the west. It cannot be seen in the middle of the night. Venus and Earth are close together in space and similar in size, which is the reason Venus is called Earth's sister planet. Venus has more volcanoes than any other planet. It is the hottest planet in the solar system, even hotter than Mercury, which is closer to the Sun. The temperature on the surface of Venus is about 460° Celsius. The atmosphere on Venus is composed of carbon dioxide. The surface is heated by radiation from the sun, but the heat cannot escape through the clouds and layer of carbon dioxide. (This is a “greenhouse effect”).&quot;,
  &quot;questions&quot;: [
    &quot;How many moons does Venus have?&quot;,
    &quot;Venus was named after which Roman goddess?&quot;,    
    &quot;At what position does Venus lie from Sun?&quot;,
    &quot;What is the temperature of Venus surface?&quot;,
    &quot;Why is Venus called Earth’s sister planet?&quot;,
    &quot;What is the atmosphere of Venus composed of?&quot;
  ]}
</code></pre>
<p>If there's any faster alternative to evaluate multiple questions against multiple passages, please let me know.</p>
<p>Thanks!</p>'''

gensim_text = '''
I have a use case where I want to find top n nearest words from a given set of words, to a vector. 
Its like similar_by_vector where I want to restrict my vocab to a given set of words.
similar_by_vector(vector, topn, vocab=[x,y,z...])
I want to create a low latency api using this, where vocab can be different for each request.
Any suggestions to how I can achieve this optimally?'''

nltk_text = '''
<p>I have:</p>

<pre><code>from __future__ import division
import nltk, re, pprint
f = open('/home/a/Desktop/Projects/FinnegansWake/JamesJoyce-FinnegansWake.txt')
raw = f.read()
tokens = nltk.wordpunct_tokenize(raw)
text = nltk.Text(tokens)
words = [w.lower() for w in text]

f2 = open('/home/a/Desktop/Projects/FinnegansWake/catted-several-long-Russian-novels-and-the-NYT.txt')
englishraw = f2.read()
englishtokens = nltk.wordpunct_tokenize(englishraw)
englishtext = nltk.Text(englishtokens)
englishwords = [w.lower() for w in englishwords]
</code></pre>

<p>which is straight from the NLTK manual. What I want to do next is to compare <code>vocab</code> to an exhaustive set of English words, like the OED, and extract the difference -- the set of Finnegans Wake words that have not, and probably never will, be in the OED. I'm much more of a verbal person than a math-oriented person, so I haven't figured out how to do that yet, and the manual goes into way too much detail about stuff I don't actually want to do. I'm assuming it's just one or two more lines of code, though. </p>'''

spaCy_text = '''
<p>I'm new to spaCy and currently trying to use spaCy english large model to identify PERSON from sentences <br>
All is fine to identify PERSON from sentences until I found some name that's identified is not PERSON. <br>
E.g. If I put &quot;Alex is eating apple&quot;. It will successfully return Alex is a PERSON <br>
But when this case happens, it won't work anymore<br>
E.g. Sun Saw Bee is eating apple <strong>or</strong> Alexandro Soon is eating apple<br>
<br>
I'm wondering if there is anything like whitelist to add in &quot;Sun Saw Bee&quot; or &quot;Alexandro Soon&quot; as a PERSON without retraining spaCy english model? <br>
or any way to somehow identify &quot;Sun Saw Bee&quot; as a PERSON instead?<br><br>
if there is any link related to this perhaps can share as well, since my keyword searching might not hitting the right key</p>'''

stanfordnlp_text = '''
<p>I have been trying to use the CoreNLP server using various python packages including <a href="https://stanfordnlp.github.io/stanza/corenlp_client.html" rel="nofollow noreferrer">Stanza</a>. I am always running into the same problem that I do not hear back from the server. So I instead looked up as to how I can test the original server based on the documentation. </p>

<p>I downloaded a copy of CoreNLP from the <a href="https://stanfordnlp.github.io/CoreNLP/download.html" rel="nofollow noreferrer">website</a>. I then try to start a server from the terminal and go to my localhost as described <a href="https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started" rel="nofollow noreferrer">here</a>. 
Based on the documentation I should see something when I go to <a href="http://localhost:9000/" rel="nofollow noreferrer">http://localhost:9000/</a>, but nothing loads up.</p>

<p>Here are to commands I use:</p>

<pre><code>cd stanford-corenlp-full-2018-10-05/
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
</code></pre>

<p>Here is the output of running the commands above:</p>

<pre><code>[main] INFO CoreNLP - --- StanfordCoreNLPServer#main() called ---
[main] INFO CoreNLP - setting default constituency parser
[main] INFO CoreNLP - warning: cannot find edu/stanford/nlp/models/srparser/englishSR.ser.gz
[main] INFO CoreNLP - using: edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz instead
[main] INFO CoreNLP - to use shift reduce parser download English models jar from:
[main] INFO CoreNLP - http://stanfordnlp.github.io/CoreNLP/download.html
[main] INFO CoreNLP -     Threads: 8
[main] INFO CoreNLP - Starting server...
[main] INFO CoreNLP - StanfordCoreNLPServer listening at /0:0:0:0:0:0:0:0:9000
</code></pre>

<p>I then go to <a href="http://localhost:9000/" rel="nofollow noreferrer">http://localhost:9000/</a>, nothing loads up. Like I mentioned above originally I have been trying to do the same thing using some of the python packages and observed similar behavior.</p>

<p><a href="https://stackoverflow.com/questions/60961208/stanfordcorenlp-server-listening-indefinitely-using-stanza">Here</a> is a stack overflow post related to server not responding using Stanza.</p>

<p>OS: MacOS 10.15.4
Python: 3.7.7
Java: 1.8</p>

<p>Could it an issue with Java version? Or with the model I downloaded from the website? </p>
'''

TextBlob_text = '''
<p>I am trying to analyze open-ended questions with Polarity and subjectivity. So what I want to achieve is to upload the CSV file, then add new columns one for polarity, subjectively, negative or positive column and here what I did:</p>
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

Transformers_text = '''
<p>I want to do chinese Textual Similarity with huggingface:</p>
<pre><code>tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese')
</code></pre>
<p>It doesn't work, system report errors:</p>
<pre><code>Some weights of the model checkpoint at bert-base-chinese were not used when initializing TFBertForSequenceClassification: ['nsp___cls', 'mlm___cls']
- This IS expected if you are initializing TFBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
- This IS NOT expected if you are initializing TFBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of TFBertForSequenceClassification were not initialized from the model checkpoint at bert-base-chinese and are newly initialized: ['classifier', 'dropout_37']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
</code></pre>
<p>But I can use huggingface to do name entity:</p>
<pre><code>tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = TFBertForTokenClassification.from_pretrained(&quot;bert-base-chinese&quot;)
</code></pre>
<p>Does that mean huggingface haven't done chinese sequenceclassification? If my judge is right, how to sove this problem with colab with only 12G memory？</p>'''

stackoverflow_text = {'allennlp': allen_text, 'gensim': gensim_text, 'nltk': nltk_text, 'spaCy': spaCy_text,
                      'stanford-nlp': stanfordnlp_text, 'TextBlob': TextBlob_text, 'Transformers': Transformers_text}