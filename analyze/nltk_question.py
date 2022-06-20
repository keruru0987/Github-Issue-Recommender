# coding=utf-8
# @Author : Eric


nltk_data1 = {'api': 'nltk',
              'so_body': "<p>I'm just starting to use NLTK and I don't quite understand how to get a list of words from text."
                            " If I use <code>nltk.word_tokenize()</code>, I get a list of words and punctuation. "
                            "I need only the words instead. How can I get rid of punctuation? "
                            "Also <code>word_tokenize</code> doesn't work with multiple sentences: dots are added to the last word.</p>",
              'so_title': 'How to get rid of punctuation using NLTK tokenizer?',
              'so_tags': '<python><nlp><tokenize><nltk>',
              'rel_num': 26}

nltk_data2 = {'api': 'nltk',
              'so_body': '''<p>When do I use each ?</p>

<p>Also...is the NLTK lemmatization dependent upon Parts of Speech?
Wouldn't it be more accurate if it was?</p>
              ''',
              'so_title': 'What is the difference between lemmatization vs stemming?',
              'so_tags': '<python><nlp><nltk><lemmatization>',
              'rel_num': 26}

nltk_data3 = {'api': 'nltk',
              'so_body': '''<p>I want to check in a Python program if a word is in the English dictionary.</p>

<p>I believe nltk wordnet interface might be the way to go but I have no clue how to use it for such a simple task.</p>

<pre><code>def is_english_word(word):
    pass # how to I implement is_english_word?

is_english_word(token.lower())
</code></pre>

<p>In the future, I might want to check if the singular form of a word is in the dictionary (e.g., properties -> property -> english word). How would I achieve that?</p>
              ''',
              'so_title': 'How to check if a word is an English word with Python?',
              'so_tags': '<python><nltk><wordnet>',
              'rel_num': 26}

nltk_data4 = {'api': 'nltk',
              'so_body': '''<p>When trying to load the <code>punkt</code> tokenizer...</p>

<pre><code>import nltk.data
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
</code></pre>

<p>...a <code>LookupError</code> was raised:</p>

<pre><code>&gt; LookupError: 
&gt;     *********************************************************************   
&gt; Resource 'tokenizers/punkt/english.pickle' not found.  Please use the NLTK Downloader to obtain the resource: nltk.download().   Searched in:
&gt;         - 'C:\\Users\\Martinos/nltk_data'
&gt;         - 'C:\\nltk_data'
&gt;         - 'D:\\nltk_data'
&gt;         - 'E:\\nltk_data'
&gt;         - 'E:\\Python26\\nltk_data'
&gt;         - 'E:\\Python26\\lib\\nltk_data'
&gt;         - 'C:\\Users\\Martinos\\AppData\\Roaming\\nltk_data'
&gt;     **********************************************************************
</code></pre>
              ''',
              'so_title': 'Failed loading english.pickle with nltk.data.load',
              'so_tags': '<python><jenkins><nltk>',
              'rel_num': 26}

nltk_data5 = {'api': 'nltk',
              'so_body': '''<p>How do I find a list with all possible pos tags used by the Natural Language Toolkit (nltk)?</p>
              ''',
              'so_title': 'What are all possible pos tags of NLTK?',
              'so_tags': '<python><nltk>',
              'rel_num': 26}

nltk_data6 = {'api': 'nltk',
              'so_body': '''<p>So I have a dataset that I would like to remove stop words from using </p>

<pre><code>stopwords.words('english')
</code></pre>

<p>I'm struggling how to use this within my code to just simply take out these words. I have a list of the words from this dataset already, the part i'm struggling with is comparing to this list and removing the stop words.
Any help is appreciated.</p>

              ''',
              'so_title': 'How to remove stop words using nltk or python',
              'so_tags': '<python><nltk><stop-words>',
              'rel_num': 26}

nltk_data7 = {'api': 'nltk',
              'so_body': '''<p>In shell script I am checking whether this packages are installed or not, if not installed then install it. So withing shell script:</p>

<pre><code>import nltk
echo nltk.__version__
</code></pre>

<p>but it stops shell script at <code>import</code> line</p>

<p>in linux terminal tried to see in this manner:</p>

<pre><code>which nltk
</code></pre>

<p>which gives nothing thought it is installed.</p>

<p>Is there any other way to verify this package installation in shell script, if not installed, also install it.</p>

              ''',
              'so_title': 'how to check which version of nltk, scikit learn installed?',
              'so_tags': '<python><linux><shell><scikit-learn><nltk>',
              'rel_num': 26}

nltk_data8 = {'api': 'nltk',
              'so_body': '''<p>I have a difficult time using pip to install almost anything. I'm new to coding, so I thought maybe this is something I've been doing wrong and have opted out to easy_install to get most of what I needed done, which has generally worked. However, now I'm trying to download the nltk library, and neither is getting the job done.</p>

<p>I tried entering</p>

<pre><code>sudo pip install nltk
</code></pre>

<p>but got the following response:</p>

<pre><code>/Library/Frameworks/Python.framework/Versions/2.7/bin/pip run on Sat May  4 00:15:38 2013
Downloading/unpacking nltk

  Getting page https://pypi.python.org/simple/nltk/
  Could not fetch URL [need more reputation to post link]: There was a problem confirming the ssl certificate: &lt;urlopen error [Errno 1] _ssl.c:504: error:0D0890A1:asn1 encoding routines:ASN1_verify:unknown message digest algorithm&gt;

  Will skip URL [need more reputation to post link]/simple/nltk/ when looking for download links for nltk

  Getting page [need more reputation to post link]/simple/
  Could not fetch URL https://pypi.python. org/simple/: There was a problem confirming the ssl certificate: &lt;urlopen error [Errno 1] _ssl.c:504: error:0D0890A1:asn1 encoding routines:ASN1_verify:unknown message digest algorithm&gt;

  Will skip URL [need more reputation to post link] when looking for download links for nltk

  Cannot fetch index base URL [need more reputation to post link]

  URLs to search for versions for nltk:
  * [need more reputation to post link]
  Getting page [need more reputation to post link]
  Could not fetch URL [need more reputation to post link]: There was a problem confirming the ssl certificate: &lt;urlopen error [Errno 1] _ssl.c:504: error:0D0890A1:asn1 encoding routines:ASN1_verify:unknown message digest algorithm&gt;

  Will skip URL [need more reputation to post link] when looking for download links for nltk

  Could not find any downloads that satisfy the requirement nltk

No distributions at all found for nltk

Exception information:
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pip-1.3.1-py2.7.egg/pip/basecommand.py", line 139, in main
    status = self.run(options, args)
  File "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pip-1.3.1-py2.7.egg/pip/commands/install.py", line 266, in run
    requirement_set.prepare_files(finder, force_root_egg_info=self.bundle, bundle=self.bundle)
  File "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pip-1.3.1-py2.7.egg/pip/req.py", line 1026, in prepare_files
    url = finder.find_requirement(req_to_install, upgrade=self.upgrade)
  File "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pip-1.3.1-py2.7.egg/pip/index.py", line 171, in find_requirement
    raise DistributionNotFound('No distributions at all found for %s' % req)
DistributionNotFound: No distributions at all found for nltk

--easy_install installed fragments of the library and the code ran into trouble very quickly upon trying to run it.
</code></pre>

<p>Any thoughts on this issue? I'd really appreciate some feedback on how I can either get pip working or something to get around the issue in the meantime.</p>

              ''',
              'so_title': 'pip issue installing almost any library',
              'so_tags': '<python><pip><nltk><easy-install>',
              'rel_num': 26}

nltk_data9 = {'api': 'nltk',
              'so_body': '''<p>I was following a tutorial which was available at <a href="http://blog.christianperone.com/?p=1589" rel="noreferrer">Part 1</a> &amp; <a href="http://blog.christianperone.com/?p=1747" rel="noreferrer">Part 2</a>. Unfortunately the author didn't have the time for the final section which involved using cosine similarity to actually find the distance between two documents. I followed the examples in the article with the help of the following link from <a href="https://stackoverflow.com/questions/11911469/tfidf-for-search-queries">stackoverflow</a>, included is the code mentioned in the above link (just so as to make life easier)</p>

<pre><code>from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

train_set = ["The sky is blue.", "The sun is bright."]  # Documents
test_set = ["The sun in the sky is bright."]  # Query
stopWords = stopwords.words('english')

vectorizer = CountVectorizer(stop_words = stopWords)
#print vectorizer
transformer = TfidfTransformer()
#print transformer

trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
testVectorizerArray = vectorizer.transform(test_set).toarray()
print 'Fit Vectorizer to train set', trainVectorizerArray
print 'Transform Vectorizer to test set', testVectorizerArray

transformer.fit(trainVectorizerArray)
print
print transformer.transform(trainVectorizerArray).toarray()

transformer.fit(testVectorizerArray)
print 
tfidf = transformer.transform(testVectorizerArray)
print tfidf.todense()
</code></pre>

<p>as a result of the above code I have the following matrix</p>

<pre><code>Fit Vectorizer to train set [[1 0 1 0]
 [0 1 0 1]]
Transform Vectorizer to test set [[0 1 1 1]]

[[ 0.70710678  0.          0.70710678  0.        ]
 [ 0.          0.70710678  0.          0.70710678]]

[[ 0.          0.57735027  0.57735027  0.57735027]]
</code></pre>

<p>I am not sure how to use this output in order to calculate cosine similarity, I know how to implement cosine similarity with respect to two vectors of similar length but here I am not sure how to identify the two vectors.</p>

              ''',
              'so_title': 'Python: tf-idf-cosine: to find document similarity',
              'so_tags': '<python><machine-learning><nltk><information-retrieval><tf-idf>',
              'rel_num': 26}

nltk_data10 = {'api': 'nltk',
               'so_body': '''<p>I am trying to process a user entered text by removing stopwords using nltk toolkit, but with stopword-removal the words like 'and', 'or', 'not' gets removed. I want these words to be present after stopword removal process as they are operators which are required for later processing text as query. I don't know which are the words which can be operators in text query, and I also want to remove unnecessary words from my text.</p>
               ''',
               'so_title': 'Stopword removal with NLTK',
               'so_tags': '<python><nlp><nltk><stop-words>',
               'rel_num': 26}