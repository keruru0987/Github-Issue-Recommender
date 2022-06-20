# coding=utf-8
# @Author : Eric


stanford_nlp_data1 = {'api': 'stanford-nlp',
                      'so_body': '''<p>How can I split a text or paragraph into sentences using <a href="http://nlp.stanford.edu/software/lex-parser.shtml" rel="noreferrer">Stanford parser</a>?</p>

<p>Is there any method that can extract sentences, such as <code>getSentencesFromString()</code> as it's provided for <a href="http://stanfordparser.rubyforge.org/" rel="noreferrer">Ruby</a>?</p>''',
                      'so_title': "How can I split a text into sentences using the Stanford parser?",
                      'so_tags': '<java><parsing><artificial-intelligence><nlp><stanford-nlp>',
                      'rel_num': 28}

stanford_nlp_data2 = {'api': 'stanford-nlp',
                      'so_body': '''<p>Is it possible to use Stanford Parser in NLTK? (I am not talking about Stanford POS.)</p>
                      ''',
                      'so_title': "How to use Stanford Parser in NLTK using Python",
                      'so_tags': '<python><parsing><nlp><nltk><stanford-nlp>',
                      'rel_num': 28}

stanford_nlp_data3 = {'api': 'stanford-nlp',
                      'so_body': '''<p>All I want to do is find the sentiment (positive/negative/neutral) of any given string. On researching I came across Stanford NLP. But sadly its in Java. Any ideas on how can I make it work for python? </p>

                      ''',
                      'so_title': "Stanford nlp for python",
                      'so_tags': '<python><stanford-nlp><sentiment-analysis>',
                      'rel_num': 28}

stanford_nlp_data4 = {'api': 'stanford-nlp',
                      'so_body': '''<p>I have recently started to use NLTK toolkit for creating few solutions using Python.</p>
<p>I hear a lot of community activity regarding using Stanford NLP.
Can anyone tell me the difference between NLTK and Stanford NLP? Are they two different libraries? I know that NLTK has an interface to Stanford NLP but can anyone throw some light on few basic differences or even more in detail.</p>
<p>Can Stanford NLP be used using Python?</p>
                      ''',
                      'so_title': "NLTK vs Stanford NLP",
                      'so_tags': '<python><nlp><nltk><stanford-nlp>',
                      'rel_num': 28}

stanford_nlp_data5 = {'api': 'stanford-nlp',
                      'so_body': '''<p>I'm using some NLP libraries now, (stanford and nltk) 
Stanford I saw the demo part but just want to ask if it possible to use it to identify more entity types.</p>

<p>So currently stanford NER system (as the demo shows) can recognize entities as person(name), organization or location. But the organizations recognized are limited to universities or some, big organizations. I'm wondering if I can use its API to write program for more entity types, like if my input is "Apple" or  "Square" it can recognize it as a company.</p>

<p>Do I have to make my own training dataset?</p>

<p>Further more, if I ever want to extract entities and their relationships between each other, I feel I should use the stanford dependency parser.
I mean, extract first the named entities and other parts tagged as "noun" and find relations between them.</p>

<p>Am I correct.</p>

<p>Thanks.</p>
                      ''',
                      'so_title': "Is it possible to train Stanford NER system to recognize more named entities types?",
                      'so_tags': '<nlp><stanford-nlp><named-entity-recognition>',
                      'rel_num': 28}

stanford_nlp_data6 = {'api': 'stanford-nlp',
                      'so_body': '''<p>I tried to follow <a href="https://nlp.stanford.edu/projects/glove/" rel="noreferrer">this.</a><br>
But some how I wasted a lot of time ending up with nothing useful.<br>
I just want to train a <code>GloVe</code> model on my own corpus (~900Mb corpus.txt file).
I downloaded the files provided in the link above and compiled it using <code>cygwin</code> (after editing the demo.sh file and changed it to <code>VOCAB_FILE=corpus.txt</code> . should I leave <code>CORPUS=text8</code> unchanged?)
the output was:  </p>

<ol>
<li>cooccurrence.bin </li>
<li>cooccurrence.shuf.bin  </li>
<li>text8</li>
<li>corpus.txt</li>
<li>vectors.txt</li>
</ol>

<p>How can I used those files to load it as a <code>GloVe</code> model on python?</p>
                      ''',
                      'so_title': "How to Train GloVe algorithm on my own corpus",
                      'so_tags': '<nlp><stanford-nlp><gensim><word2vec><glove>',
                      'rel_num': 28}

stanford_nlp_data7 = {'api': 'stanford-nlp',
                      'so_body': '''<p>I have a tree, specifically a parse tree with tags at the nodes and strings/words at the leaves. I want to pass this tree as input into a neural network all the while preserving its structure.</p>

<p>Current approach
Assume we have some dictionary of words w1,w2.....wn
Encode the words that appear in the parse tree as n dimensional binary vectors with a 1 showing up in the ith spot whenever the word in the parse tree is wi</p>

<p>Now how about the tree structure? There are about 2^n  possible parent tags for n words that appear at the leaves So we cant set a max length of input words and then just brute force enumerate all trees.</p>

<p>Right now all i can think of is to approximate the tree by choosing the direct parent of a leaf. This can be represented by a binary vector as well with dimension equal to number of different types of tags - on the order of ~ 100 i suppose.
My input is then two dimensional. The first is just the vector representation of a word and the second is the vector representation of its parent tag</p>

<p>Except this will lose a lot of the structure in the sentence. Is there a standard/better way of solving this problem?</p>
                      ''',
                      'so_title': "How can a tree be encoded as input to a neural network?",
                      'so_tags': '<machine-learning><nlp><neural-network><stanford-nlp><deep-learning>',
                      'rel_num': 28}

stanford_nlp_data8 = {'api': 'stanford-nlp',
                      'so_body': '''<p>I want to compute how similar two arbitrary sentences are to each other.  For example:</p>

<blockquote>
  <ol>
  <li>A mathematician found a solution to the problem.</li>
  <li>The problem was solved by a young mathematician.</li>
  </ol>
</blockquote>

<p>I can use a tagger, a stemmer, and a parser, but I don’t know how detect that these sentences are similar.</p>
                      ''',
                      'so_title': "How to detect that two sentences are similar?",
                      'so_tags': '<nlp><similarity><stanford-nlp><opennlp>',
                      'rel_num': 28}

stanford_nlp_data9 = {'api': 'stanford-nlp',
                      'so_body': '''<p>I just started using Stanford Parser but I do not understand the tags very well. This might be a stupid question to ask but can anyone tell me what does the SBARQ and SQ tags represent and where can I find a complete list for them? I know how the Penn Treebank looks like but these are slightly different. </p>

<pre><code>Sentence: What is the highest waterfall in the United States ?

(ROOT
  (SBARQ
    (WHNP (WP What))
    (SQ (VBZ is)
      (NP
        (NP (DT the) (JJS highest) (NN waterfall))
        (PP (IN in)
          (NP (DT the) (NNP United) (NNPS States)))))
    (. ?)))
</code></pre>

<p>I have looked at Stanford Parser website and read a few of the journals listed there but there are no explanation of the tags mentioned earlier. I found a manual describing all the dependencies used but it doesn't explain what I am looking for. Thanks!</p>
                      ''',
                      'so_title': "Stanford Parser tags",
                      'so_tags': '<stanford-nlp>',
                      'rel_num': 28}

stanford_nlp_data10 = {'api': 'stanford-nlp',
                       'so_body': '''<p>Hell everyone! I'm using the Stanford Core NLP package and my goal is to perform sentiment analysis on a live-stream of tweets. </p>

<p>Using the sentiment analysis tool as is returns a very poor analysis of text's 'attitude' .. many positives are labeled neutral, many negatives rated positive. I've gone ahead an acquired well over a million tweets in a text file, but I haven't a clue how to actually <em>train</em> the tool and create my own model.</p>

<p><a href="http://nlp.stanford.edu/sentiment/code.html">Link to Stanford Sentiment Analysis page</a></p>

<p>"Models can be retrained using the following command using the PTB format dataset:"</p>

<pre><code>java -mx8g edu.stanford.nlp.sentiment.SentimentTraining -numHid 25 -trainPath train.txt -devPath     dev.txt -train -model model.ser.gz
</code></pre>

<p>Sample from dev.txt (The leading 4 represents polarity out of 5 ... 4/5 positive)</p>

<pre><code>(4 (4 (2 A) (4 (3 (3 warm) (2 ,)) (3 funny))) (3 (2 ,) (3 (4 (4 engaging) (2 film)) (2 .))))
</code></pre>

<p>Sample from test.txt</p>

<pre><code>(3 (3 (2 If) (3 (2 you) (3 (2 sometimes) (2 (2 like) (3 (2 to) (3 (3 (2 go) (2 (2 to) (2 (2 the) (2 movies)))) (3 (2 to) (3 (2 have) (4 fun))))))))) (2 (2 ,) (2 (2 Wasabi) (3 (3 (2 is) (2 (2 a) (2 (3 good) (2 (2 place) (2 (2 to) (2 start)))))) (2 .)))))
</code></pre>

<p>Sample from train.txt</p>

<pre><code>(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's)) (2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 '')) (2 and)) (3 (2 that) (3 (2 he) (3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) (2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) (2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .)))
</code></pre>

<p>I have two questions going forward.</p>

<p>What is the significance and difference between each file? Train.txt/Dev.txt/Test.txt ?</p>

<p>How would I train my own model with a raw, unparsed text file full of tweets?</p>

<p>I'm very new to NLP so if I am missing any required information or anything at all please critique! Thank you!</p>

                       ''',
                       'so_title': "How to train the Stanford NLP Sentiment Analysis tool",
                       'so_tags': '<java><nlp><stanford-nlp><sentiment-analysis>',
                       'rel_num': 28}


