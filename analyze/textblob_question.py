# coding=utf-8
# @Author : Eric


TextBlob_data1 = {'api': "TextBlob",
                  'so_body': "<p>I want to analyze sentiment of texts that are written in German. "
                                "I found a lot of tutorials on how to do this with English, "
                                "but I found none on how to apply it to different languages.</p>",
                  'so_title': "Sentiment analysis of non-English texts",
                  'so_tags': "<python><machine-learning><nlp><sentiment-analysis><textblob>",
                  'rel_num': 40}

TextBlob_data2 = {'api': "TextBlob",
                  'so_body': '''<p>I am training the <code>NaiveBayesClassifier</code> in Python using sentences, and it gives me the error below. I do not understand what the error might be, and any help would be good. </p>

<p>I have tried many other input formats, but the error remains. The code given below:</p>

<pre><code>from text.classifiers import NaiveBayesClassifier
from text.blob import TextBlob
train = [('I love this sandwich.', 'pos'),
         ('This is an amazing place!', 'pos'),
         ('I feel very good about these beers.', 'pos'),
         ('This is my best work.', 'pos'),
         ("What an awesome view", 'pos'),
         ('I do not like this restaurant', 'neg'),
         ('I am tired of this stuff.', 'neg'),
         ("I can't deal with this", 'neg'),
         ('He is my sworn enemy!', 'neg'),
         ('My boss is horrible.', 'neg') ]

test = [('The beer was good.', 'pos'),
        ('I do not enjoy my job', 'neg'),
        ("I ain't feeling dandy today.", 'neg'),
        ("I feel amazing!", 'pos'),
        ('Gary is a friend of mine.', 'pos'),
        ("I can't believe I'm doing this.", 'neg') ]
classifier = nltk.NaiveBayesClassifier.train(train)
</code></pre>

<p>I am including the traceback below.</p>
                  ''',
                  'so_title': "nltk NaiveBayesClassifier training for sentiment analysis",
                  'so_tags': "<python><nlp><nltk><sentiment-analysis><textblob>",
                  'rel_num': 40}

TextBlob_data3 = {'api': "TextBlob",
                  'so_body': '''<p>i have a data frame with a col which has text. I want to apply textblob and calculate sentiment value for each row.</p>

<pre><code>text                sentiment
</code></pre>

<p>this is great<br>
great movie 
great story </p>

<p>When i execute the below code:</p>

<p><code>df['sentiment'] = list(map(lambda tweet: TextBlob(tweet), df['text']))</code></p>

<p>I get the error:</p>

<pre><code>TypeError: The `text` argument passed to `__init__(text)` must be a string, not &lt;class 'float'&gt;
</code></pre>

<p>How do you apply textBLob to each row of a col in a dataframe to get the sentiment value?</p>
                  ''',
                  'so_title': "Apply textblob in for each row of a dataframe",
                  'so_tags': "<python><pandas><textblob>",
                  'rel_num': 40}

TextBlob_data4 = {'api': "TextBlob",
                  'so_body': '''<p>i have searched the web about normalizing tf grades on cases when the documents' lengths are very different
(for example, having the documents lengths vary from 500 words to 2500 words)</p>

<p>the only normalizing i've found talk about dividing the term frequency in the length of the document, hence causing the length of the document to not have any meaning.</p>

<p>this method though is a really bad one for normalizing tf. if any, it causes the tf grades for each document to have a very large bias (unless all documents are constructed from pretty much the same dictionary, which is not the case when using tf-idf)</p>

<p>for example lets take 2 documents - one consisting of 100 unique words, and the other of 1000 unique words. each word in doc1 will have a tf of 0.01 while in doc2 each word will have a tf of 0.001</p>

<p>this causes tf-idf grades to automatically be bigger when matching words with doc1 than doc2</p>

<p>have anyone got any suggustion of a more suitable normalizing formula?</p>

<p>thank you</p>

<p><strong><em>edit</em></strong>
i also saw a method stating we should divide the term frequency with the maximum term frequency of the doc for each doc
this also isnt solving my problem</p>

<p>what i was thinking, is calculating the maximum term frequency from all the documents and then normalizing all of the terms by dividing each term frequency with the maximum</p>

<p>would love to know what you think</p>
                  ''',
                  'so_title': "tf-idf documents of different length",
                  'so_tags': "<python><normalization><tf-idf><textblob>",
                  'rel_num': 40}

TextBlob_data5 = {'api': "TextBlob",
                  'so_body': '''<p>I have been using TextBlob, a package for Python (<a href="https://pypi.python.org/pypi/textblob" rel="noreferrer">https://pypi.python.org/pypi/textblob</a>) for translating articles to different language . </p>

<p>After reading their docs, I got to know that TextBlob makes use of Google Translate. Since google translate is not a free service, I wanted to know whether there is any usage limit on translating articles using TextBlob services? </p>
                  ''',
                  'so_title': "Is there a limit on TextBlob translation?",
                  'so_tags': "<python><google-translate><textblob>",
                  'rel_num': 40}

TextBlob_data6 = {'api': "TextBlob",
                  'so_body': '''<p>I installed textblob using pip as given <a href="http://textblob.readthedocs.org/en/dev/install.html" rel="noreferrer">here</a>.</p>

<p>Now, when I try to import this in python3.4 in terminal then it says </p>

<pre><code>ImportError: No module named 'textblob'
</code></pre>

<p>Whereas, in python2.7 it imports happily. I have tried reinstalling it. I have even reinstalled pip. What is the problem here?</p>
                  ''',
                  'so_title': "Running TextBlob in Python3",
                  'so_tags': "<python><python-3.4><textblob>",
                  'rel_num': 40}

TextBlob_data7 = {'api': "TextBlob",
                  'so_body': '''<p>Using the <a href="http://textblob.readthedocs.org/en/dev/quickstart.html#spelling-correction" rel="nofollow">TextBlob</a> library it is possible to improve the spelling of strings by defining them as TextBlob objects first and then using the <code>correct</code> method. </p>

<p>Example:</p>

<pre><code>from textblob import TextBlob
data = TextBlob('Two raods diverrged in a yullow waod and surry I culd not travl bouth')
print (data.correct())
Two roads diverged in a yellow wood and sorry I could not travel both
</code></pre>

<p>Is it possible to do this to strings in a Pandas DataFrame series such as this one:</p>

<pre><code>data = [{'one': '3', 'two': 'two raods'}, 
         {'one': '7', 'two': 'diverrged in a yullow'}, 
        {'one': '8', 'two': 'waod and surry I'}, 
        {'one': '9', 'two': 'culd not travl bouth'}]
df = pd.DataFrame(data)
df

    one   two
0   3     Two raods
1   7     diverrged in a yullow
2   8     waod and surry I
3   9     culd not travl bouth
</code></pre>

<p>To return this:</p>

<pre><code>    one   two
0   3     Two roads
1   7     diverged in a yellow
2   8     wood and sorry I
3   9     could not travel both
</code></pre>

<p>Either using TextBlob or some other method. </p>

                  ''',
                  'so_title': "How to correct spelling in a Pandas DataFrame",
                  'so_tags': "<python><pandas><nlp><textblob>",
                  'rel_num': 40}

TextBlob_data8 = {'api': "TextBlob",
                  'so_body': '''<p>The built-in classifier in textblob is pretty dumb. It's trained on movie reviews, so I created a huge set of examples in my context (57,000 stories, categorized as positive or negative) and then trained it using <code>nltk.</code> I tried using textblob to train it but it always failed:</p>

<pre><code>with open('train.json', 'r') as fp:
    cl = NaiveBayesClassifier(fp, format="json")
</code></pre>

<p>That would run for hours and end in a memory error. </p>

<p>I looked at the source and found it was just using nltk and wrapping that, so I used that instead, and it worked.</p>

<p>The structure for nltk training set needed to be a list of tuples, with the first part was a Counter of words in the text and frequency of appearance. The second part of tuple was 'pos' or 'neg' for sentiment.</p>

<pre><code>&gt;&gt;&gt; train_set = [(Counter(i["text"].split()),i["label"]) for i in data[200:]]
&gt;&gt;&gt; test_set = [(Counter(i["text"].split()),i["label"]) for i in data[:200]] # withholding 200 examples for testing later

&gt;&gt;&gt; cl = nltk.NaiveBayesClassifier.train(train_set) # &lt;-- this is the same thing textblob was using

&gt;&gt;&gt; print("Classifier accuracy percent:",(nltk.classify.accuracy(cl, test_set))*100)
('Classifier accuracy percent:', 66.5)
&gt;&gt;&gt;&gt;cl.show_most_informative_features(75)
</code></pre>

<p>Then I pickled it.</p>

<pre><code>with open('storybayes.pickle','wb') as f:
    pickle.dump(cl,f)
</code></pre>

<p>Now... I took this pickled file, and re opened it to get the nltk.classifier 'nltk.classify.naivebayes.NaiveBayesClassifier'&gt; -- and tried to feed it into textblob. Instead of </p>

<pre><code>from textblob.classifiers import NaiveBayesClassifier
blob = TextBlob("I love this library", analyzer=NaiveBayesAnalyzer())
</code></pre>

<p>I tried:</p>

<pre><code>blob = TextBlob("I love this library", analyzer=myclassifier)
Traceback (most recent call last):
  File "&lt;pyshell#116&gt;", line 1, in &lt;module&gt;
    blob = TextBlob("I love this library", analyzer=cl4)
  File "C:\python\lib\site-packages\textblob\blob.py", line 369, in __init__
    parser, classifier)
  File "C:\python\lib\site-packages\textblob\blob.py", line 323, in 
_initialize_models
    BaseSentimentAnalyzer, BaseBlob.analyzer)
  File "C:\python\lib\site-packages\textblob\blob.py", line 305, in 
_validated_param
    .format(name=name, cls=base_class_name))
ValueError: analyzer must be an instance of BaseSentimentAnalyzer
</code></pre>

<p>what now? I looked at the source and both are classes, but not quite exactly the same. </p>

                  ''',
                  'so_title': "After training my own classifier with nltk, how do I load it in textblob?",
                  'so_tags': "<python><nltk><naivebayes><textblob>",
                  'rel_num': 40}

TextBlob_data9 = {'api': "TextBlob",
                  'so_body': '''<p>I have followed the instruction in <a href="https://stackoverflow.com/questions/20562768/trouble-installing-textblob-for-python">Trouble installing TextBlob for Python</a> for TextBlob installation in the Windows 7.
It got installed but when I go to Python Idle and type <code>import TextBlob</code> it says</p>

<blockquote>
  <p>No module named TextBlob</p>
</blockquote>

<p>How to solve this problem?</p>

<p>Or can I directly place the libraries associated with the package in the Python Lib folder and try to import it in the program? If it is advisable please tell the procedure to do that.
Will it work?</p>

<p>Any help will be highly appreciated. </p>
                  ''',
                  'so_title': "TextBlob installation in windows",
                  'so_tags': "<python><windows><python-2.7><textblob>",
                  'rel_num': 40}

TextBlob_data10 = {'api': "TextBlob",
                   'so_body': '''<p>how does TextBlob calculate an empirical value for the sentiment polarity. I have used naive bayes but it just predicts whether it is positive or negative. How could I calculate a value for the sentiment like TextBlob does?</p>
                   ''',
                   'so_title': "How does TextBlob calculate sentiment polarity? How can I calculate a value for sentiment with machine learning classifier?",
                   'so_tags': "<python><python-3.x><machine-learning><sentiment-analysis><textblob>",
                   'rel_num': 40}