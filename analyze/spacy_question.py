# coding=utf-8
# @Author : Eric


spaCy_data1 = {'api': 'spaCy',
               'so_body': '''<p>what is difference between <code>spacy.load('en_core_web_sm')</code> and <code>spacy.load('en')</code>? <a href="https://stackoverflow.com/questions/50487495/what-is-difference-between-en-core-web-sm-en-core-web-mdand-en-core-web-lg-mod">This link</a> explains different model sizes. But i am still not clear how <code>spacy.load('en_core_web_sm')</code> and <code>spacy.load('en')</code> differ</p>

<p><code>spacy.load('en')</code> runs fine for me. But the <code>spacy.load('en_core_web_sm')</code> throws error</p>

<p>i have installed <code>spacy</code>as below. when i go to jupyter notebook and run command <code>nlp = spacy.load('en_core_web_sm')</code> I get the below error </p>
''',
               'so_title': "spacy Can't find model 'en_core_web_sm' on windows 10 and Python 3.5.3 :: Anaconda custom (64-bit)",
               'so_tags': '<python><python-3.x><nlp><spacy>',
               'rel_num': 36}

spaCy_data2 = {'api': 'spaCy',
               'so_body': '''<p>What is the best way to add/remove stop words with spacy? I am using <a href="https://spacy.io/docs/api/token" rel="noreferrer"><code>token.is_stop</code></a> function and would like to make some custom changes to the set. I was looking at the documentation but could not find anything regarding of stop words. Thanks!</p>
               ''',
               'so_title': "Add/remove custom stop words with spacy",
               'so_tags': '<python><nlp><stop-words><spacy>',
               'rel_num': 36}

spaCy_data3 = {'api': 'spaCy',
               'so_body': '''<p>I have been trying to find how to get the dependency tree with spaCy but I can't find anything on how to get the tree, only on <a href="https://spacy.io/usage/examples#subtrees" rel="noreferrer">how to navigate the tree</a>.</p>
               ''',
               'so_title': "How to get the dependency tree with spaCy?",
               'so_tags': '<python><spacy>',
               'rel_num': 36}

spaCy_data4 = {'api': 'spaCy',
               'so_body': '''<p>even though I downloaded the model it cannot load it</p>

<pre><code>[jalal@goku entity-sentiment-analysis]$ which python
/scratch/sjn/anaconda/bin/python
[jalal@goku entity-sentiment-analysis]$ sudo python -m spacy download en
[sudo] password for jalal: 
Collecting https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz (37.4MB)
    100% |████████████████████████████████| 37.4MB 9.4MB/s 
Installing collected packages: en-core-web-sm
  Running setup.py install for en-core-web-sm ... done
Successfully installed en-core-web-sm-2.0.0

    Linking successful
    /usr/lib/python2.7/site-packages/en_core_web_sm --&gt;
    /usr/lib64/python2.7/site-packages/spacy/data/en

    You can now load the model via spacy.load('en')

import spacy 

nlp = spacy.load('en')
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
&lt;ipython-input-2-0fcabaab8c3d&gt; in &lt;module&gt;()
      1 import spacy
      2 
----&gt; 3 nlp = spacy.load('en')

/scratch/sjn/anaconda/lib/python3.6/site-packages/spacy/__init__.py in load(name, **overrides)
     17             "to load. For example:\nnlp = spacy.load('{}')".format(depr_path),
     18             'error')
---&gt; 19     return util.load_model(name, **overrides)
     20 
     21 

/scratch/sjn/anaconda/lib/python3.6/site-packages/spacy/util.py in load_model(name, **overrides)
    118     elif hasattr(name, 'exists'):  # Path or Path-like to model data
    119         return load_model_from_path(name, **overrides)
--&gt; 120     raise IOError("Can't find model '%s'" % name)
    121 
    122 

OSError: Can't find model 'en'
</code></pre>

<p>How should I fix this?</p>

<p>If I don't use sudo for downloading the en model, I get:</p>

<pre><code>Collecting https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz (37.4MB)
    100% |████████████████████████████████| 37.4MB 9.6MB/s ta 0:00:011   62% |████████████████████            | 23.3MB 8.6MB/s eta 0:00:02
Requirement already satisfied (use --upgrade to upgrade): en-core-web-sm==2.0.0 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz in /scratch/sjn/anaconda/lib/python3.6/site-packages
You are using pip version 10.0.0, however version 10.0.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.

    Error: Couldn't link model to 'en'
    Creating a symlink in spacy/data failed. Make sure you have the required
    permissions and try re-running the command as admin, or use a
    virtualenv. You can still import the model as a module and call its
    load() method, or create the symlink manually.

    /scratch/sjn/anaconda/lib/python3.6/site-packages/en_core_web_sm --&gt;
    /scratch/sjn/anaconda/lib/python3.6/site-packages/spacy/data/en


    Download successful but linking failed
    Creating a shortcut link for 'en' didn't work (maybe you don't have
    admin permissions?), but you can still load the model via its full
    package name:

    nlp = spacy.load('en_core_web_sm')
</code></pre>

               ''',
               'so_title': "SpaCy OSError: Can't find model 'en'",
               'so_tags': '<nlp><spacy>',
               'rel_num': 36}

spaCy_data5 = {'api': 'spaCy',
               'so_body': '''<p>I'm working on a codebase that uses Spacy. I installed spacy using:</p>

<pre><code>sudo pip3 install spacy
</code></pre>

<p>and then </p>

<pre><code>sudo python3 -m spacy download en
</code></pre>

<p>At the end of this last command, I got a message:</p>

<pre><code>    Linking successful
/home/rayabhik/.local/lib/python3.5/site-packages/en_core_web_sm --&gt;
/home/rayabhik/.local/lib/python3.5/site-packages/spacy/data/en

You can now load the model via spacy.load('en')
</code></pre>

<p>Now, when I try running my code, on the line:</p>

<pre><code>    from spacy.en import English
</code></pre>

<p>it gives me the following error:</p>

<pre><code>ImportError: No module named 'spacy.en'
</code></pre>

<p>I've looked on Stackexchange and the closest is:  <a href="https://stackoverflow.com/questions/34842052/import-error-with-spacy-no-module-named-en">Import error with spacy: &quot;No module named en&quot;</a>
which does not solve my problem.</p>

<p>Any help would be appreciated. Thanks.</p>

<p>Edit: I might have solved this by doing the following:</p>

<pre><code> Python 3.5.2 (default, Sep 14 2017, 22:51:06) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
&gt;&gt;&gt; import spacy
&gt;&gt;&gt; spacy.load('en')
&lt;spacy.lang.en.English object at 0x7ff414e1e0b8&gt;
</code></pre>

<p>and then using:</p>

<pre><code>from spacy.lang.en import English
</code></pre>

<p>I'm still keeping this open in case there are any other answers.</p>

               ''',
               'so_title': "ImportError: No module named 'spacy.en'",
               'so_tags': '<python><spacy>',
               'rel_num': 36}

spaCy_data6 = {'api': 'spaCy',
               'so_body': '''<p>I am new to spacy and I want to use its lemmatizer function, but I don't know how to use it, like I into strings of word, which will return the string with the basic form the words.</p>

<p>Examples:</p>

<ul>
<li>'words'=> 'word'</li>
<li>'did' => 'do'</li>
</ul>

<p>Thank you.</p>
               ''',
               'so_title': "how to use spacy lemmatizer to get a word into basic form",
               'so_tags': '<python><nltk><spacy><lemmatization>',
               'rel_num': 36}

spaCy_data7 = {'api': 'spaCy',
               'so_body': '''<p>what is difference between <code>spacy.load('en_core_web_sm')</code> and <code>spacy.load('en')</code>? <a href="https://stackoverflow.com/questions/50487495/what-is-difference-between-en-core-web-sm-en-core-web-mdand-en-core-web-lg-mod">This link</a> explains different model sizes. But i am still not clear how <code>spacy.load('en_core_web_sm')</code> and <code>spacy.load('en')</code> differ</p>

<p><code>spacy.load('en')</code> runs fine for me. But the <code>spacy.load('en_core_web_sm')</code> throws error</p>

<p>i have installed <code>spacy</code>as below. when i go to jupyter notebook and run command <code>nlp = spacy.load('en_core_web_sm')</code> I get the below error </p>
''',
               'so_title': "Could not install packages due to an EnvironmentError: [Errno 28] No space left on device",
               'so_tags': '<python><python-3.x><nlp><spacy>',
               'rel_num': 36}

spaCy_data8 = {'api': 'spaCy',
               'so_body': '''<p>I have installed <strong>spaCy</strong> with python for my NLP project.</p>

<p>I have installed that using <code>pip</code>.  How can I verify installed spaCy version?</p>

<p>using </p>

<pre><code>pip install -U spacy
</code></pre>

<p>What is command to verify installed spaCy version?</p>''',
               'so_title': "How to verify installed spaCy version?",
               'so_tags': '<python><nlp><pip><version><spacy>',
               'rel_num': 36}

spaCy_data9 = {'api': 'spaCy',
               'so_body': '''<p>Say I have a dataset, like</p>

<pre><code>iris = pd.DataFrame(sns.load_dataset('iris'))
</code></pre>

<p>I can use <code>Spacy</code> and <code>.apply</code> to parse a string column into tokens (my real dataset has >1 word/token per entry of course)</p>

<pre><code>import spacy # (I have version 1.8.2)
nlp = spacy.load('en')
iris['species_parsed'] = iris['species'].apply(nlp)
</code></pre>

<p>result:</p>

<pre><code>   sepal_length   ... species    species_parsed
0           1.4   ... setosa          (setosa)
1           1.4   ... setosa          (setosa)
2           1.3   ... setosa          (setosa)
</code></pre>

<p>I can also use this convenient multiprocessing function (<a href="http://www.racketracer.com/2016/07/06/pandas-in-parallel/" rel="noreferrer">thanks to this blogpost</a>) to do most arbitrary apply functions on a dataframe in parallel:</p>

<pre><code>from multiprocessing import Pool, cpu_count
def parallelize_dataframe(df, func, num_partitions):

    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_partitions)
    df = pd.concat(pool.map(func, df_split))

    pool.close()
    pool.join()
    return df
</code></pre>

<p>for example:</p>

<pre><code>def my_func(df):
    df['length_of_word'] = df['species'].apply(lambda x: len(x))
    return df

num_cores = cpu_count()
iris = parallelize_dataframe(iris, my_func, num_cores)
</code></pre>

<p>result:</p>

<pre><code>   sepal_length species  length_of_word
0           5.1  setosa               6
1           4.9  setosa               6
2           4.7  setosa               6
</code></pre>

<p>...But for some reason, I can't apply the Spacy parser to a dataframe using multiprocessing this way. </p>

<pre><code>def add_parsed(df):
    df['species_parsed'] = df['species'].apply(nlp)
    return df

iris = parallelize_dataframe(iris, add_parsed, num_cores)
</code></pre>

<p>result:</p>

<pre><code>   sepal_length species  length_of_word species_parsed
0           5.1  setosa               6             ()
1           4.9  setosa               6             ()
2           4.7  setosa               6             ()
</code></pre>

<p>Is there some other way to do this? I'm loving Spacy for NLP but I have a lot of text data and so I'd like to parallelize some processing functions, but ran into this issue.</p>

               ''',
               'so_title': "Applying Spacy Parser to Pandas DataFrame w/ Multiprocessing",
               'so_tags': '<python><spacy>',
               'rel_num': 36}

spaCy_data10 = {'api': 'spaCy',
                'so_body': '''<p>I am trying to evaluate a trained NER Model created using <a href="https://spacy.io/docs/usage/training-ner" rel="noreferrer">spacy lib</a>.
 Normally for these kind of problems you can use f1 score (a ratio between precision and recall). I could not find in the documentation an accuracy function for a trained NER model. </p>

<p>I am not sure if it's correct but I am trying to do it with the following way(example) and using <code>f1_score</code> from <code>sklearn</code>:</p>

<pre><code>from sklearn.metrics import f1_score
import spacy
from spacy.gold import GoldParse


nlp = spacy.load("en") #load NER model
test_text = "my name is John" # text to test accuracy
doc_to_test = nlp(test_text) # transform the text to spacy doc format

# we create a golden doc where we know the tagged entity for the text to be tested
doc_gold_text= nlp.make_doc(test_text)
entity_offsets_of_gold_text = [(11, 15,"PERSON")]
gold = GoldParse(doc_gold_text, entities=entity_offsets_of_gold_text)

# bring the data in a format acceptable for sklearn f1 function
y_true = ["PERSON" if "PERSON" in x else 'O' for x in gold.ner]
y_predicted = [x.ent_type_ if x.ent_type_ !='' else 'O' for x in doc_to_test]
f1_score(y_true, y_predicted, average='macro')`[1]
&gt; 1.0
</code></pre>

<p>Any thoughts are or insights are useful. </p>
                ''',
                'so_title': "Evaluation in a Spacy NER model",
                'so_tags': '<python><spacy>',
                'rel_num': 36}