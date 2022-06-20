# coding=utf-8
# @Author : Eric


gensim_data1 = {'api': 'gensim',
                'so_body': '''<p>According to the <a href="http://radimrehurek.com/gensim/models/word2vec.html" rel="noreferrer">Gensim Word2Vec</a>, I can use the word2vec model in gensim package to calculate the similarity between 2 words.</p>

    <p>e.g.</p>

    <pre><code>trained_model.similarity('woman', 'man') 
    0.73723527
    </code></pre>

    <p>However, the word2vec model fails to predict the sentence similarity. I find out the LSI model with sentence similarity in gensim, but, which doesn't seem that can be combined with word2vec model. The length of corpus of each sentence I have is not very long (shorter than 10 words).  So, are there any simple ways to achieve the goal?</p>
    ''',
                'so_title': 'How to calculate the sentence similarity using word2vec model of gensim with python',
                'so_tags': '<python><gensim><word2vec>',
                'rel_num': 26}

gensim_data2 = {'api': 'gensim',
                'so_body': '''<p>From the <a href="https://code.google.com/p/word2vec/">word2vec</a> site I can download GoogleNews-vectors-negative300.bin.gz.  The .bin file (about 3.4GB) is a binary format not useful to me.  Tomas Mikolov <a href="https://groups.google.com/d/msg/word2vec-toolkit/lxbl_MB29Ic/g4uEz5rNV08J">assures us</a> that "It should be fairly straightforward to convert the binary format to text format (though that will take more disk space). Check the code in the distance tool, it's rather trivial to read the binary file."  Unfortunately, I don't know enough C to understand <a href="http://word2vec.googlecode.com/svn/trunk/distance.c">http://word2vec.googlecode.com/svn/trunk/distance.c</a>.</p>

<p>Supposedly <a href="http://radimrehurek.com/2014/02/word2vec-tutorial/">gensim</a> can do this also, but all the tutorials I've found seem to be about converting <em>from</em> text, not the other way.</p>

<p>Can someone suggest modifications to the C code or instructions for gensim to emit text?</p>

    ''',
                'so_title': 'Convert word2vec bin file to text',
                'so_tags': '<python><c><gensim><word2vec>',
                'rel_num': 26}

gensim_data3 = {'api': 'gensim',
                'so_body': '''<p>How to get document vectors of two text documents using Doc2vec?
I am new to this, so it would be helpful if someone could point me in the right direction / help me with some tutorial</p>

<p>I am using gensim.</p>

<pre><code>doc1=["This is a sentence","This is another sentence"]
documents1=[doc.strip().split(" ") for doc in doc1 ]
model = doc2vec.Doc2Vec(documents1, size = 100, window = 300, min_count = 10, workers=4)
</code></pre>

<p>I get </p>

<blockquote>
  <p>AttributeError: 'list' object has no attribute 'words'</p>
</blockquote>

<p>whenever I run this.</p>
                ''',
                'so_title': 'Doc2vec: How to get document vectors',
                'so_tags': '<python><gensim><word2vec>',
                'rel_num': 26}

gensim_data4 = {'api': 'gensim',
                'so_body': '''<p>I'm trying to compare my implementation of Doc2Vec (via tf) and gensims implementation. It seems atleast visually that the gensim ones are performing better.</p>

<p>I ran the following code to train the gensim model and the one below that for tensorflow model. My questions are as follows:</p>

<ol>
<li>Is my tf implementation of Doc2Vec correct. Basically is it supposed to be concatenating the word vectors and the document vector to predict the middle word in a certain context?</li>
<li>Does the <code>window=5</code> parameter in gensim mean that I am using two words on either side to predict the middle one? Or is it 5 on either side. Thing is there are quite a few documents that are smaller than length 10.</li>
<li>Any insights as to why Gensim is performing better? Is my model any different to how they implement it?</li>
<li>Considering that this is effectively a matrix factorisation problem, why is the TF model even getting an answer? There are infinite solutions to this since its a rank deficient problem. &lt;- This last question is simply a bonus.</li>
</ol>

<h3>Gensim</h3>

<pre><code>model = Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=10, hs=0, min_count=2, workers=cores)
model.build_vocab(corpus)
epochs = 100
for i in range(epochs):
    model.train(corpus)
</code></pre>

<h3>TF</h3>

<pre><code>batch_size = 512
embedding_size = 100 # Dimension of the embedding vector.
num_sampled = 10 # Number of negative examples to sample.


graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_word_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_doc_dataset = tf.placeholder(tf.int32, shape=[batch_size/context_window])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size/context_window, 1])

    # The variables   
    word_embeddings =  tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
    doc_embeddings = tf.Variable(tf.random_uniform([len_docs,embedding_size],-1.0,1.0))
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, (context_window+1)*embedding_size],
                             stddev=1.0 / np.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    ###########################
    # Model.
    ###########################
    # Look up embeddings for inputs and stack words side by side
    embed_words = tf.reshape(tf.nn.embedding_lookup(word_embeddings, train_word_dataset),
                            shape=[int(batch_size/context_window),-1])
    embed_docs = tf.nn.embedding_lookup(doc_embeddings, train_doc_dataset)
    embed = tf.concat(1,[embed_words, embed_docs])
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                   train_labels, num_sampled, vocabulary_size))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
</code></pre>

<h2>Update:</h2>

<p>Check out the jupyter notebook <a href="https://github.com/sachinruk/doc2vec_tf" rel="noreferrer">here</a> (I have both models working and tested in here). It still feels like the gensim model is performing better in this initial analysis.</p>

                ''',
                'so_title': 'gensim Doc2Vec vs tensorflow Doc2Vec',
                'so_tags': '<python><tensorflow><nlp><gensim><doc2vec>',
                'rel_num': 26}

gensim_data5 = {'api': 'gensim',
                'so_body': '''<p>After training a word2vec model using python <a href="http://radimrehurek.com/gensim/models/word2vec.html" rel="noreferrer">gensim</a>, how do you find the number of words in the model's vocabulary?</p>

                ''',
                'so_title': 'gensim word2vec: Find number of words in vocabulary',
                'so_tags': '<python><neural-network><nlp><gensim><word2vec>',
                'rel_num': 26}

gensim_data6 = {'api': 'gensim',
                'so_body': '''<p>I want to load a pre-trained word2vec embedding with gensim into a PyTorch embedding layer.</p>

<p>So my question is, how do I get the embedding weights loaded by gensim into the PyTorch embedding layer.</p>

<p>Thanks in Advance!</p>
                ''',
                'so_title': 'PyTorch / Gensim - How to load pre-trained word embeddings',
                'so_tags': '<python><neural-network><pytorch><gensim><embedding>',
                'rel_num': 26}

gensim_data7 = {'api': 'gensim',
                'so_body': '''<p>From <a href="https://stackoverflow.com/questions/15502802/creating-a-subset-of-words-from-a-corpus-in-r">Creating a subset of words from a corpus in R</a>, the answerer can easily convert a <code>term-document matrix</code> into a word cloud easily.</p>

<p>Is there a similar function from python libraries that takes either a raw word textfile or <code>NLTK</code> corpus or <code>Gensim</code> Mmcorpus into a word cloud?</p>

<p>The result will look somewhat like this:
<img src="https://i.stack.imgur.com/ieYK2.png" alt="enter image description here"></p>
                ''',
                'so_title': 'How to create a word cloud from a corpus in Python?',
                'so_tags': '<python><nltk><corpus><gensim><word-cloud>',
                'rel_num': 26}

gensim_data8 = {'api': 'gensim',
                'so_body': '''<p>I have trained a word2vec model using a corpus of documents with Gensim. Once the model is training, I am writing the following piece of code to get the raw feature vector of a word say "view".</p>

<pre><code>myModel["view"]
</code></pre>

<p>However, I get a KeyError for the word which is probably because this doesn't exist as a key in the list of keys indexed by word2vec. How can I check if a key exits in the index before trying to get the raw feature vector?</p>
                ''',
                'so_title': 'How to check if a key exists in a word2vec trained model or not',
                'so_tags': '<python><gensim><word2vec>',
                'rel_num': 26}

gensim_data9 = {'api': 'gensim',
                'so_body': '''<p>For preprocessing the corpus I was planing to extarct common phrases from the corpus, for this I tried using <strong>Phrases</strong> model in gensim, I tried below code but it's not giving me desired output.</p>

<p><strong>My code</strong></p>

<pre><code>from gensim.models import Phrases
documents = ["the mayor of new york was there", "machine learning can be useful sometimes"]

sentence_stream = [doc.split(" ") for doc in documents]
bigram = Phrases(sentence_stream)
sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
print(bigram[sent])
</code></pre>

<p><strong>Output</strong></p>

<pre><code>[u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
</code></pre>

<p><strong>But it should come as</strong> </p>

<pre><code>[u'the', u'mayor', u'of', u'new_york', u'was', u'there']
</code></pre>

<p>But when I tried to print vocab of train data, I can see bigram, but its not working with test data, where I am going wrong?</p>

<pre><code>print bigram.vocab

defaultdict(&lt;type 'int'&gt;, {'useful': 1, 'was_there': 1, 'learning_can': 1, 'learning': 1, 'of_new': 1, 'can_be': 1, 'mayor': 1, 'there': 1, 'machine': 1, 'new': 1, 'was': 1, 'useful_sometimes': 1, 'be': 1, 'mayor_of': 1, 'york_was': 1, 'york': 1, 'machine_learning': 1, 'the_mayor': 1, 'new_york': 1, 'of': 1, 'sometimes': 1, 'can': 1, 'be_useful': 1, 'the': 1}) 
</code></pre>
                ''',
                'so_title': 'How to extract phrases from corpus using gensim',
                'so_tags': '<python><nlp><gensim>',
                'rel_num': 26}

gensim_data10 = {'api': 'gensim',
                 'so_body': '''<p>I want to calculate tf-idf from the documents below. I'm using python and pandas.</p>

<pre><code>import pandas as pd
df = pd.DataFrame({'docId': [1,2,3], 
               'sent': ['This is the first sentence','This is the second sentence', 'This is the third sentence']})
</code></pre>

<p>First, I thought I would need to get word_count for each row. So I wrote a simple function:</p>

<pre><code>def word_count(sent):
    word2cnt = dict()
    for word in sent.split():
        if word in word2cnt: word2cnt[word] += 1
        else: word2cnt[word] = 1
return word2cnt
</code></pre>

<p>And then, I applied it to each row.</p>

<pre><code>df['word_count'] = df['sent'].apply(word_count)
</code></pre>

<p>But now I'm lost. I know there's an easy method to calculate tf-idf if I use Graphlab, but I want to stick with an open source option. Both Sklearn and gensim look overwhelming. What's the simplest solution to get tf-idf?</p>
                 ''',
                 'so_title': 'How to get tfidf with pandas dataframe?',
                 'so_tags': '<python><pandas><scikit-learn><tf-idf><gensim>',
                 'rel_num': 26}