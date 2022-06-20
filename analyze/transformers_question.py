# coding=utf-8
# @Author : Eric


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

Transformers_data2 = {'api': "Transformers",
                      'so_body': '''<p>There is a problem we are trying to solve where we want to do a semantic search on our set of data,
i.e we have a domain-specific data (example: sentences talking about automobiles)</p>

<p>Our data is just a bunch of sentences and what we want is to give a phrase and get back the sentences which are:</p>

<ol>
<li>Similar to that phrase</li>
<li>Has a part of a sentence that is similar to the phrase</li>
<li>A sentence which is having contextually similar meanings </li>
</ol>

<p><br/></p>

<p>Let me try giving you an example suppose I search for the phrase "Buying Experience", I should get the sentences like:</p>

<ul>
<li>I never thought car buying could take less than 30 minutes to sign
and buy.</li>
<li><p>I found a car that i liked and the purchase process was<br>
straightforward and easy</p></li>
<li><p>I absolutely hated going car shopping, but today i’m glad i did</p></li>
</ul>

<p><br/>
I want to lay emphasis on the fact that we are looking for <strong>contextual similarity</strong> and not just a brute force word search.</p>

<p>If the sentence uses different words then also it should be able to find it.</p>

<p>Things that we have already tried:</p>

<ol>
<li><p><a href="https://www.opensemanticsearch.org/" rel="noreferrer">Open Semantic Search</a> the problem we faced here is generating ontology from the data we have, or
for that sake searching for available ontology from different domains of our interest.</p></li>
<li><p>Elastic Search(BM25 + Vectors(tf-idf)), we tried this where it gave a few sentences but precision was not that great. The accuracy was bad
as well. We tried against a human-curated dataset, it was able to get around 10% of the sentences only.</p></li>
<li><p>We tried different embeddings like the once mentioned in <a href="https://github.com/UKPLab/sentence-transformers" rel="noreferrer">sentence-transformers</a> and also went through the <a href="https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.p" rel="noreferrer">example</a> and tried evaluating against our human-curated set
and that also had very low accuracy.</p></li>
<li><p>We tried <a href="https://towardsdatascience.com/elmo-contextual-language-embedding-335de2268604" rel="noreferrer">ELMO</a>. This was better but still lower accuracy than we expected and there is a
cognitive load to decide the cosine value below which we shouldn't consider the sentences. This even applies to point 3.</p></li>
</ol>

<p>Any help will be appreciated. Thanks a lot for the help in advance</p>

                      ''',
                      'so_title': "How to build semantic search for a given domain",
                      'so_tags': "<python><elasticsearch><nlp><sentence-similarity><huggingface-transformers>",
                      'rel_num': 26}

Transformers_data3 = {'api': "Transformers",
                      'so_body': '''<p>Running the below code downloads a model - does anyone know what folder it downloads it to?</p>

<pre><code>!pip install -q transformers
from transformers import pipeline
model = pipeline('fill-mask')
</code></pre>

                      ''',
                      'so_title': "Where does hugging face's transformers save models?",
                      'so_tags': "<huggingface-transformers>",
                      'rel_num': 26}

Transformers_data4 = {'api': "Transformers",
                      'so_body': '''<p>I am using the HuggingFace Transformers package to access pretrained models. As my use case needs functionality for both English and Arabic, I am using the <a href="https://github.com/google-research/bert/blob/master/multilingual.md" rel="noreferrer">bert-base-multilingual-cased</a> pretrained model. I need to be able to compare the similarity of sentences using something such as cosine similarity. To use  this, I first need to get an embedding vector for each sentence, and can then compute the cosine similarity.</p>

<p>Firstly, what is the best way to extratc the semantic embedding from the BERT model? Would taking the last hidden state of the model after being fed the sentence suffice?</p>

<pre><code>import torch
from transformers import BertModel, BertTokenizer

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-multilingual-cased'

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

sentence = 'this is a test sentence'

input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
with torch.no_grad():
    output_tuple = model(input_ids)
    last_hidden_states = output_tuple[0]

print(last_hidden_states.size(), last_hidden_states)
</code></pre>

<p>Secondly, if this is a sufficient way to get embeddings from my sentence, I now have another problem where the embedding vectors have different lengths depending on the length of the original sentence. The shapes output are <code>[1, n, vocab_size]</code>, where <code>n</code> can have any value. </p>

<p>In order to compute two vectors' cosine similarity, they need to be the same  length. How can I do this here? Could something as naive as first summing across <code>axis=1</code> still work? What other options do I have? </p>
                      ''',
                      'so_title': "How to compare sentence similarities using embeddings from BERT",
                      'so_tags': "<python><vector><nlp><cosine-similarity><huggingface-transformers>",
                      'rel_num': 26}

Transformers_data5 = {'api': "Transformers",
                      'so_body': '''<p>From the documentation <a href="https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained" rel="noreferrer">for from_pretrained</a>, I understand I don't have to download the pretrained vectors every time, I can save them and load from disk with this syntax:</p>
<pre><code>  - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
  - (not applicable to all derived classes, deprecated) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.
</code></pre>
<p>So, I went to the model hub:</p>
<ul>
<li><a href="https://huggingface.co/models" rel="noreferrer">https://huggingface.co/models</a></li>
</ul>
<p>I found the model I wanted:</p>
<ul>
<li><a href="https://huggingface.co/bert-base-cased" rel="noreferrer">https://huggingface.co/bert-base-cased</a></li>
</ul>
<p>I downloaded it from the link they provided to this repository:</p>
<blockquote>
<p>Pretrained model on English language using a masked language modeling
(MLM) objective. It was introduced in this paper and first released in
this repository. This model is case-sensitive: it makes a difference
between english and English.</p>
</blockquote>
<p>Stored it in:</p>
<pre><code>  /my/local/models/cased_L-12_H-768_A-12/
</code></pre>
<p>Which contains:</p>
<pre><code> ./
 ../
 bert_config.json
 bert_model.ckpt.data-00000-of-00001
 bert_model.ckpt.index
 bert_model.ckpt.meta
 vocab.txt
</code></pre>
<p>So, now I have the following:</p>
<pre><code>  PATH = '/my/local/models/cased_L-12_H-768_A-12/'
  tokenizer = BertTokenizer.from_pretrained(PATH, local_files_only=True)
</code></pre>
<p>And I get this error:</p>
<pre><code>&gt;           raise EnvironmentError(msg)
E           OSError: Can't load config for '/my/local/models/cased_L-12_H-768_A-12/'. Make sure that:
E           
E           - '/my/local/models/cased_L-12_H-768_A-12/' is a correct model identifier listed on 'https://huggingface.co/models'
E           
E           - or '/my/local/models/cased_L-12_H-768_A-12/' is the correct path to a directory containing a config.json file
</code></pre>
<p>Similarly for when I link to the config.json directly:</p>
<pre><code>  PATH = '/my/local/models/cased_L-12_H-768_A-12/bert_config.json'
  tokenizer = BertTokenizer.from_pretrained(PATH, local_files_only=True)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location=&quot;cpu&quot;)
            except Exception:
                raise OSError(
&gt;                   &quot;Unable to load weights from pytorch checkpoint file. &quot;
                    &quot;If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. &quot;
                )
E               OSError: Unable to load weights from pytorch checkpoint file. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True.
</code></pre>
<p>What should I do differently to get huggingface to use my local pretrained model?</p>
<h1>Update to address the comments</h1>
<pre><code>YOURPATH = '/somewhere/on/disk/'

name = 'transfo-xl-wt103'
tokenizer = TransfoXLTokenizerFast(name)
model = TransfoXLModel.from_pretrained(name)
tokenizer.save_pretrained(YOURPATH)
model.save_pretrained(YOURPATH)

&gt;&gt;&gt; Please note you will not be able to load the save vocabulary in Rust-based TransfoXLTokenizerFast as they don't share the same structure.
('/somewhere/on/disk/vocab.bin', '/somewhere/on/disk/special_tokens_map.json', '/somewhere/on/disk/added_tokens.json')

</code></pre>
<p>So all is saved, but then....</p>
<pre><code>YOURPATH = '/somewhere/on/disk/'
TransfoXLTokenizerFast.from_pretrained('transfo-xl-wt103', cache_dir=YOURPATH, local_files_only=True)

    &quot;Cannot find the requested files in the cached path and outgoing traffic has been&quot;
ValueError: Cannot find the requested files in the cached path and outgoing traffic has been disabled. To enable model look-ups and downloads online, set 'local_files_only' to False.
</code></pre>

                      ''',
                      'so_title': "Load a pre-trained model from disk with Huggingface Transformers",
                      'so_tags': "<huggingface-transformers>",
                      'rel_num': 26}

Transformers_data6 = {'api': "Transformers",
                      'so_body': '''<p>I'm following the transformer's pretrained model <a href="https://huggingface.co/joeddav/xlm-roberta-large-xnli?text=%0A&amp;candidate_labels=&amp;multi_class=true" rel="noreferrer">xlm-roberta-large-xnli</a> example</p>
<pre><code>from transformers import pipeline
classifier = pipeline(&quot;zero-shot-classification&quot;,
                      model=&quot;joeddav/xlm-roberta-large-xnli&quot;)
</code></pre>
<p>and I get the following error</p>
<pre><code>ValueError: Couldn't instantiate the backend tokenizer from one of: (1) a `tokenizers` library serialization file, (2) a slow tokenizer instance to convert or (3) an equivalent slow tokenizer class to instantiate and convert. You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
</code></pre>
<p>I'm using Transformers version <code>'4.1.1'</code></p>
                      ''',
                      'so_title': "Transformers v4.x: Convert slow tokenizer to fast tokenizer",
                      'so_tags': "<python><nlp><huggingface-transformers><huggingface-tokenizers>",
                      'rel_num': 26}

Transformers_data7 = {'api': "Transformers",
                      'so_body': '''<pre class="lang-py prettyprint-override"><code>from transformers import AutoModel, AutoTokenizer

tokenizer1 = AutoTokenizer.from_pretrained("roberta-base")
tokenizer2 = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "A Titan RTX has 24GB of VRAM"
print(tokenizer1.tokenize(sequence))
print(tokenizer2.tokenize(sequence))
</code></pre>

<p>Output:</p>

<p>['A', 'ĠTitan', 'ĠRTX', 'Ġhas', 'Ġ24', 'GB', 'Ġof', 'ĠVR', 'AM']</p>

<p>['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']</p>

<p>Bert model uses WordPiece tokenizer. Any word that does not occur in the WordPiece vocabulary is broken down into sub-words greedily. For example, 'RTX' is broken into 'R', '##T' and '##X' where ## indicates it is a subtoken. </p>

<p>Roberta uses BPE tokenizer but I'm unable to understand </p>

<p>a) how BPE tokenizer works? </p>

<p>b) what does G represents in each of tokens?</p>
                      ''',
                      'so_title': "Difficulty in understanding the tokenizer used in Roberta model",
                      'so_tags': "<nlp><pytorch><huggingface-transformers><bert-language-model>",
                      'rel_num': 26}

Transformers_data8 = {'api': "Transformers",
                      'so_body': '''<p>I was trying the hugging face gpt2 model. I have seen the <a href="https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py" rel="noreferrer"><code>run_generation.py</code> script</a>, which generates a sequence of tokens given a prompt. I am aware that we can use GPT2 for NLG.</p>
<p>In my use case, I wish to determine the probability distribution for (only) the immediate next word following the given prompt. Ideally this distribution would be over the entire vocab.</p>
<p>For example, given the prompt: &quot;How are &quot;, it should give a probability distribution where &quot;you&quot; or &quot;they&quot; have the some high floating point values and other vocab words have very low floating values.</p>
<p>How to do this using hugging face transformers? If it is not possible in hugging face, is there any other transformer model that does this?</p>
                      ''',
                      'so_title': "How to get immediate next word probability using GPT2 model?",
                      'so_tags': "<transformer><huggingface-transformers>",
                      'rel_num': 26}

Transformers_data9 = {'api': "Transformers",
                      'so_body': '''<p>I would like to create a minibatch by encoding multiple sentences using transform.BertTokenizer. It seems working for a single sentence. How to make it work for several sentences?</p>
<pre><code>from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize a single sentence seems working
tokenizer.encode('this is the first sentence')
&gt;&gt;&gt; [2023, 2003, 1996, 2034, 6251]

# tokenize two sentences
tokenizer.encode(['this is the first sentence', 'another setence'])
&gt;&gt;&gt; [100, 100] # expecting 7 tokens
</code></pre>
                      ''',
                      'so_title': "How to encode multiple setence using transformers.BertTokenizer?",
                      'so_tags': "<word-embedding><huggingface-transformers><huggingface-tokenizers>",
                      'rel_num': 26}

Transformers_data10 = {'api': "Transformers",
                       'so_body': '''<pre><code>ner_model = pipeline('ner', model=model, tokenizer=tokenizer, device=0, grouped_entities=True)
</code></pre>
<p>the <em><strong>device</strong></em> indicated pipeline to use no_gpu=0(only using GPU), please show me how to use multi-gpu.</p>
                       ''',
                       'so_title': "How to use transformers pipeline with multi-gpu?",
                       'so_tags': "<python><huggingface-transformers>",
                       'rel_num': 26}
