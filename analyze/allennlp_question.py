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
                  'rel_num': 26}

allennlp_data2 = {'api': 'allennlp',
                  'so_body': '''<p>I am trying to install allennlp on my mac. I have tried installing macOS headers which solved the missing headers problem but now i am experiencing new problems. </p>

<p>The error when i run <code>pip install allennlp</code>:</p>

<pre><code>Running setup.py bdist_wheel for jsonnet ... error
  Complete output from command /anaconda3/bin/python -u -c "import setuptools, tokenize;__file__='/private/var/folders/qf/jkn4v43j08xgst0r9yxyl0dc0000gn/T/pip-install-i4nyb384/jsonnet/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" bdist_wheel -d /private/var/folders/qf/jkn4v43j08xgst0r9yxyl0dc0000gn/T/pip-wheel-eof7cc6k --python-tag cp37:
  running bdist_wheel
  running build
  running build_ext
  x86_64-apple-darwin13.4.0-clang++ -c -march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe -stdlib=libc++ -fvisibility-inlines-hidden -std=c++14 -fmessage-length=0 core/desugarer.cpp -o core/desugarer.o
  In file included from core/desugarer.cpp:17:
  In file included from /Library/Developer/CommandLineTools/usr/include/c++/v1/cassert:21:
  In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/usr/include/assert.h:44:
  /Library/Developer/CommandLineTools/usr/include/c++/v1/stdlib.h:111:82: error: use of undeclared identifier 'labs'; did you mean 'abs'?
  inline _LIBCPP_INLINE_VISIBILITY long      abs(     long __x) _NOEXCEPT {return  labs(__x);}
                                                                                   ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/stdlib.h:111:44: note: 'abs' declared here
  inline _LIBCPP_INLINE_VISIBILITY long      abs(     long __x) _NOEXCEPT {return  labs(__x);}
                                             ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/stdlib.h:113:81: error: use of undeclared identifier 'llabs'
  inline _LIBCPP_INLINE_VISIBILITY long long abs(long long __x) _NOEXCEPT {return llabs(__x);}
                                                                                  ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/stdlib.h:116:35: error: unknown type name 'ldiv_t'
  inline _LIBCPP_INLINE_VISIBILITY  ldiv_t div(     long __x,      long __y) _NOEXCEPT {return  ldiv(__x, __y);}
                                    ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/stdlib.h:116:95: error: use of undeclared identifier 'ldiv'; did you mean 'div'?
  inline _LIBCPP_INLINE_VISIBILITY  ldiv_t div(     long __x,      long __y) _NOEXCEPT {return  ldiv(__x, __y);}
                                                                                                ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/stdlib.h:116:42: note: 'div' declared here
  inline _LIBCPP_INLINE_VISIBILITY  ldiv_t div(     long __x,      long __y) _NOEXCEPT {return  ldiv(__x, __y);}
                                           ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/stdlib.h:118:34: error: unknown type name 'lldiv_t'
  inline _LIBCPP_INLINE_VISIBILITY lldiv_t div(long long __x, long long __y) _NOEXCEPT {return lldiv(__x, __y);}
                                   ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/stdlib.h:118:94: error: use of undeclared identifier 'lldiv'
  inline _LIBCPP_INLINE_VISIBILITY lldiv_t div(long long __x, long long __y) _NOEXCEPT {return lldiv(__x, __y);}
                                                                                               ^
  In file included from core/desugarer.cpp:19:
  In file included from /Library/Developer/CommandLineTools/usr/include/c++/v1/algorithm:642:
  In file included from /Library/Developer/CommandLineTools/usr/include/c++/v1/cstring:61:
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:74:64: error: use of undeclared identifier 'strchr'
  char* __libcpp_strchr(const char* __s, int __c) {return (char*)strchr(__s, __c);}
                                                                 ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:81:75: error: use of undeclared identifier 'strpbrk'
  char* __libcpp_strpbrk(const char* __s1, const char* __s2) {return (char*)strpbrk(__s1, __s2);}
                                                                            ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:88:65: error: use of undeclared identifier 'strrchr'; did you mean 'strchr'?
  char* __libcpp_strrchr(const char* __s, int __c) {return (char*)strrchr(__s, __c);}
                                                                  ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:76:13: note: 'strchr' declared here
  const char* strchr(const char* __s, int __c) {return __libcpp_strchr(__s, __c);}
              ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:95:76: error: use of undeclared identifier 'memchr'
  void* __libcpp_memchr(const void* __s, int __c, size_t __n) {return (void*)memchr(__s, __c, __n);}
                                                                             ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:102:74: error: use of undeclared identifier 'strstr'; did you mean 'strchr'?
  char* __libcpp_strstr(const char* __s1, const char* __s2) {return (char*)strstr(__s1, __s2);}
                                                                           ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:78:13: note: 'strchr' declared here
        char* strchr(      char* __s, int __c) {return __libcpp_strchr(__s, __c);}
              ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:102:74: error: no matching function for call to 'strchr'
  char* __libcpp_strstr(const char* __s1, const char* __s2) {return (char*)strstr(__s1, __s2);}
                                                                           ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:78:13: note: candidate disabled: &lt;no message provided&gt;
        char* strchr(      char* __s, int __c) {return __libcpp_strchr(__s, __c);}
              ^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:102:81: error: cannot initialize a parameter of type 'char *' with an lvalue of type 'const char *'
  char* __libcpp_strstr(const char* __s1, const char* __s2) {return (char*)strstr(__s1, __s2);}
                                                                                  ^~~~
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:78:32: note: passing argument to parameter '__s' here
        char* strchr(      char* __s, int __c) {return __libcpp_strchr(__s, __c);}
                                 ^
  In file included from core/desugarer.cpp:19:
  In file included from /Library/Developer/CommandLineTools/usr/include/c++/v1/algorithm:642:
  /Library/Developer/CommandLineTools/usr/include/c++/v1/cstring:70:9: error: no member named 'memcpy' in the global namespace; did you mean 'memchr'?
  using ::memcpy;
        ~~^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:97:13: note: 'memchr' declared here
  const void* memchr(const void* __s, int __c, size_t __n) {return __libcpp_memchr(__s, __c, __n);}
              ^
  In file included from core/desugarer.cpp:19:
  In file included from /Library/Developer/CommandLineTools/usr/include/c++/v1/algorithm:642:
  /Library/Developer/CommandLineTools/usr/include/c++/v1/cstring:71:9: error: no member named 'memmove' in the global namespace
  using ::memmove;
        ~~^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/cstring:72:9: error: no member named 'strcpy' in the global namespace; did you mean 'strchr'?
  using ::strcpy;
        ~~^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:76:13: note: 'strchr' declared here
  const char* strchr(const char* __s, int __c) {return __libcpp_strchr(__s, __c);}
              ^
  In file included from core/desugarer.cpp:19:
  In file included from /Library/Developer/CommandLineTools/usr/include/c++/v1/algorithm:642:
  /Library/Developer/CommandLineTools/usr/include/c++/v1/cstring:73:9: error: no member named 'strncpy' in the global namespace
  using ::strncpy;
        ~~^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/cstring:74:9: error: no member named 'strcat' in the global namespace; did you mean 'strchr'?
  using ::strcat;
        ~~^
  /Library/Developer/CommandLineTools/usr/include/c++/v1/string.h:76:13: note: 'strchr' declared here
  const char* strchr(const char* __s, int __c) {return __libcpp_strchr(__s, __c);}
              ^
  In file included from core/desugarer.cpp:19:
  In file included from /Library/Developer/CommandLineTools/usr/include/c++/v1/algorithm:642:
  /Library/Developer/CommandLineTools/usr/include/c++/v1/cstring:75:9: error: no member named 'strncat' in the global namespace
  using ::strncat;
        ~~^
  fatal error: too many errors emitted, stopping now [-ferror-limit=]
  20 errors generated.
  make: *** [Makefile:118: core/desugarer.o] Error 1
  Traceback (most recent call last):
    File "&lt;string&gt;", line 1, in &lt;module&gt;
    File "/private/var/folders/qf/jkn4v43j08xgst0r9yxyl0dc0000gn/T/pip-install-i4nyb384/jsonnet/setup.py", line 75, in &lt;module&gt;
      test_suite="python._jsonnet_test",
    File "/anaconda3/lib/python3.7/site-packages/setuptools/__init__.py", line 143, in setup
      return distutils.core.setup(**attrs)
    File "/anaconda3/lib/python3.7/distutils/core.py", line 148, in setup
      dist.run_commands()
    File "/anaconda3/lib/python3.7/distutils/dist.py", line 966, in run_commands
      self.run_command(cmd)
    File "/anaconda3/lib/python3.7/distutils/dist.py", line 985, in run_command
      cmd_obj.run()
    File "/anaconda3/lib/python3.7/site-packages/wheel/bdist_wheel.py", line 188, in run
      self.run_command('build')
    File "/anaconda3/lib/python3.7/distutils/cmd.py", line 313, in run_command
      self.distribution.run_command(command)
    File "/anaconda3/lib/python3.7/distutils/dist.py", line 985, in run_command
      cmd_obj.run()
    File "/anaconda3/lib/python3.7/distutils/command/build.py", line 135, in run
      self.run_command(cmd_name)
    File "/anaconda3/lib/python3.7/distutils/cmd.py", line 313, in run_command
      self.distribution.run_command(command)
    File "/anaconda3/lib/python3.7/distutils/dist.py", line 985, in run_command
      cmd_obj.run()
    File "/private/var/folders/qf/jkn4v43j08xgst0r9yxyl0dc0000gn/T/pip-install-i4nyb384/jsonnet/setup.py", line 54, in run
      raise Exception('Could not build %s' % (', '.join(LIB_OBJECTS)))
  Exception: Could not build core/desugarer.o, core/formatter.o, core/libjsonnet.o, core/lexer.o, core/parser.o, core/pass.o, core/static_analysis.o, core/string_utils.o, core/vm.o, third_party/md5/md5.o

  ----------------------------------------
  Failed building wheel for jsonnet
  Running setup.py clean for jsonnet
Failed to build jsonnet
</code></pre>

<p>My compiler and gcc:</p>

<pre><code>(base) Sakets-MacBook-Pro:usr saketkhandelwal$ gcc -v
Configured with: --prefix=/Library/Developer/CommandLineTools/usr --with-gxx-include-dir=/usr/include/c++/4.2.1
Apple LLVM version 10.0.1 (clang-1001.0.46.4)
Target: x86_64-apple-darwin18.5.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
(base) Sakets-MacBook-Pro:usr saketkhandelwal$ clang --version
clang version 4.0.1 (tags/RELEASE_401/final)
Target: x86_64-apple-darwin18.5.0
Thread model: posix
InstalledDir: /anaconda3/bin
</code></pre>

<p>How do i fix this, i have tried reinstalling command line tools and package headers but still no luck. </p>

                     ''',
                  'so_title': "Cant install allennlp with pip on mac",
                  'so_tags': '<python-3.x><macos><clang++><command-line-tool><allennlp>',
                  'rel_num': 20}

allennlp_data3 = {'api': 'allennlp',
                  'so_body': '''<p>I want to make a cross validation in my project based on Pytorch.
And I didn't find any method that pytorch provided to delete the current model and empty the memory of GPU. Could you tell that how can I do it?</p>
                     ''',
                  'so_title': "pytorch delete model from gpu",
                  'so_tags': '<gpu><pytorch><allennlp>',
                  'rel_num': 5}

allennlp_data4 = {'api': 'allennlp',
                  'so_body': '''<p>I am trying to load an AllenNLP model weights. I could not find any documentation on how to save/load a whole model, so playing with weights only.</p>

<pre class="lang-py prettyprint-override"><code>from allennlp.nn import util
model_state = torch.load(filename_model, map_location=util.device_mapping(-1))
model.load_state_dict(model_state)
</code></pre>

<p>I modified my input corpus a bit and I am guessing because of this I am getting corpus-size mismatch:</p>

<pre class="lang-py prettyprint-override"><code>RuntimeError: Error(s) in loading state_dict for BasicTextFieldEmbedder:

    size mismatch for token_embedder_tokens.weight: 
    copying a param with shape torch.Size([2117, 16]) from checkpoint, 
    the shape in current model is torch.Size([2129, 16]).
</code></pre>

<p>Seemingly there is no official way to save model with corpus vocabulary. Any hacks around it?</p>

                     ''',
                  'so_title': "Saving/Loading models in AllenNLP package",
                  'so_tags': '<pytorch><allennlp>',
                  'rel_num': 24}

allennlp_data5 = {'api': 'allennlp',
                  'so_body': '''<p>I am trying my hand at ELMo by simply using it as part of a larger PyTorch model. A basic example is given <a href="https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#using-elmo-as-a-pytorch-module-to-train-a-new-model" rel="nofollow noreferrer">here</a>.</p>

<blockquote>
  <p>This is a torch.nn.Module subclass that computes any number of ELMo
  representations and introduces trainable scalar weights for each. For
  example, this code snippet computes two layers of representations (as
  in the SNLI and SQuAD models from our paper):</p>
</blockquote>

<pre><code>from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)

# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector
</code></pre>

<p>My question concerns the 'representations'. Can you compare them to normal word2vec output layers? You can choose how <em>many</em> ELMo will give back (increasing an n-th dimension), but what is the difference between these generated representations and what is their typical use? </p>

<p>To give you an idea, for the above code, <code>embeddings['elmo_representations']</code> returns a list of two items (the two representation layers) but they are identical.</p>

<p>In short, how can one define the 'representations' in ELMo? </p>

                     ''',
                  'so_title': "Understanding ELMo's number of presentations",
                  'so_tags': '<python><allennlp><elmo>',
                  'rel_num': 24}

allennlp_data6 = {'api': 'allennlp',
                  'so_body': '''<p>I have a homework which requires me to build an algorithm that can guess a missing word from a sentence. For example, when the input sentence is : " I took my **** for a walk this morning" , I want the output to guess the missing word(dog). My assignment requires me to train my own model from scratch. I built my corpus which has about 500.000 sentences. I cleaned the corpus. It is all lower-case and every sentence is seperated with a new line (\n) character. I also have vocabulary.txt file which lists all the unique words in descending order in frequency. The vocabulary file starts with the first 3 line 'S', '/S' and 'UNK' (these 3 tokens are surrounded with &lt;> in vocabulary.txt but using &lt;> in this website hides the characters between them for some reason) . I also have a small set of sentences with one missing word in every sentence which is denoted with [MASK], one sentence per line.</p>

<p>I followed the instructions in the <a href="https://github.com/allenai/bilm-tf" rel="nofollow noreferrer">https://github.com/allenai/bilm-tf</a> ,which provides steps to train your own model from scratch using Elmo.</p>

<p>After gathering the data.txt and vocabulary file, I used the</p>

<pre><code>python bin/train_elmo.py --train_prefix= &lt;path to training folder&gt; --vocab_file &lt;path to vocab file&gt; --save_dir &lt;path where models will be checkpointed&gt;`
</code></pre>

<p>and trained my corpus with tensorflow and CUDA enabled gpu.</p>

<p>After the training is finished, I used the following command:</p>

<pre><code>python bin/dump_weights.py --save_dir /output_path/to/checkpoint --outfile/output_path/to/weights.hdf5
</code></pre>

<p>Which gave me the weights.hdf5 and options.json files. The only warning I received while training my model is : </p>

<pre><code>WARNING : Error encountered when serializing lstm_output_embeddings.Type is unsupported, or the types of the items don't match field type in CollectionDef. 'list' object has no attribute 'name'
</code></pre>

<p>which was mentioned in the AllenAI repo as harmless. So it is safe to assume that the model training phase is finished correctly. My problem is, I have no idea what to do after this point. In this stackOverflow link <a href="https://stackoverflow.com/questions/54978443/predicting-missing-words-in-a-sentence-natural-language-processing-model">Predicting Missing Words in a sentence - Natural Language Processing Model</a>, the answer states that the following code can be used to predict the missing word:</p>

<pre><code>import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel,BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening,activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = '[CLS] I want to [MASK] the car because it is cheap . [SEP]'
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Create the segments tensors.
segments_ids = [0] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
with torch.no_grad():
predictions = model(tokens_tensor, segments_tensors)

predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(predicted_token)
</code></pre>

<p>Unfortunately, the code above is for Bert models. My assignment requires me to use Elmo models. I tried to find a library similiar to <strong>pytorch_pretrained_bert</strong> for Elmo but I couldn't find anything. What can I do to predict the masked words using my Elmo model?</p>

<p>Thanks.</p>

                     ''',
                  'so_title': "Using Elmo models to predict the masked word in a sentence",
                  'so_tags': '<python><machine-learning><nlp><allennlp><elmo>',
                  'rel_num': 24}

allennlp_data7 = {'api': 'allennlp',
                  'so_body': '''<p>Please first search our GitHub repository for similar questions.  If you don't find a similar example you can use the following template:</p>

<p><strong>System (please complete the following information):</strong>
 - OS: Ubunti 18.04
 - Python version: 3.6.7
 - AllenNLP version: v0.8.3
 - PyTorch version: 1.1.0</p>

<p><strong>Question</strong>
When I Try to predict string using SimpleSeq2SeqPredictor, It always show that</p>

<pre class="lang-sh prettyprint-override"><code>Traceback (most recent call last):
  File "predict.py", line 96, in &lt;module&gt;
    p = predictor.predict(i)
  File "venv/lib/python3.6/site-packages/allennlp/predictors/seq2seq.py", line 17, in predict
    return self.predict_json({"source" : source})
  File "/venv/lib/python3.6/site-packages/allennlp/predictors/predictor.py", line 56, in predict_json
    return self.predict_instance(instance)
  File "/venv/lib/python3.6/site-packages/allennlp/predictors/predictor.py", line 93, in predict_instance
    outputs = self._model.forward_on_instance(instance)
  File "/venv/lib/python3.6/site-packages/allennlp/models/model.py", line 124, in forward_on_instance
    return self.forward_on_instances([instance])[0]
  File "/venv/lib/python3.6/site-packages/allennlp/models/model.py", line 153, in forward_on_instances
    outputs = self.decode(self(**model_input))
  File "/venv/lib/python3.6/site-packages/allennlp/models/encoder_decoders/simple_seq2seq.py", line 247, in decode
    predicted_indices = output_dict["predictions"]
KeyError: 'predictions'
</code></pre>

<p>I try to do a translate system, but I am newbie, most of code come from
<a href="https://github.com/mhagiwara/realworldnlp/blob/master/examples/mt/mt.py" rel="nofollow noreferrer">https://github.com/mhagiwara/realworldnlp/blob/master/examples/mt/mt.py</a>
<a href="http://www.realworldnlpbook.com/blog/building-seq2seq-machine-translation-models-using-allennlp.html" rel="nofollow noreferrer">http://www.realworldnlpbook.com/blog/building-seq2seq-machine-translation-models-using-allennlp.html</a></p>

<p>this is my training code</p>

<pre class="lang-py prettyprint-override"><code>EN_EMBEDDING_DIM = 256
ZH_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
CUDA_DEVICE = 0
prefix = 'small'

reader = Seq2SeqDatasetReader(
    source_tokenizer=WordTokenizer(),
    target_tokenizer=CharacterTokenizer(),
    source_token_indexers={'tokens': SingleIdTokenIndexer()},
    target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')},
    lazy = True)
train_dataset = reader.read(f'./{prefix}-data/train.tsv')
validation_dataset = reader.read(f'./{prefix}-data/val.tsv')

vocab = Vocabulary.from_instances(train_dataset,
                                    min_count={'tokens': 3, 'target_tokens': 3})

en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EN_EMBEDDING_DIM)
# encoder = PytorchSeq2SeqWrapper(
#     torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

# attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
# attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
attention = DotProductAttention()

max_decoding_steps = 20   # TODO: make this variable
model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                        target_embedding_dim=ZH_EMBEDDING_DIM,
                        target_namespace='target_tokens',
                        attention=attention,
                        beam_size=8,
                        use_bleu=True)
optimizer = optim.Adam(model.parameters())
iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

iterator.index_with(vocab)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
trainer = Trainer(model=model,
                    optimizer=optimizer,
                    iterator=iterator,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    num_epochs=50,
                    serialization_dir=f'ck/{prefix}/',
                    cuda_device=cuda_device)

# for i in range(50):
    # print('Epoch: {}'.format(i))
trainer.train()

predictor = SimpleSeq2SeqPredictor(model, reader)

for instance in itertools.islice(validation_dataset, 10):
    print('SOURCE:', instance.fields['source_tokens'].tokens)
    print('GOLD:', instance.fields['target_tokens'].tokens)
    print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])

# Here's how to save the model.
with open(f"ck/{prefix}/manually_save_model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files(f"ck/{prefix}/vocabulary")
</code></pre>

<p>and this is my predict code</p>

<pre class="lang-py prettyprint-override"><code>EN_EMBEDDING_DIM = 256
ZH_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
CUDA_DEVICE = 0
prefix = 'big'

reader = Seq2SeqDatasetReader(
    source_tokenizer=WordTokenizer(),
    target_tokenizer=CharacterTokenizer(),
    source_token_indexers={'tokens': SingleIdTokenIndexer()},
    target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')},
    lazy = True)
# train_dataset = reader.read(f'./{prefix}-data/train.tsv')
# validation_dataset = reader.read(f'./{prefix}-data/val.tsv')

# vocab = Vocabulary.from_instances(train_dataset,
#                                     min_count={'tokens': 3, 'target_tokens': 3})
vocab = Vocabulary.from_files("ck/small/vocabulary")

en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EN_EMBEDDING_DIM)
# encoder = PytorchSeq2SeqWrapper(
#     torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

# attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
# attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
attention = DotProductAttention()

max_decoding_steps = 20   # TODO: make this variable
model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                        target_embedding_dim=ZH_EMBEDDING_DIM,
                        target_namespace='target_tokens',
                        attention=attention,
                        beam_size=8,
                        use_bleu=True)

# And here's how to reload the model.
with open("./ck/small/best.th", 'rb') as f:
    model.load_state_dict(torch.load(f))

predictor = Seq2SeqPredictor(model, dataset_reader=reader)
# print(predictor.predict("The dog ate the apple"))


test = [
    'Surely ,he has no power over those who believe and put their trust in their Lord ;',
    'And assuredly We have destroyed the generations before you when they did wrong ,while their apostles came unto them with the evidences ,and they were not such as to believe . In this wise We requite the sinning people .',
    'And warn your tribe ( O Muhammad SAW ) of near kindred .',
    'And to the Noble Messengers whom We have mentioned to you before ,and to the Noble Messengers We have not mentioned to you ; and Allah really did speak to Moosa .',
    'It is He who gave you hearing ,sight ,and hearts ,but only few of you give thanks .',
    'spreading in them much corruption ?',
    'That will envelop the people . This will be a painful punishment .',
    'When you received it with your tongues and spoke with your mouths what you had no knowledge of ,and you deemed it an easy matter while with Allah it was grievous .',
    'of which you are disregardful .',
    'Whoever disbelieves ,then the calamity of his disbelief is only on him ; and those who do good deeds ,are preparing for themselves .'
]




for i in test:
    p = predictor.predict(i) # &lt;------------------- ERROR !!!!!!!
    print(p) 
</code></pre>

<p>Am I do something wrong ?</p>

                     ''',
                  'so_title': "KeyError: 'predictions' When use SimpleSeq2SeqPredictor to predict string",
                  'so_tags': '<python><pytorch><allennlp>',
                  'rel_num': 24}

allennlp_data8 = {'api': 'allennlp',
                  'so_body': '''<p>I'm pretty new to AllenNLP and I'm currently using its pre-trained question answering model. I wonder if it has a passage length limit to ensure its performance? I know BERT will have a maximum length of 512 and will truncate longer passages.</p>
<p>I have tried longer passages on AllenNLP and it seems working but I just want to confirm. Thank you.</p>

                     ''',
                  'so_title': "Passage Length Limit for AllenNLP Question Answering",
                  'so_tags': '<maxlength><question-answering><allennlp>',
                  'rel_num': 24}

allennlp_data9 = {'api': 'allennlp',
                  'so_body': '''<p>How can I use JupyterLab with allennlp==0.3.0?</p>

<p>When I go to jupyterlab through my browser, the python kernel dies:</p>

<pre><code>notebook_1    |     from prompt_toolkit.shortcuts import create_prompt_application, create_eventloop, create_prompt_layout, create_output
notebook_1    | ImportError: cannot import name 'create_prompt_application'
notebook_1    | [I 18:47:49.552 LabApp] KernelRestarter: restarting kernel (3/5), new random ports
notebook_1    | Traceback (most recent call last):
notebook_1    |   File "/opt/conda/envs/pytorch-py3.6/lib/python3.6/runpy.py", line 193, in _run_module_as_main
notebook_1    |     "__main__", mod_spec)
notebook_1    |   File "/opt/conda/envs/pytorch-py3.6/lib/python3.6/runpy.py", line 85, in _run_code
notebook_1    |     exec(code, run_globals)
notebook_1    |   File "/opt/conda/envs/pytorch-py3.6/lib/python3.6/site-packages/ipykernel_launcher.py", line 15, in &lt;module&gt;
notebook_1    |     from ipykernel import kernelapp as app
notebook_1    |   File "/opt/conda/envs/pytorch-py3.6/lib/python3.6/site-packages/ipykernel/__init__.py", line 2, in &lt;module&gt;
notebook_1    |     from .connect import *
notebook_1    |   File "/opt/conda/envs/pytorch-py3.6/lib/python3.6/site-packages/ipykernel/connect.py", line 13, in &lt;module&gt;
notebook_1    |     from IPython.core.profiledir import ProfileDir
notebook_1    |   File "/opt/conda/envs/pytorch-py3.6/lib/python3.6/site-packages/IPython/__init__.py", line 55, in &lt;module&gt;
notebook_1    |     from .terminal.embed import embed
notebook_1    |   File "/opt/conda/envs/pytorch-py3.6/lib/python3.6/site-packages/IPython/terminal/embed.py", line 16, in &lt;module&gt;
notebook_1    |     from IPython.terminal.interactiveshell import TerminalInteractiveShell
notebook_1    |   File "/opt/conda/envs/pytorch-py3.6/lib/python3.6/site-packages/IPython/terminal/interactiveshell.py", line 22, in &lt;module&gt;
notebook_1    |     from prompt_toolkit.shortcuts import create_prompt_application, create_eventloop, create_prompt_layout, create_output
notebook_1    | ImportError: cannot import name 'create_prompt_application'
notebook_1    | [W 18:47:55.574 LabApp] KernelRestarter: restart failed
</code></pre>

<p>Installing an older version of create_prompt_application didn't help (it's causing other issues).</p>

                     ''',
                  'so_title': "How can I use JupyterLab with allennlp==0.3.0?",
                  'so_tags': '<python><jupyter-lab><allennlp>',
                  'rel_num': 24}

allennlp_data10 = {'api': 'allennlp',
                   'so_body': '''<p>How can I train the <a href="https://demo.allennlp.org/semantic-role-labeling/NjU2MjA3" rel="nofollow noreferrer">semantic role labeling model in AllenNLP</a>?</p>

<p>I am aware of the <a href="https://allenai.github.io/allennlp-docs/api/allennlp.training.trainer.html" rel="nofollow noreferrer"><code>allennlp.training.trainer</code></a> function but I don't know how to use it to train the semantic role labeling model.</p>

<p>Let's assume that the training samples are BIO tagged, e.g.:</p>

<pre><code>Remove B_O
the B_ARG1
fish I_ARG1
in B_LOC
the I_LOC 
background I_LOC 
</code></pre>
                     ''',
                   'so_title': "How can I train the semantic role labeling model in AllenNLP?",
                   'so_tags': '<python><nlp><allennlp>',
                   'rel_num': 24}


