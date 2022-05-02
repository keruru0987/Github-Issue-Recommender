# coding=utf-8
# @Author : Eric

from flask import Flask, render_template, request
from stackoverflow_search import SOSearcher

app = Flask(__name__)


@app.route('/mainpage', methods=["GET", "POST"])
def mainpage():
    if request.method == 'GET':
        return render_template("github issue recommender.html")
    else:
        selected_api = request.form.get("selected_api")
        so_query = request.form.get("so_query")
        print(selected_api, so_query)

        # 需要构造一个列表，其中的每一项内容需要包括:链接（由ID号进行构造）,标题，内容
        SO_Seacher = SOSearcher(selected_api, so_query)
        result = SO_Seacher.search()


        u = [['How to train AllenNLP SRL on non-English languages?',
              '''
              <p>I have been reading through the AllenNLP guide and documentation and was hoping to train an SRL Bert model on French.</p>
<p>On the SRL demo page you have the command for training a SRL Bert model as seen below:</p>
<pre><code>allennlp train \
        https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/structured_prediction/bert_base_srl.jsonnet \
        -s /path/to/output
</code></pre>
<p>Looking into that jsonnet file AllenNLP points out that they use the CONLL formatted Ontonotes 5.0 data. Since, as AllenNLP mentions, this data is not publicly available I went searching for what the format of this data looked like. Which lead me <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">here</a>.</p>
<p>Not fully understanding the format at that link I found this <a href="https://github.com/allenai/allennlp/blob/e5a74abccb969efe33001e20b8c749780d8f657e/allennlp/data/dataset_readers/dataset_utils/ontonotes.py#L83" rel="nofollow noreferrer">description</a> in AllenNLP's code for their Ontonotes class which was extremely helpful.</p>
<p>In light of all the details above I have a couple questions:</p>
<ol>
<li><p>When setting the environment variables <code>SRL_TRAIN_DATA_PATH</code> and <code>SRL_VALIDATION_DATA_PATH</code> that are used in the jsonnet file does the directory structure need to look exactly like the structure described in the Ontonotes class code (seen below) or what is the bare minimum if I will only have one file for training?</p>
<pre><code>└── train
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
</code></pre>
</li>
<li><p>My second question, using whatever directory structure is necessary, will I be able to train a French model if I create a file just like the <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">CONLL one</a> but all the words would be in French?</p>
</li>
<li><p>Third and finally if I can train a SRL Bert model using a CONLL file in the appropriate format are all of the columns in the CONLL file necessary to have data in. For example, Column 11 is the named entities, is it necessary to have named entities for training or can that column just be blank (i.e. nothing but hyphens). If it is the case that not all columns need data, which columns need to have data for training and which can be empty?</p>
</li>
</ol>
<p>I know it's a fair amount of questions so thank you in advance.</p>''',
              'https://stackoverflow.com/questions/69090025'],['How to train AllenNLP SRL on non-English languages?',
              '''
              <p>I have been reading through the AllenNLP guide and documentation and was hoping to train an SRL Bert model on French.</p>
<p>On the SRL demo page you have the command for training a SRL Bert model as seen below:</p>
<pre><code>allennlp train \
        https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/structured_prediction/bert_base_srl.jsonnet \
        -s /path/to/output
</code></pre>
<p>Looking into that jsonnet file AllenNLP points out that they use the CONLL formatted Ontonotes 5.0 data. Since, as AllenNLP mentions, this data is not publicly available I went searching for what the format of this data looked like. Which lead me <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">here</a>.</p>
<p>Not fully understanding the format at that link I found this <a href="https://github.com/allenai/allennlp/blob/e5a74abccb969efe33001e20b8c749780d8f657e/allennlp/data/dataset_readers/dataset_utils/ontonotes.py#L83" rel="nofollow noreferrer">description</a> in AllenNLP's code for their Ontonotes class which was extremely helpful.</p>
<p>In light of all the details above I have a couple questions:</p>
<ol>
<li><p>When setting the environment variables <code>SRL_TRAIN_DATA_PATH</code> and <code>SRL_VALIDATION_DATA_PATH</code> that are used in the jsonnet file does the directory structure need to look exactly like the structure described in the Ontonotes class code (seen below) or what is the bare minimum if I will only have one file for training?</p>
<pre><code>└── train
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
</code></pre>
''',
              'https://stackoverflow.com/questions/69090026'],['How to train AllenNLP SRL on non-English languages?',
              '''
              <p>I have been reading through the AllenNLP guide and documentation and was hoping to train an SRL Bert model on French.</p>
<p>On the SRL demo page you have the command for training a SRL Bert model as seen below:</p>
<pre><code>allennlp train \
        https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/structured_prediction/bert_base_srl.jsonnet \
        -s /path/to/output
</code></pre>
<p>Looking into that jsonnet file AllenNLP points out that they use the CONLL formatted Ontonotes 5.0 data. Since, as AllenNLP mentions, this data is not publicly available I went searching for what the format of this data looked like. Which lead me <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">here</a>.</p>
<p>Not fully understanding the format at that link I found this <a href="https://github.com/allenai/allennlp/blob/e5a74abccb969efe33001e20b8c749780d8f657e/allennlp/data/dataset_readers/dataset_utils/ontonotes.py#L83" rel="nofollow noreferrer">description</a> in AllenNLP's code for their Ontonotes class which was extremely helpful.</p>
<p>In light of all the details above I have a couple questions:</p>
<ol>
<li><p>When setting the environment variables <code>SRL_TRAIN_DATA_PATH</code> and <code>SRL_VALIDATION_DATA_PATH</code> that are used in the jsonnet file does the directory structure need to look exactly like the structure described in the Ontonotes class code (seen below) or what is the bare minimum if I will only have one file for training?</p>
<pre><code>└── train
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
</code></pre>
</li>
<li><p>My second question, using whatever directory structure is necessary, will I be able to train a French model if I create a file just like the <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">CONLL one</a> but all the words would be in French?</p>
</li>
<li><p>Third and finally if I can train a SRL Bert model using a CONLL file in the appropriate format are all of the columns in the CONLL file necessary to have data in. For example, Column 11 is the named entities, is it necessary to have named entities for training or can that column just be blank (i.e. nothing but hyphens). If it is the case that not all columns need data, which columns need to have data for training and which can be empty?</p>
</li>
</ol>
<p>I know it's a fair amount of questions so thank you in advance.</p>''',
              'https://stackoverflow.com/questions/69090025'],['How to train AllenNLP SRL on non-English languages?',
              '''
              <p>I have been reading through the AllenNLP guide and documentation and was hoping to train an SRL Bert model on French.</p>
<p>On the SRL demo page you have the command for training a SRL Bert model as seen below:</p>
<pre><code>allennlp train \
        https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/structured_prediction/bert_base_srl.jsonnet \
        -s /path/to/output
</code></pre>
<p>Looking into that jsonnet file AllenNLP points out that they use the CONLL formatted Ontonotes 5.0 data. Since, as AllenNLP mentions, this data is not publicly available I went searching for what the format of this data looked like. Which lead me <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">here</a>.</p>
<p>Not fully understanding the format at that link I found this <a href="https://github.com/allenai/allennlp/blob/e5a74abccb969efe33001e20b8c749780d8f657e/allennlp/data/dataset_readers/dataset_utils/ontonotes.py#L83" rel="nofollow noreferrer">description</a> in AllenNLP's code for their Ontonotes class which was extremely helpful.</p>
<p>In light of all the details above I have a couple questions:</p>
<ol>
<li><p>When setting the environment variables <code>SRL_TRAIN_DATA_PATH</code> and <code>SRL_VALIDATION_DATA_PATH</code> that are used in the jsonnet file does the directory structure need to look exactly like the structure described in the Ontonotes class code (seen below) or what is the bare minimum if I will only have one file for training?</p>
<pre><code>└── train
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
</code></pre>
</li>
<li><p>My second question, using whatever directory structure is necessary, will I be able to train a French model if I create a file just like the <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">CONLL one</a> but all the words would be in French?</p>
</li>
<li><p>Third and finally if I can train a SRL Bert model using a CONLL file in the appropriate format are all of the columns in the CONLL file necessary to have data in. For example, Column 11 is the named entities, is it necessary to have named entities for training or can that column just be blank (i.e. nothing but hyphens). If it is the case that not all columns need data, which columns need to have data for training and which can be empty?</p>
</li>
</ol>
<p>I know it's a fair amount of questions so thank you in advance.</p>''',
              'https://stackoverflow.com/questions/69090025'],['How to train AllenNLP SRL on non-English languages?',
              '''
              <p>I have been reading through the AllenNLP guide and documentation and was hoping to train an SRL Bert model on French.</p>
<p>On the SRL demo page you have the command for training a SRL Bert model as seen below:</p>
<pre><code>allennlp train \
        https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/structured_prediction/bert_base_srl.jsonnet \
        -s /path/to/output
</code></pre>
<p>Looking into that jsonnet file AllenNLP points out that they use the CONLL formatted Ontonotes 5.0 data. Since, as AllenNLP mentions, this data is not publicly available I went searching for what the format of this data looked like. Which lead me <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">here</a>.</p>
<p>Not fully understanding the format at that link I found this <a href="https://github.com/allenai/allennlp/blob/e5a74abccb969efe33001e20b8c749780d8f657e/allennlp/data/dataset_readers/dataset_utils/ontonotes.py#L83" rel="nofollow noreferrer">description</a> in AllenNLP's code for their Ontonotes class which was extremely helpful.</p>
<p>In light of all the details above I have a couple questions:</p>
<ol>
<li><p>When setting the environment variables <code>SRL_TRAIN_DATA_PATH</code> and <code>SRL_VALIDATION_DATA_PATH</code> that are used in the jsonnet file does the directory structure need to look exactly like the structure described in the Ontonotes class code (seen below) or what is the bare minimum if I will only have one file for training?</p>
<pre><code>└── train
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
</code></pre>
</li>
<li><p>My second question, using whatever directory structure is necessary, will I be able to train a French model if I create a file just like the <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">CONLL one</a> but all the words would be in French?</p>
</li>
<li><p>Third and finally if I can train a SRL Bert model using a CONLL file in the appropriate format are all of the columns in the CONLL file necessary to have data in. For example, Column 11 is the named entities, is it necessary to have named entities for training or can that column just be blank (i.e. nothing but hyphens). If it is the case that not all columns need data, which columns need to have data for training and which can be empty?</p>
</li>
</ol>
<p>I know it's a fair amount of questions so thank you in advance.</p>''',
              'https://stackoverflow.com/questions/69090025'],['How to train AllenNLP SRL on non-English languages?',
              '''
              <p>I have been reading through the AllenNLP guide and documentation and was hoping to train an SRL Bert model on French.</p>
<p>On the SRL demo page you have the command for training a SRL Bert model as seen below:</p>
<pre><code>allennlp train \
        https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/structured_prediction/bert_base_srl.jsonnet \
        -s /path/to/output
</code></pre>
<p>Looking into that jsonnet file AllenNLP points out that they use the CONLL formatted Ontonotes 5.0 data. Since, as AllenNLP mentions, this data is not publicly available I went searching for what the format of this data looked like. Which lead me <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">here</a>.</p>
<p>Not fully understanding the format at that link I found this <a href="https://github.com/allenai/allennlp/blob/e5a74abccb969efe33001e20b8c749780d8f657e/allennlp/data/dataset_readers/dataset_utils/ontonotes.py#L83" rel="nofollow noreferrer">description</a> in AllenNLP's code for their Ontonotes class which was extremely helpful.</p>
<p>In light of all the details above I have a couple questions:</p>
<ol>
<li><p>When setting the environment variables <code>SRL_TRAIN_DATA_PATH</code> and <code>SRL_VALIDATION_DATA_PATH</code> that are used in the jsonnet file does the directory structure need to look exactly like the structure described in the Ontonotes class code (seen below) or what is the bare minimum if I will only have one file for training?</p>
<pre><code>└── train
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
</code></pre>
</li>
<li><p>My second question, using whatever directory structure is necessary, will I be able to train a French model if I create a file just like the <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">CONLL one</a> but all the words would be in French?</p>
</li>
<li><p>Third and finally if I can train a SRL Bert model using a CONLL file in the appropriate format are all of the columns in the CONLL file necessary to have data in. For example, Column 11 is the named entities, is it necessary to have named entities for training or can that column just be blank (i.e. nothing but hyphens). If it is the case that not all columns need data, which columns need to have data for training and which can be empty?</p>
</li>
</ol>
<p>I know it's a fair amount of questions so thank you in advance.</p>''',
              'https://stackoverflow.com/questions/69090025'],['How to train AllenNLP SRL on non-English languages?',
              '''
              <p>I have been reading through the AllenNLP guide and documentation and was hoping to train an SRL Bert model on French.</p>
<p>On the SRL demo page you have the command for training a SRL Bert model as seen below:</p>
<pre><code>allennlp train \
        https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/structured_prediction/bert_base_srl.jsonnet \
        -s /path/to/output
</code></pre>
<p>Looking into that jsonnet file AllenNLP points out that they use the CONLL formatted Ontonotes 5.0 data. Since, as AllenNLP mentions, this data is not publicly available I went searching for what the format of this data looked like. Which lead me <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">here</a>.</p>
<p>Not fully understanding the format at that link I found this <a href="https://github.com/allenai/allennlp/blob/e5a74abccb969efe33001e20b8c749780d8f657e/allennlp/data/dataset_readers/dataset_utils/ontonotes.py#L83" rel="nofollow noreferrer">description</a> in AllenNLP's code for their Ontonotes class which was extremely helpful.</p>
<p>In light of all the details above I have a couple questions:</p>
<ol>
<li><p>When setting the environment variables <code>SRL_TRAIN_DATA_PATH</code> and <code>SRL_VALIDATION_DATA_PATH</code> that are used in the jsonnet file does the directory structure need to look exactly like the structure described in the Ontonotes class code (seen below) or what is the bare minimum if I will only have one file for training?</p>
<pre><code>└── train
       └── data
           └── english
               └── annotations
                   ├── bc
                   ├── bn
                   ├── mz
                   ├── nw
                   ├── pt
                   ├── tc
                   └── wb
</code></pre>
</li>
<li><p>My second question, using whatever directory structure is necessary, will I be able to train a French model if I create a file just like the <a href="https://github.com/ontonotes/conll-formatted-ontonotes-5.0/blob/master/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/cctv/00/cctv_0001.gold_skel" rel="nofollow noreferrer">CONLL one</a> but all the words would be in French?</p>
</li>
<li><p>Third and finally if I can train a SRL Bert model using a CONLL file in the appropriate format are all of the columns in the CONLL file necessary to have data in. For example, Column 11 is the named entities, is it necessary to have named entities for training or can that column just be blank (i.e. nothing but hyphens). If it is the case that not all columns need data, which columns need to have data for training and which can be empty?</p>
</li>
</ol>
<p>I know it's a fair amount of questions so thank you in advance.</p>''',
              'https://stackoverflow.com/questions/69090025']]

        return render_template("so_results.html", u=result, api=selected_api, query=so_query)


if __name__ == '__main__':
    app.run()
