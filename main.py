import data_process
from bm25 import BM25
from vsm import VSM
from word2vec import Word2Vec
from sentence2vec import Sentence2Vec
from analyze import Analyzer


def baseline_analyze():
    # 对照组，随机选择
    docs = data_process.get_data()
    baseline_analyzer = Analyzer(docs)
    baseline_analyzer.baseline_analyze_lda()


def model_analyze(model_name):
    docs = data_process.get_data()
    query = data_process.get_query()
    if model_name == 'bm25':
        select_model = BM25(docs)
    elif model_name == 'vsm':
        select_model = VSM(docs)
    elif model_name == 'word2vec':
        select_model = Word2Vec(docs)
    elif model_name == 'sentence2vec':
        select_model = Sentence2Vec(docs)
    else:
        raise Exception('no such model found')
    scores = select_model.score_all(query)

    # analyze方法需要改进
    analyzer = Analyzer(docs, scores)
    analyzer.analyze_lda()


if __name__ == '__main__':
    # baseline_analyze()
    model_analyze('word2vec')



