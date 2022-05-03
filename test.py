# coding=utf-8
# @Author : Eric


from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'D:\stanford-corenlp-4.4.0')
sentence = "I traveled to New York last year"
print(nlp.dependency_parse(sentence))
nlp.close()
