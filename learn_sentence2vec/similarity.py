# coding=utf-8
# @Author : Eric

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')            # 0.6053
# model = SentenceTransformer('bert-base-nli-mean-tokens')    # 0.8581
# model = SentenceTransformer('all-distilroberta-v1')         # 0.6067
# model = SentenceTransformer('all-MiniLM-L12-v2')            # 0.6399
# model = SentenceTransformer('multi-qa-distilbert-cos-v1')   # 0.6046
# model = SentenceTransformer('paraphrase-albert-small-v2')   # 0.5633
# model = SentenceTransformer('distiluse-base-multilingual-cased-v1')  # 0.6103

# Sentences are encoded by calling model.encode()
query = model.encode("I am a student of HIT")
doc1 = model.encode("I hope I can study in HIT")
doc2 = model.encode("HIT is located in Harbin")
doc3 = model.encode("ZJU is located in HangZhou")
count = 1
for sentence in [doc1, doc2, doc3]:
    cos_sim = util.cos_sim(query, sentence)
    print("query与doc" + str(count) + "相似度：", cos_sim)
    count += 1


