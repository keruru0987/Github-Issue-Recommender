# coding=utf-8
# @Author : Eric

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')               # 0.6153
# model = SentenceTransformer('all-mpnet-base-v2')            # 0.6053
# model = SentenceTransformer('bert-base-nli-mean-tokens')    # 0.8581
# model = SentenceTransformer('all-distilroberta-v1')         # 0.6067
# model = SentenceTransformer('all-MiniLM-L12-v2')            # 0.6399
# model = SentenceTransformer('multi-qa-distilbert-cos-v1')   # 0.6046
# model = SentenceTransformer('paraphrase-albert-small-v2')   # 0.5633
# model = SentenceTransformer('distiluse-base-multilingual-cased-v1')  # 0.6103

# Sentences are encoded by calling model.encode()
emb1 = model.encode("This is a red cat with a hat.")
emb2 = model.encode("Have you seen my red cat?")

cos_sim = util.cos_sim(emb1, emb2)
print("Cosine-Similarity:", cos_sim)

