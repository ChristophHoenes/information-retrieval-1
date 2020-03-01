from doc2vec import Doc2Vec
import read_ap
from tqdm import tqdm
import pytrec_eval
import json

# get queries
qrels, queries = read_ap.read_qrels()

vec_dim_def = 300
wind_size_def = 15
vocab_size_def = 50

docs_by_id = read_ap.get_processed_docs()

#test = {}
#count = 0
#for doc in docs_by_id:
#    if count > 50:
#        break
#    test[doc] = docs_by_id[doc]
#    count+=1
#docs_by_id = test
wind_sizes = [5]#10, 15, 20]
vector_dims = [400] #[300, 400, 500]
# for each vocabulary size, the following min_count values approximate them {10: 250, 25: 50, 50: 15, 100: 5, 200: 2}
vocab_sizes = [2]#{250, 50, 15, 5, 2}
"""
for embedding_dim in vector_dims:
    overall_ser = {}
    d2v = Doc2Vec(docs_by_id, wind_size_def, embedding_dim, vocab_size_def)
    d2v.get_doc_vecs(docs_by_id)
    for qid in tqdm(qrels):
        query_text = queries[qid]
        results = d2v.search(query_text)
        overall_ser[qid] = dict(results)
    with open("d2v_vecdim_"+str(embedding_dim)+".json", "w") as writer:
        json.dump(overall_ser, writer, indent=1)

for wind_size in wind_sizes:
    overall_ser = {}
    d2v = Doc2Vec(docs_by_id, wind_size, vec_dim_def, vocab_size_def)
    d2v.get_doc_vecs(docs_by_id)
    for qid in tqdm(qrels):
        query_text = queries[qid]
        results = d2v.search(query_text)
        overall_ser[qid] = dict(results)
    with open("d2v_windsize_"+str(wind_size)+".json", "w") as writer:
        json.dump(overall_ser, writer, indent=1)

for vocab_size in vocab_sizes:
    overall_ser = {}
    d2v = Doc2Vec(docs_by_id, wind_size_def, vec_dim_def, vocab_size)
    d2v.get_doc_vecs(docs_by_id)
    for qid in tqdm(qrels):
        query_text = queries[qid]
        results = d2v.search(query_text)
        overall_ser[qid] = dict(results)
    with open("d2v_vocabsize_"+str(vocab_size)+".json", "w") as writer:
        json.dump(overall_ser, writer, indent=1)
"""
wind_size = 5
vec_dim = 400
vocab = 2
overall_ser = {}
d2v = Doc2Vec(docs_by_id, wind_size, vec_dim, vocab)
d2v.get_doc_vecs(docs_by_id)
for qid in tqdm(qrels):
    query_text = queries[qid]
    results = d2v.search(query_text)
    overall_ser[qid] = dict(results)
with open("d2v_bestvalid_"+str(wind_size)+".json", "w") as writer:
    json.dump(overall_ser, writer, indent=1)