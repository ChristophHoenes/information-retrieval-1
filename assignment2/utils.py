import numpy as np
from scipy.sparse import csc_matrix

def bow_matrix(index, num_docs, binarize=False):
    vocabulary = list(index.df.keys())
    row = []
    col = []
    data = []
    for w, word in enumerate(vocabulary):
        for d, (doc_id, tf) in enumerate(index.ii[word]):
            row.append(w)
            col.append(d)
            data.append(tf)
    if binarize:
        data = np.ones_like(row)
    else:
        data = np.array(data)
    return csc_matrix((data, (np.array(row), np.array(col))), shape=(len(vocabulary), num_docs))


def tf_idf_matrix(index, num_docs):
    vocabulary = list(index.df.keys())
    row = []
    col = []
    data = []
    for w, word in enumerate(vocabulary):
        for d, (doc_id, tf) in enumerate(index.ii[word]):
            tf_idf = np.log(1 + tf) / index.df[word]
            if tf_idf != 0:
                row.append(w)
                col.append(d)
                data.append(tf_idf)
    return csc_matrix((np.array(data), (np.array(row), np.array(col))), shape=(len(vocabulary), num_docs))


def filter_extremes(index, num_docs, no_below=50, no_above=0.65):
    deleted_below = 0
    deleted_above = 0
    print("Initial vocabulary size: {}".format(len(index)))
    for k in list(index):
        if len(index[k]) < no_below:
            del index[k]
            deleted_below += 1
        if len(index[k]) > no_above*num_docs:
            del index[k]
            deleted_above += 1
    print("Deleted {} tokens below threshold and {} above threshold. New vocabulary size: {}".format(deleted_below,
                                                                                                     deleted_above,
                                                                                                     len(index)))


def bow2tfidf(bow_vec, d):
    result = []
    bow_vec = [(id, np.log(1 + tf)) for (id, tf) in bow_vec]
    for (id, tf) in bow_vec:
        result.append((id, tf / d.dfs[id]))
    return result


def list_of_tuples2np_array(li, num_topics):
    result = np.zeros(num_topics)
    for (idx, value) in li:
        result[idx] = value
    return result+1e-8


def kl_divergence(p, q, num_topics):
    p = list_of_tuples2np_array(p, num_topics)
    q = list_of_tuples2np_array(q, num_topics)
    sum_pq = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    sum_qp = np.sum(np.where(q != 0, q * np.log(q / p), 0))
    return float((sum_pq + sum_qp) / 2)


def cosine_similarity(a,b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos


def rank2(model, query):
    query_repr = read_ap.process_text(query)
    q_k = np.linalg.inv(model.projection.s) @ model.projection.u.T @ query_repr
    #scores = np.zeros(len(model.projection.s[:,0]))
    scores = defaultdict(float)
    for doc_id in range(len(scores)):
        scores[doc_id] = cosine_similarity(model.projection.s[:, doc_id], q_k)
    scores = list(scores.items())
    scores.sort(key=lambda _: _[1])
    return scores