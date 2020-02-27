from collections import defaultdict

import read_ap
import download_ap

import numpy as np
from scipy.sparse import csc_matrix
import math

from tf_idf import TfIdfRetrieval
from gensim.test.utils import get_tmpfile
from gensim.models import LsiModel, LdaModel
from gensim import similarities

from gensim.corpora import Dictionary
import logging


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

def bow_matrix(index, num_docs, binarize=False):
    vocabulary = list(index.df.keys())
    #matrix = np.zeros((len(vocabulary), num_docs))
    row = []
    col = []
    data = []
    for w, word in enumerate(vocabulary):
        for d, (doc_id, tf) in enumerate(index.ii[word]):
            row.append(w)
            col.append(d)
            #matrix[word, doc_id] = 1.0
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

def cosine_similarity(a,b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos

def bow2tfidf(bow_vec, d):
    result = []
    bow_vec = [(id, np.log(1 + tf)) for (id, tf) in bow_vec]
    for (id, tf) in bow_vec:
        result.append((id, tf / d.dfs[id]))
    return result

def rank(model, query, d, score='BoW'):
    query_repr = read_ap.process_text(query)
    vec_bow = d.doc2bow(query_repr)
    if score == 'tfidf':
        vec_bow = bow2tfidf(vec_bow, d)
    vec_lsi = model[vec_bow]  # convert the query to LSI space
    print(vec_lsi)
    index = similarities.Similarity(model[corpus])  # transform corpus to LSI space and index it
    index.save('/tmp/deerwester.index')
    index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #for i, s in enumerate(sims):
    #   print(s, documents[i])
    return sims

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

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # ensure dataset is downloaded
    download_ap.download_dataset()
    print("preparing data ...")
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # Set training parameters.
    num_topics = 50 #500
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    docs = [docs_by_id[id] for id in docs_by_id]
    d = Dictionary(docs)
    d.filter_extremes(no_below=50, no_above=0.5)
    corpus = [d.doc2bow(doc) for doc in docs]
    corpus_tfidf = [bow2tfidf(bow_vec, d) for bow_vec in corpus]

    lsi_bow = LsiModel(corpus, id2word=d, num_topics=num_topics)
    lsi_bow.save("./bow_lsi_01.model")  # save model
    print(lsi_bow.print_topics(5, 10))
    lsi_tfidf = LsiModel(corpus_tfidf, id2word=d, num_topics=num_topics)
    lsi_tfidf.save("./lsi_tfidf_01.model")  # save model
    print(lsi_tfidf.print_topics(5, 10))
    print("Start LDA")
    lda_model = LdaModel(
        corpus=corpus_tfidf,
        id2word=d,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    print("save LDA")
    lda_model.save("./lda.model")  # save model
    lda_model.print_topics(5, 10)
    """
    #dictionary = Dictionary(docs_by_id)
    #dictionary.filter_extremes(no_below=50, no_above=0.5)

    # Create instance for retrieval
    print("creating index ...")
    tfidf_index = TfIdfRetrieval(docs_by_id)
    filter_extremes(tfidf_index.ii, len(docs_by_id))
    vocab = list(tfidf_index.df.keys())
    id2word = dict(zip(range(len(vocab)), vocab))

    #tfidf_matrix = tf_idf_matrix(tfidf_index, len(docs_by_id))
    #t = tfidf_matrix.tolil(copy=True)
    #print("whaaat")
    #print(t.getcol(0))

    print("BoW-matrix [{},{}] being computed ...".format(len(list(tfidf_index.df.keys())), len(docs_by_id)))
    bow_matrix = bow_matrix(tfidf_index, len(docs_by_id))
    print("corpus size: {}".format(bow_matrix.shape))
    print("Start LSI-BoW")
    bow_model = LsiModel(corpus=bow_matrix, num_topics=num_topics, id2word=id2word)  # train model
    #vector = model[common_corpus[4]]  # apply model to BoW document
    #model.add_documents(common_corpus[4:])  # update model with new documents
    #tmp_fname = get_tmpfile("bow_lsi.model")
    print("save LSI-BoW")
    bow_model.save("./bow_lsi.model")  # save model
    #loaded_model = LsiModel.load(tmp_fname)  # load model
    print(bow_model.print_topics(5, 10))

    print("tf-idf-matrix being computed ...")
    tfidf_matrix = tf_idf_matrix(tfidf_index, len(docs_by_id))
    print("Start LSI-tf-idf")
    tfidf_model = LsiModel(corpus=tfidf_matrix, num_topics=num_topics, id2word=id2word)  # train model

    #tmp_fname = get_tmpfile("tfidf_lsi.model")
    print("save LSI-tf-idf")
    tfidf_model.save("./tfidf_lsi.model")  # save model
    print(tfidf_model.print_topics(5,10))

    print("Start LDA")
    lda_model = LdaModel(
        corpus=tfidf_matrix.tolil(),
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    #tmp_fname = get_tmpfile("lda.model")
    print("save LDA")
    lda_model.save("./lda.model")  # save model
    lda_model.print_topics(5, 10)
    """