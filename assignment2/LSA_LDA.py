import os
import json
import pickle as pkl
from collections import defaultdict, Counter

import numpy as np
import pytrec_eval
from tqdm import tqdm

import read_ap
import download_ap

import numpy as np
from tf_idf import TfIdfRetrieval
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from gensim.models import LsiModel

def bow_matrix(index, num_docs):
    vocabulary = list(index.df.keys())
    matrix = np.zeros((len(vocabulary),num_docs))
    for word in vocabulary:
        for (doc_id, tf) in index.ii[word]:
            matrix[word, doc_id] = np.log(1 + tf) / index.df[word]
    return matrix

def tf_idf_matrix(index, num_docs):
    vocabulary = list(index.df.keys())
    matrix = np.zeros((len(vocabulary),num_docs))
    for word in vocabulary:
        for (doc_id, tf) in index.ii[word]:
            matrix[word, doc_id] = 1.0
    return matrix

def cosine_similarity(a,b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos

def rank(model, query):
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

    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # Create instance for retrieval
    tfidf_index = TfIdfRetrieval(docs_by_id)

    bow_matrix = bow_matrix(tfidf_index, len(docs_by_id))
    bow_model = LsiModel(corpus=bow_matrix, num_topics=500)  # train model
    #vector = model[common_corpus[4]]  # apply model to BoW document
    #model.add_documents(common_corpus[4:])  # update model with new documents
    tmp_fname = get_tmpfile("bow_lsi.model")
    bow_model.save(tmp_fname)  # save model
    #loaded_model = LsiModel.load(tmp_fname)  # load model

    tfidf_matrix = tf_idf_matrix(tfidf_index, len(docs_by_id))
    tfidf_model = LsiModel(corpus=tfidf_matrix, num_topics=500)  # train model

    tmp_fname = get_tmpfile("tfidf_lsi.model")
    tfidf_model.save(tmp_fname)  # save model