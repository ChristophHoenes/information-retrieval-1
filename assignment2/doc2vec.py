import torch
from torch import nn
import pickle as pkl
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import read_ap
import random
import download_ap
import numpy as np
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Doc2Vec:
    def __init__(self, docs, wind_size, embedding_dim):
        corpus = self.read_docs(docs)
        self.docs = docs
        model = gensim.models.doc2vec.Doc2Vec(vector_size=embedding_dim, window=wind_size, min_count=50, workers=4, epochs=6)
        #model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
        print('building vocab...')
        model.build_vocab(corpus)
        print('training...')
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        print('done training')
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        self.model = model

    def read_docs(self, docs):
        corpus = []
        for i, doc in enumerate(docs):
            corpus.append(gensim.models.doc2vec.TaggedDocument(docs[doc], [doc]))
        print('done transforming')
        return corpus

    def get_doc_vec(self, tokens):
        vector = self.model.infer_vector(tokens)
        return vector

    def find_most_similar(self, text):
        orig = self.get_doc_vec(text)
        for doc in self.docs:
            doc_vec = self.get_doc_vec(self.docs[doc])


if __name__ == "__main__":
    # ensure dataset is downloaded
    wind_size = 10
    embedding_dim = 100
    download_ap.download_dataset()
    docs_by_id = read_ap.get_processed_docs()
    d2v = Doc2Vec(docs_by_id, wind_size, embedding_dim)