import read_ap
import download_ap
from utils import bow2tfidf, kl_divergence

import numpy as np
import os
import json
import pickle as pkl

from gensim.models import LsiModel, LdaModel
from gensim import similarities

from gensim.corpora import Dictionary
import logging

class LSI():
    def __init__(self, docs, num_topics=500, chunksize=2000, no_below=50, no_above=0.5,
                 tfidf=True, model_path="./lsi_data"):
        # Set training parameters.
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.iterations = iterations
        self.eval_every = eval_every
        self.no_below = no_below
        self.no_above = no_above
        self.tfidf = tfidf
        self.model_path = model_path

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        index_path = os.path.join(model_path, 'lsi_index_train.index')
        if os.path.exists(index_path):
            with open(index_path, "rb") as reader:
                index = pkl.load(reader)
                self.index = index["index"]
                self.index2docid = index["index2docid"]
        else:
            self.rebuild_index(docs, index_path)

    def train(self):
        print("Start LSI training: ")
        temp = self.index[0]
        id2word = self.index.id2token
        lsi_model = LsiModel(
            corpus=self.corpus_tfidf if self.tfidf else self.corpus_bow,
            id2word=id2word,
            chunksize=chunksize,
            num_topics=num_topics
        )
        print("done.")
        return lsi_model

    def save(self, path="./lsi.model"):
        print("saving LSI model...")
        self.model.save(path)
        print("done.")

    def rebuild_index(self, docs, index_path, retrain=True):
        self.index2docid = {i: id for i, docid in enumerate(docs)}
        docs2 = [docs[id] for id in docs]
        self.index = Dictionary(docs2)
        self.index.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        with open(index_path, "wb") as writer:
            index = {
                "index": self.index,
                "index2docid": self.index2docid
            }
            pkl.dump(self.index, writer)
        self.corpus_bow = [d.doc2bow(doc) for doc in docs2]
        self.corpus_tfidf = [bow2tfidf(bow_vec, d) for bow_vec in self.corpus_bow]
        if retrain:
            self.model = self.train()

    def rank(self, query, first_query=True):
        query_repr = read_ap.process_text(query)
        vec_bow = self.index.doc2bow(query_repr)
        if self.tfidf:
            vec_bow = bow2tfidf(vec_bow, self.index)
        vec_lsi = self.model[vec_bow]  # convert the query to LSI space

        index_path = os.path.join(self.model_path, 'lsi_index_rank.index')
        if first_query:
            index = similarities.Similarity(self.model[self.corpus_tfidf if self.tfidf else self.corpus_bow])  # transform corpus to LSI space and index it
            index.save(index_path)
        else:
            index = similarities.Similarity.load(index_path)
        sims = index[vec_lsi]  # query similarity
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims

class LDA():
    def __init__(self, docs, num_topics=500, chunksize=2000, passes=1, iterations=1000, eval_every=None, no_below=50,
                 no_above=0.5, tfidf=True, model_path="./lda_data"):
        # Set training parameters.
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.iterations = iterations
        self.eval_every = eval_every
        self.no_below = no_below
        self.no_above = no_above
        self.tfidf = tfidf

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        index_path = os.path.join(model_path, 'lda_index_train.index')
        if os.path.exists(index_path):
            with open(index_path, "rb") as reader:
                index = pkl.load(reader)
                self.index = index["index"]
                self.index2docid = index["index2docid"]
        else:
            self.rebuild_index(docs, index_path)

    def train(self):
        print("Start LDA training: ")
        temp = self.index[0]
        id2word = self.index.id2token
        lda_model = LdaModel(
            corpus=self.corpus_tfidf if self.tfidf else self.corpus_bow,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )
        print("done.")
        return  lda_model

    def save(self, path="./lda.model" ):
        print("saving LDA model...")
        self.model.save(path)
        print("done.")

    def rebuild_index(self, docs, index_path, retrain=True):
        self.index2docid = {i: id for i, docid in enumerate(docs)}
        docs2 = [docs[id] for id in docs]
        self.index = Dictionary(docs2)
        self.index.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        with open(index_path, "wb") as writer:
            index = {
                "index": self.index,
                "index2docid": self.index2docid
            }
            pkl.dump(self.index, writer)
        self.corpus_bow = [d.doc2bow(doc) for doc in docs2]
        self.corpus_tfidf = [bow2tfidf(bow_vec, d) for bow_vec in self.corpus_bow]
        if retrain:
            self.model = self.train()


def rank(model, query, d, tfidf_matrix=False, lda=False):
    query_repr = read_ap.process_text(query)
    vec_bow = d.doc2bow(query_repr)
    if tfidf_matrix:
        vec_bow = bow2tfidf(vec_bow, d)
    vec_lsi = model[vec_bow]  # convert the query to LSI space
    print(vec_lsi)
    if lda:
        sims = [(index2docid[i], kl_divergence(doc, vec_lsi)) for i, doc in enumerate(index)]
    index = similarities.Similarity(model[corpus])  # transform corpus to LSI space and index it
    index.save('/tmp/deerwester.index')
    index = similarities.Similarity.load('/tmp/deerwester.index')
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #for i, s in enumerate(sims):
    #   print(s, documents[i])
    return sims


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
    passes = 2
    iterations = 400
    eval_every = 1  # Don't evaluate model perplexity, takes too much time.

    index2docid = {i:id for i, id in enumerate(docs_by_id)}
    docs = [docs_by_id[id] for id in docs_by_id]
    d = Dictionary(docs)
    d.filter_extremes(no_below=50, no_above=0.5)
    corpus = [d.doc2bow(doc) for doc in docs]
    corpus_tfidf = [bow2tfidf(bow_vec, d) for bow_vec in corpus]
    """
    lsi_bow = LsiModel(corpus, id2word=d, num_topics=num_topics)
    lsi_bow.save("./bow_lsi_01.model")  # save model
    print(lsi_bow.print_topics(5, 10))
    lsi_tfidf = LsiModel(corpus_tfidf, id2word=d, num_topics=num_topics)
    lsi_tfidf.save("./lsi_tfidf_01.model")  # save model
    print(lsi_tfidf.print_topics(5, 10))
    """
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
    query_repr = read_ap.process_text('costly response technology')
    vec_bow = d.doc2bow(query_repr)
    vec_bow = bow2tfidf(vec_bow, d)
    vec_lsi = lda_model[vec_bow]  # convert the query to LSI space
    print(vec_lsi)
    corp_lda = vec_lsi = lda_model[corpus_tfidf]
    print(len(corp_lda))
    print(len(corp_lda[0]))
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