import read_ap
import download_ap
from utils import bow2tfidf
from evaluate import evaluate_model

import numpy as np
import os
import pickle as pkl
from pprint import pprint

from gensim.models import LsiModel
from gensim import similarities

from gensim.corpora import Dictionary
import logging

class LSI():
    def __init__(self, docs, num_topics=500, chunksize=20000, no_below=50, no_above=0.5,
                 tfidf=True, model_path="./lsi_data"):
        # Set training parameters.
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.no_below = no_below
        self.no_above = no_above
        self.tfidf = tfidf
        self.model_path = model_path

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        index_path = './data.index'
        if os.path.exists(index_path):
            assert os.path.exists("./corpus_bow") and os.path.exists(os.path.join("./corpus_tfidf")),\
                "Corpus file missing! Please rebuild index."
            with open(index_path, "rb") as reader:
                index = pkl.load(reader)
                self.index = index["index"]
                self.index2docid = index["index2docid"]
            with open("./corpus_bow", "rb") as reader:
                self.corpus_bow = pkl.load(reader)
            with open("./corpus_tfidf", "rb") as reader:
                self.corpus_tfidf = pkl.load(reader)
            if os.path.exists(os.path.join(self.model_path, "lsi.model")):
                self.model = LsiModel.load(os.path.join(self.model_path, "lsi.model"))
            else:
                self.model = self.train()
        else:
            self.rebuild_index(docs, index_path)

    def train(self):
        print("Start LSI training: ")
        temp = self.index[0]
        id2word = self.index.id2token
        lsi_model = LsiModel(
            corpus=self.corpus_tfidf if self.tfidf else self.corpus_bow,
            id2word=id2word,
            chunksize=self.chunksize,
            num_topics=self.num_topics
        )
        self.model = lsi_model
        print("done.")
        return lsi_model

    def save(self, path="./lsi.model"):
        print("saving LSI model...")
        self.model.save(os.path.join(self.model_path,path))
        print("done.")

    def rebuild_index(self, docs, index_path, retrain=True):
        self.index2docid = {i: docid for i, docid in enumerate(docs)}
        docs2 = [docs[docid] for docid in docs]
        self.index = Dictionary(docs2)
        self.index.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        self.corpus_bow = [self.index.doc2bow(doc) for doc in docs2]
        self.corpus_tfidf = [bow2tfidf(bow_vec, self.index) for bow_vec in self.corpus_bow]
        with open(index_path, "wb") as writer:
            index = {
                "index": self.index,
                "index2docid": self.index2docid
            }
            pkl.dump(index, writer)
        with open("./corpus_bow", "wb") as writer:
            pkl.dump(self.corpus_bow, writer)
        with open("./corpus_tfidf", "wb") as writer:
            pkl.dump(self.corpus_tfidf, writer)
        if retrain:
            _ = self.train()

    def rank(self, query, first_query=True):
        query_repr = read_ap.process_text(query)
        vec_bow = self.index.doc2bow(query_repr)
        if self.tfidf:
            vec_bow = bow2tfidf(vec_bow, self.index)
        vec_lsi = self.model[vec_bow]  # convert the query to LSI space

        index_path = os.path.join(self.model_path, 'lsi_index_rank.index')
        if first_query:  # and not os.path.exists(os.path.join(self.model_path, 'lsi_index_rank.index')):
            used_corpus = self.corpus_tfidf if self.tfidf else self.corpus_bow
            index = similarities.Similarity(os.path.join(self.model_path,"shard"), self.model[used_corpus], self.num_topics)  #len(self.index))  # transform corpus to LSI space and index it
            index.save(index_path)
        else:
            index = similarities.Similarity.load(index_path)
        sims = index[vec_lsi]  # query similarity
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        sims = [(self.index2docid[idx], np.float64(value)) for (idx, value) in sims]
        return sims

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # ensure dataset is downloaded
    download_ap.download_dataset()
    print("preparing data ...")
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # read in the qrels
    print("read in queries...")
    qrels, queries = read_ap.read_qrels()
    print("done")

    lsi = LSI(docs_by_id, num_topics=10, tfidf=True, model_path="./lsi_data_")
    topic_params = [1000, 2000, 500, 100, 50, 10]
    for t in topic_params:
        for tfidf in [False, True]:
            lsi.num_topics = t
            lsi.tfidf = tfidf
            tfidf_tag = "tfidf" if tfidf else "bow"
            run_token = "Lsi" + tfidf_tag + str(t)
            print("train "+run_token)
            lsi.train()
            with open(os.path.join(lsi.model_path,"top_topics_"+run_token+".txt"), 'w') as f:
                pprint(lsi.model.print_topics(num_topics=5), stream=f)
            eval_path = os.path.join(lsi.model_path, "lsi_" + tfidf_tag + str(t))
            evaluate_model(lsi, qrels, queries, eval_path+".json",
                           eval_path+".trec", "Lsi"+tfidf_tag+str(t))

