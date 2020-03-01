import read_ap
import download_ap
from utils import bow2tfidf, kl_divergence
from evaluate import evaluate_model
import pytrec_eval
from tqdm import tqdm

import numpy as np
import os
import json
import pickle as pkl
from pprint import pprint

from gensim.models import LsiModel, LdaModel
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
        self.model_path = model_path

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        index_path = './data.index'
        if os.path.exists(index_path):
            assert os.path.exists("./corpus_bow") and os.path.exists(os.path.join("./corpus_tfidf")), \
                "Corpus file missing! Please rebuild index."
            with open(index_path, "rb") as reader:
                index = pkl.load(reader)
                self.index = index["index"]
                self.index2docid = index["index2docid"]
            with open("./corpus_bow", "rb") as reader:
                self.corpus_bow = pkl.load(reader)
            with open("./corpus_tfidf", "rb") as reader:
                self.corpus_tfidf = pkl.load(reader)
            if os.path.exists(os.path.join(self.model_path, "lda.model")):
                self.model = LsiModel.load(os.path.join(self.model_path, "lda.model"))
            else:
                self.model = self.train()
        else:
            self.rebuild_index(docs, index_path)

    def train(self):
        print("Start LDA training: ")
        temp = self.index[0]
        id2word = self.index.id2token
        lda_model = LdaModel(
            corpus=self.corpus_tfidf if self.tfidf else self.corpus_bow,
            id2word=id2word,
            chunksize=self.chunksize,
            alpha='auto',
            eta='auto',
            iterations=self.iterations,
            num_topics=self.num_topics,
            passes=self.passes,
            eval_every=self.eval_every
        )
        self.model = lda_model
        print("done.")
        return lda_model

    def save(self, path="./lda.model" ):
        print("saving LDA model...")
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
        vec_lda = self.model[vec_bow]  # convert the query to LSI space
        index_path = os.path.join(self.model_path, 'lda_index_rank.index')
        if first_query:
            with open(index_path, "wb") as writer:
                self.rank_index = self.model[self.corpus_tfidf if self.tfidf else self.corpus_bow]
                pkl.dump(self.rank_index, writer)
        #else:
        #    with open(index_path, "rb") as reader:
        #        index = pkl.load(reader)
        #sims = [(self.index2docid[i], kl_divergence(self.model[doc], vec_lda)) for i, doc in enumerate(index)]
        sims = [kl_divergence(doc, vec_lda, self.num_topics) for doc in self.rank_index]
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
    topic_params = [10]
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

    #lsi_bow = LSI(docs_by_id, num_topics=10, tfidf=False, model_path="./lsi_data_bow10")
    #lsi_bow.save()
    #lsi_tfidf = LSI(docs_by_id, num_topics=10, tfidf=True, model_path="./lsi_data_tfidf10")
    #lsi_tfidf.save()

    #lda_tfidf = LDA(docs_by_id, num_topics=10, tfidf=False, model_path="./lda_data_bow10")
    #lda_tfidf.save()

    # read in the qrels
    #print("read in queries...")
    #qrels, queries = read_ap.read_qrels()
    #print("done")
    #evaluate_model(lda_bow, qrels, queries, "./lda_data_bow500/lda_bow500.json", "./lda_data_bow10/lda_bow500.trec", 'LdaBow500')

    """
    overall_ser_lsi_bow = {}
    overall_ser_lsi_tfidf = {}
    overall_ser_lda_tfidf = {}

    print("Running Benchmarks...")
    first_query = True
    # collect results
    for qid in tqdm(qrels):
        query_text = queries[qid]

        results_lsi_bow = lsi_bow.rank(query_text, first_query=first_query)
        overall_ser_lsi_bow[qid] = dict(results_lsi_bow)

        results_lsi_tfidf = lsi_tfidf.rank(query_text, first_query=first_query)
        overall_ser_lsi_tfidf[qid] = dict(results_lsi_tfidf)

        results_lda_tfidf = lda_tfidf.rank(query_text, first_query=first_query)
        overall_ser_lda_tfidf[qid] = dict(results_lda_tfidf)
        first_query = False
    # run evaluation with `qrels` as the ground truth relevance judgements
    # here, we are measuring MAP and NDCG, but this can be changed to
    # whatever you prefer
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics_lsi_bow = evaluator.evaluate(overall_ser_lsi_bow)
    metrics_lsi_tfidf = evaluator.evaluate(overall_ser_lsi_tfidf)
    print('get metrics LDA...')
    metrics_lda_tfidf = evaluator.evaluate(overall_ser_lda_tfidf)
    print('done.')


    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open("lsi_bow.json", "w") as writer:
        json.dump(metrics_lsi_bow, writer, indent=1)
    with open("lsi_tfidf.json", "w") as writer:
        json.dump(metrics_lsi_tfidf, writer, indent=1)
    with open("lda_tfidf.json", "w") as writer:
        json.dump(metrics_lda_tfidf, writer, indent=1)

    overall_rankings = [('overall_ser_lsi_bow', overall_ser_lsi_bow),
                        ('overall_ser_lsi_tfidf', overall_ser_lsi_tfidf)]#,
                        #('overall_ser_lda_tfidf', overall_ser_lda_tfidf)]
    # write file with all query-doc pairs, scores, ranks, etc.
    for o_name, o in overall_rankings:
        f = open(o_name + ".dat", "w")
        for qid in o:
            prevscore = 1e9
            for rank, docid in enumerate(o[qid], 1):
                score = o[qid][docid]
                if score > prevscore:
                    f.close()
                    raise Exception("'results_dic' not ordered! Stopped writing results")
                f.write(f"{qid} Q0 {docid} {rank} {score} STANDARD\n")
                prevscore = score
        f.close()
    """
    print('programm finished without error')
    """
    # Set training parameters.
    num_topics = 50 #500
    chunksize = 2000
    passes = 2
    iterations = 400
    eval_every = 1  # Don't evaluate model perplexity, takes too much time.

    index2docid = {i:docid for i, docid in enumerate(docs_by_id)}
    docs = [docs_by_id[docid] for docid in docs_by_id]
    d = Dictionary(docs)
    d.filter_extremes(no_below=50, no_above=0.5)
    corpus = [d.doc2bow(doc) for doc in docs]
    corpus_tfidf = [bow2tfidf(bow_vec, d) for bow_vec in corpus]
    
    #lsi_bow = LsiModel(corpus, id2word=d, num_topics=num_topics)
    #lsi_bow.save("./bow_lsi_01.model")  # save model
    #print(lsi_bow.print_topics(5, 10))
    #lsi_tfidf = LsiModel(corpus_tfidf, id2word=d, num_topics=num_topics)
    #lsi_tfidf.save("./lsi_tfidf_01.model")  # save model
    #print(lsi_tfidf.print_topics(5, 10))

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