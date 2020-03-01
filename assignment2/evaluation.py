
# coding: utf-8

# In[1]:


import os
import json
import pickle as pkl
import numpy as np
import pytrec_eval
import read_ap
import download_ap
import scipy.stats
import timeit
from doc2vec import Doc2Vec

from collections import defaultdict, Counter
from tf_idf import TfIdfRetrieval


# In[2]:


#write results function

def write_results(model, mdic):
    results_path = "results"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        
    #dump metrics to JSON
    with open(os.path.join(results_path, model+".json"), "w") as writer:
        json.dump(mdic["metrics"], writer, indent=1)
        
    #write file with all query-doc pairs, scores, ranks, etc.
    f = open(os.path.join(results_path, model+".dat"), "w")
    for qid in mdic["results"]:
        prevscore = 1e9
        for rank, docid in enumerate(mdic["results"][qid], 1):
            score = mdic["results"][qid][docid]
            if score > prevscore:
                f.close()
                raise Exception("'results_dic' not ordered! Stopped writing results")
            f.write(f"{qid} Q0 {docid} {rank} {score} STANDARD\n")
            prevscore = score
    f.close()
    
def perform_ttest(m1, m2, metric, models, thresh=0.05, print_res=True):
    #if pvalue < thresh (usually 0.05), then diff is significant
    
    qids = [qid for qid in models[m1]["metrics"]]
    scores1 = [models[m1]["metrics"][qid][metric] for qid in qids]
    scores2 = [models[m2]["metrics"][qid][metric] for qid in qids]   
    for i in range(len(scores1)):
        scores2[i] += np.random.normal(0,0.001) + 0.0001
    pvalue = scipy.stats.ttest_rel(scores1, scores2).pvalue
    conclusion = "significant diff" if pvalue < thresh else "insignificant diff"
    print("{:<12} {:<12} {:<19} {:<7} p-value = {:<5.3}".format(m1, m2, conclusion, "("+metric+")", pvalue))
    return pvalue


# In[3]:


#read data

docs = read_ap.get_processed_docs()
qrels, queries = read_ap.read_qrels()

print('done reading data')
# In[4]:


#prepare models
#for Doc2Vec
wind_size = 15
embedding_dim = 300
min_count = 5

models = {}

models["TF-IDF"]     = {"model": TfIdfRetrieval(docs), "results": {}, "metrics": {}}
# models["word2vec"]   = {"model": ..., "results": {}, "metrics": {}}
models["doc2vec"]    = {"model": Doc2Vec(docs, wind_size, embedding_dim, min_count=min_count), "results": {}, "metrics": {}}
# models["LSI-BoW"]    = {"model": ..., "results": {}, "metrics": {}}
# models["LSI-TF-IDF"] = {"model": ..., "results": {}, "metrics": {}}
# models["LDA"]        = {"model": ..., "results": {}, "metrics": {}}


# In[5]:


#run each model for each query

for qid in qrels: 
    query_text = queries[qid]

    #this might be slightly different for each model
    models["TF-IDF"]["results"][qid] = dict(models["TF-IDF"]["model"].search(query_text))
    # models["word2vec"]["results"]   = ...
    models["doc2vec"]["results"][qid] = dict(models["doc2vec"]["model"].search(query_text))
    # models["LSI-BoW"]["results"]    = ...
    # models["LSI-TF-IDF"]["results"] = ...
    # models["LDA"]["results"]        = ...


# In[6]:


#evaluate results

metrics = {'map', 'ndcg'}
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

for model in models:
    models[model]["metrics"] = evaluator.evaluate(models[model]["results"])


# In[7]:


#write results

for model in models:
    write_results(model, models[model])


# In[8]:


#perform t-tests

ttest = {}

for model1 in models:
    for model2 in models:
        if model1 != model2:
            #or, to reduce redundancy:
#         if model1 != model2 and model1+" "+model2 not in ttest and model2+" "+model1 not in ttest:      
            for metric in metrics:
                ttest[model1+" "+model2] = {metric: perform_ttest(model1, model2, metric, models)}

