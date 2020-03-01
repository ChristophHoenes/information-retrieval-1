from doc2vec import Doc2Vec
import read_ap
from tqdm import tqdm
import pytrec_eval
import json
import numpy as np
"""
wind_sizes = [5]#[5, 10, 15, 20]
vector_dims = [200]#[300, 400, 500]
# for each vocabulary size, the following min_count values approximate them {10: 250, 25: 50, 50: 15, 100: 5, 200: 2}
vocab_dict = {250:'10k', 50: '25k', 15: '50k', 5: '100k', 2: '200k'}
vocab_sizes = {250, 50, 15, 5, 2}
qrels, queries = read_ap.read_qrels()
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
maps = {}
ndcgs = []

maps = {'embedding_dim': {}}

for embedding_dim in vector_dims:
    with open("d2v_vecdim_"+str(embedding_dim)+".json", "r") as reader:
        overall_ser = json.load(reader)
    metrics = evaluator.evaluate(overall_ser)
    #print(metrics)
    with open("d2v_"+str(embedding_dim)+".json", "w") as writer:
        json.dump(metrics, writer, indent=1)
    break
    tmp1 = np.zeros(len(metrics))
    tmp2 = np.zeros(len(metrics))
    #print(tmp1)
    for i, q in enumerate(metrics):
        #print(i, metrics[q]['map'])
        tmp1[i] = metrics[q]['map']
        tmp2[i] = metrics[q]['ndcg']
    maps['embedding_dim'][embedding_dim] = np.average(tmp1)

maps['wind_size'] = {}
for wind_size in wind_sizes:
    with open("d2v_vecdim_200"+str(wind_size)+".json", "r") as reader:
        overall_ser = json.load(reader)
    metrics = evaluator.evaluate(overall_ser)
    with open("d2v_bestvalid_metrics_"+str(wind_size)+".json", "w") as writer:
        json.dump(metrics, writer, indent=1)
    tmp1 = np.zeros(len(metrics))
    tmp2 = np.zeros(len(metrics))
    for i, q in enumerate(metrics):
        tmp1[i] = (metrics[q]['map'])
        tmp2[i] = (metrics[q]['ndcg'])
    maps['wind_size'][wind_size] = np.average(tmp1)

maps['vocab_size'] = {}
for vocab_size in vocab_sizes:
    with open("d2v_vocabsize_"+str(vocab_size)+".json", "r") as reader:
        overall_ser = json.load(reader)
    metrics = evaluator.evaluate(overall_ser)
    with open("d2v_"+vocab_dict[vocab_size]+".json", "w") as writer:
        json.dump(metrics, writer, indent=1)
    tmp1 = np.zeros(len(metrics))
    tmp2 = np.zeros(len(metrics))
    for i, q in enumerate(metrics):
        tmp1[i] = (metrics[q]['map'])
        tmp2[i] = (metrics[q]['ndcg'])
    maps['vocab_size'][vocab_size] = np.average(tmp1)
"""
def run_evaluation(in_json, json_path_name, trec_path_name, run):
    with open(in_json, 'r') as reader:
        overall_ser = json.load(reader)
    qrels, queries = read_ap.read_qrels()
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    print('get metrics...')
    metrics = evaluator.evaluate(overall_ser)
    print('done')

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open(json_path_name, "w") as writer:
        json.dump(metrics, writer, indent=1)

    # write file with all query-doc pairs, scores, ranks, etc.
    f = open(trec_path_name, "w")
    for qid in overall_ser:
        prevscore = 1e9
        for rank, docid in enumerate(overall_ser[qid], 1):
            score = overall_ser[qid][docid]
            if score > prevscore:
                f.close()
                raise Exception("'results_dic' not ordered! Stopped writing results")
            f.write(f"{qid} Q0 {docid} {rank} {score} {run}\n")
            prevscore = score
    f.close()

if __name__ == '__main__':
    input = 'w2v_ranking.json'
    file = 'w2v.json'
    trec_name = 'w2v.trec'
    run_evaluation(input, file, trec_name, '300_5_25k')