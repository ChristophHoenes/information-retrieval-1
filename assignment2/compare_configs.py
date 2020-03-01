import read_ap
import pytrec_eval
import json

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