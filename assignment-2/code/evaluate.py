from tqdm import tqdm
import pytrec_eval
import json


def evaluate_model(model, qrels, queries, json_path_name, trec_path_name, run):
    overall_ser = {}

    print("Running Evaluation...")
    first_query = True
    # collect results
    for qid in tqdm(qrels):
        query_text = queries[qid]

        results_lsi_bow = model.rank(query_text, first_query=first_query)
        overall_ser[qid] = dict(results_lsi_bow)
        first_query = False
    # run evaluation with `qrels` as the ground truth relevance judgements
    # here, we are measuring MAP and NDCG, but this can be changed to
    # whatever you prefer
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    print('get metrics...')
    metrics = evaluator.evaluate(overall_ser)
    print('done')

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open(json_path_name, "w") as writer:
        json.dump(metrics, writer, indent=1)

    # write trec file with all query-doc pairs, scores, ranks, etc.
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

