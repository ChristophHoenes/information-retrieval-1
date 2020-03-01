import torch
from torch import nn
from tqdm import tqdm
import read_ap
import random
import download_ap
import gensim
import logging
from scipy.spatial.distance import cosine
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Doc2Vec:
    def __init__(self, docs, wind_size=15, embedding_dim=200, min_count=50):
        corpus = self.read_docs(docs)
        self.docs = docs
        model = gensim.models.doc2vec.Doc2Vec(vector_size=embedding_dim, window=wind_size, min_count=min_count, workers=4, epochs=4)
        print('building vocab...')
        model.build_vocab(corpus)
        print('Vocab Size: ', len(model.wv.vocab))
        print('training...')
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        print('done training')
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        self.model = model

    def read_docs(self, docs):
        corpus = []
        for i, doc in enumerate(docs):
            corpus.append(gensim.models.doc2vec.TaggedDocument(docs[doc], [doc]))
        return corpus

    def get_doc_vec(self, tokens):
        vector = self.model.infer_vector(tokens)
        return torch.tensor(vector)

    # finds n most similar docs for a doc (given as doc_id)
    def find_most_similar(self, doc_id, n, orig_docs):
        orig = self.get_doc_vec(self.docs[doc_id])
        similarities = {}
        for doc in self.docs:
            prod = cosine(orig, self.get_doc_vec(self.docs[doc]))
            similarities[doc] = prod
        ranking = [(k, orig_docs[k], v) for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)]
        results = {0: orig_docs[doc_id]}
        for i, item in enumerate(ranking[:n]):
            results[i+1] = item
        return results

    def get_doc_vecs(self):
        print('getting vectors')
        doc_vecs_list = []
        idx2docid = {}
        for i, doc_id in enumerate(tqdm(self.docs)):
            doc_vecs_list.append(self.get_doc_vec(self.docs[doc_id]))
            idx2docid[i] = doc_id
        doc_vecs = torch.stack(doc_vecs_list, dim=1)
        print(doc_vecs.shape)
        self.doc_vecs = doc_vecs
        self.idx2docid = idx2docid

    def search(self, query):
        query_repr = read_ap.process_text(query)
        orig = self.get_doc_vec(query_repr)
        orig = orig.unsqueeze(1).repeat(1, len(self.docs))
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        prod = cos(orig, self.doc_vecs)
        indices = (-prod.numpy()).argsort()
        results = [(self.idx2docid[index], float(prod.numpy()[index])) for index in indices]
        return results


if __name__ == "__main__":
    # ensure dataset is downloaded
    wind_size = 10
    embedding_dim = 200

    download_ap.download_dataset()
    docs_by_id = read_ap.get_processed_docs()
    d2v = Doc2Vec(docs_by_id, wind_size, embedding_dim)
    doc_id = random.sample(docs_by_id.keys(), 1)[0]

    # finds most similar docs for a random doc

    #orig_doc_text, orig_doc_ids = read_ap.read_ap_docs()
    #orig_docs = {}
    #for i, idx in enumerate(orig_doc_ids):
    #    orig_docs[idx] = orig_doc_text[i]

    #results = d2v.find_most_similar(doc_id, 10, orig_docs)

    #with open('doc2vec_similarities.json', 'w') as f:
    #    json.dump(results, f, indent=1)
