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
import json



class W2v:
    def __init__(self, wind_size, docs=None, embedding_dim=300):
        # if docs is not given, the embeddings are loaded from file
        if docs == None:
            self.load_embedding(wind_size)
            self.embedding_dim = embedding_dim
            self.doc_vecs = None
        else:
            self.doc_ids = docs.keys()
            self.docs = docs
            index_path = "./tfidf_index"
            if os.path.exists(index_path):

                with open(index_path, "rb") as reader:
                    index = pkl.load(reader)

                self.ii = index["ii"]
                self.df = index["df"]
                print('done2')
            else:
                self.ii = defaultdict(list)
                self.df = defaultdict(int)

                doc_ids = list(docs.keys())

                print("Building Index")
                # build an inverted index
                for doc_id in tqdm(doc_ids):
                    doc = docs[doc_id]

                    counts = Counter(doc)
                    for t, c in counts.items():
                        self.ii[t].append((doc_id, c))
                    # count df only once - use the keys
                    for t in counts:
                        self.df[t] += 1

                with open(index_path, "wb") as writer:
                    index = {
                        "ii": self.ii,
                        "df": self.df
                    }
                    pkl.dump(index, writer)

            accepted_words = set()
            for token in self.ii:
                occurrences = [c[1] for c in self.ii[token]]
                occurrences = sum(occurrences)
                if occurrences>50:
                    accepted_words.add(token)

            # start index from 1 and reserve 0 for unknown words
            self.word2idx = {w: idx+1 for (idx, w) in enumerate(accepted_words)}
            self.idx2word = {idx+1: w for (idx, w) in enumerate(accepted_words)}
            self.word2idx['<unk>'] = 0
            self.idx2word[0] = '<unk>'
            self.vocab = self.word2idx.keys()
            self.embedding = None
            self.embedding_dim = embedding_dim
            self.doc_vecs = None

    def load_embedding(self, wind_size):
        weights = torch.tensor(np.load('./w2v_weights_'+str(wind_size)+'.npy', allow_pickle=True))
        self.embedding = nn.Embedding.from_pretrained(weights)
        with open('./word2idx_'+str(wind_size)+'.pickle', 'rb') as pickle_file:
            self.word2idx = pkl.load(pickle_file)
        with open('./idx2word_'+str(wind_size)+'.pickle', 'rb') as pickle_file:
            self.idx2word = pkl.load(pickle_file)


    def get_pairs(self, num, wind_size):
        # gets positive and negative pairs. Gets three times more negative pairs
        pos_pairs = []
        neg_pairs = []
        count = 0
        while count < num:
            doc_id = random.sample(self.doc_ids, 1)[0]
            indices = [self.word2idx[word] if word in self.word2idx else 0 for word in self.docs[doc_id]]
            for pos in range(len(indices)):
                context_poses = list(range(pos-wind_size, pos+wind_size+1))
                contexts = [indices[cont] for cont in context_poses if (0 <= cont < len(indices))]
                for context_pos in context_poses:
                    if context_pos == pos:
                        continue
                    if context_pos < 0 or context_pos >= len(indices):
                        continue
                    pos_pairs.append((indices[pos], indices[context_pos], 1))
                    count += 1
                    if count >= num:
                        break

                    # get negative samples
                    for j in range(3):
                        # each word has the same probability to be picked
                        neg_sample = random.sample(range(len(self.vocab)), 1)
                        # exclude the actual target words
                        if neg_sample in contexts:
                            continue
                        neg_pairs.append((indices[pos], random.sample(range(len(self.vocab)), 1)[0], 0))
                        count += 1
                        if count >= num:
                            break
                    if count >= num:
                        break
                if count >= num:
                    break

        pairs = np.array(pos_pairs+neg_pairs)
        np.random.shuffle(pairs)
        return pairs


    def train_nn(self, embedding_dim, wind_size):
        iterations = 200000

        l1 = nn.Embedding(len(self.vocab), embedding_dim, sparse=True)
        l2 = nn.Embedding(len(self.vocab), embedding_dim, sparse=True)
        params = list(l1.parameters()) + list(l2.parameters())
        lr = 0.001
        batch_size = 1024
        optimizer = torch.optim.SparseAdam(params, lr=lr)
        criterion = nn.BCELoss()

        for i in range(iterations):
            optimizer.zero_grad()
            pairs = self.get_pairs(batch_size, wind_size)
            center = torch.tensor(pairs[:, 0]).long()
            context = torch.tensor(pairs[:, 1]).long()
            labels = torch.tensor(pairs[:, 2]).float()

            out1 = l1(center)
            out2 = l2(context)
            shape = out1.shape
            dot_prods = torch.bmm(out1.view(shape[0], 1, shape[1]), out2.view(shape[0], shape[1], 1))
            dot_prods = dot_prods.squeeze()

            logits = torch.sigmoid(dot_prods)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print("Iteration {}: {}".format(i, loss))
        wvs = l1.weight.data.cpu().numpy()
        np.save('./w2v_weights_'+str(wind_size)+'.npy', wvs)
        with open('./word2idx_'+str(wind_size)+'.pickle', 'wb') as pickle_file:
            pkl.dump(self.word2idx, pickle_file)
        with open('./idx2word_'+str(wind_size)+'.pickle', 'wb') as pickle_file:
            pkl.dump(self.idx2word, pickle_file)
        self.embedding = l1

    # gets k most similar words
    def most_similar(self, word, k):
        word_vector = self.get_word_vec(word)
        similarities = torch.zeros(len(self.idx2word))
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in self.idx2word:
            similarities[i] = abs(cos(word_vector, self.get_word_vec(self.idx2word[i])))
        indices = (-similarities.numpy()).argsort()[:k]
        results = [self.idx2word[idx] for idx in indices]
        return results

    def get_word_vec(self, word):
        if word in self.word2idx:
            return self.embedding(torch.tensor(self.word2idx[word], dtype=torch.long))
        else:
            # if word is not in vocab, use index for unknown words (0)
            return self.embedding(torch.tensor(0, dtype=torch.long))

    def get_doc_vec(self, doc, agg_mode=torch.mean):
        wvs = torch.zeros(len(doc), self.embedding_dim)
        for i, token in enumerate(doc):
            wvs[i] = self.get_word_vec(token)
        doc_vec = agg_mode(wvs, dim=0)
        return doc_vec

    def get_doc_vecs(self, docs):
        self.docs = docs
        print('getting vectors')
        doc_vecs_list = []
        idx2docid = {}
        for i, doc_id in enumerate(tqdm(self.docs)):
            doc_vecs_list.append(self.get_doc_vec(self.docs[doc_id]))
            idx2docid[i] = doc_id
        doc_vecs = torch.stack(doc_vecs_list, dim=1)
        self.doc_vecs = doc_vecs
        self.idx2docid = idx2docid
        with open('v2w_docvecs.pkl', 'wb') as writer:
            pkl.dump(self.doc_vecs, writer)
        with open('v2w_docvecs_idx.pkl', 'wb') as writer:
            pkl.dump(self.idx2docid, writer)

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
    download_ap.download_dataset()
    window_size = 5
    embedding_dim = 300

    docs_by_id = read_ap.get_processed_docs()
    # uncomment the following to train word2vec
    #w2v = W2v(window_size, docs_by_id, embedding_dim)
    #w2v.train_nn(embedding_dim, window_size)
    #print('done training')
    w2v = W2v(window_size, embedding_dim=embedding_dim)
    #uncomment the following to get k most similar words (word must be stemmed)
    #word = 'green'
    #k = 10
    #similar = w2v.most_similar(word, k)
    #print(similar)

    # get queries
    qrels, queries = read_ap.read_qrels()
    overall_ser = {}
    w2v.get_doc_vecs(docs_by_id)
    for qid in tqdm(qrels):
        query_text = queries[qid]
        results = w2v.search(query_text)
        overall_ser[qid] = dict(results)
    with open("w2v_ranking.json", "w") as writer:
        json.dump(overall_ser, writer, indent=1)
