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



class W2v:
    def __init__(self, wind_size, docs=None):
        if docs == None:
            self.load_embedding(wind_size)
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
            #    print(count/num)
            for pos in range(len(indices)):
                context_poses = list(range(pos-wind_size, pos+wind_size+1))

                try:
                    contexts = [indices[cont] for cont in context_poses if (0 <= cont < len(indices))]
                except Exception:
                    print('here', context_poses)
                    print(indices)
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

        l1 = nn.Embedding(len(self.vocab), embedding_dim, sparse=True).cuda()
        l2 = nn.Embedding(len(self.vocab), embedding_dim, sparse=True).cuda()
        params = list(l1.parameters()) + list(l2.parameters())
        lr = 0.001
        batch_size = 1024
        optimizer = torch.optim.SparseAdam(params, lr=lr)
        criterion = nn.BCELoss()

        for i in range(iterations):
            optimizer.zero_grad()
            pairs = self.get_pairs(batch_size, wind_size)
            center = torch.tensor(pairs[:, 0]).long().cuda()
            context = torch.tensor(pairs[:, 1]).long().cuda()
            labels = torch.tensor(pairs[:, 2]).float().cuda()

            #input_center = torch.zeros(batch_size, len(self.vocab))
            #input_context = torch.zeros(batch_size, len(self.vocab))
            """
            for j in range(len(center)):
                input_center[j, center[j]] = 1.0
                input_context[j, context[j]] = 1.0
            """
            out1 = l1(center)
            out2 = l2(context)
            shape = out1.shape
            #print(shape)
            dot_prods = torch.bmm(out1.view(shape[0], 1, shape[1]), out2.view(shape[0], shape[1], 1))
            dot_prods = dot_prods.squeeze()

            logits = torch.sigmoid(dot_prods)
            loss = criterion(logits, labels)
            #print("Iteration {}: {}".format(i, loss))
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print("Iteration {}: {}".format(i, loss))
        wvs = l1.weight.data.cpu().numpy()
        #print(wvs)
        np.save('./w2v_weights_'+str(wind_size)+'.npy', wvs)
        with open('./word2idx_'+str(wind_size)+'.pickle', 'wb') as pickle_file:
            pkl.dump(self.word2idx, pickle_file)
        with open('./idx2word_'+str(wind_size)+'.pickle', 'wb') as pickle_file:
            pkl.dump(self.idx2word, pickle_file)
        self.embedding = l1

    def most_similar(self, word, k):
        word_vector = self.get_word_vec(word)
        similarities = torch.zeros(len(self.idx2word))
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in self.idx2word:
            similarities[i] = abs(cos(word_vector, self.get_word_vec(self.idx2word[i])))
        #print(similarities.shape)
        indices = (-similarities.numpy()).argsort()[:k]
        results = [self.idx2word[idx] for idx in indices]
        return results

    def get_word_vec(self, word):
        return self.embedding(torch.tensor(self.word2idx[word], dtype=torch.long))

if __name__ == "__main__":
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    """
    docs_path = "./w2v_docs"
    if os.path.exists(docs_path):
        with open(docs_path, "rb") as reader:
            docs_by_id = pkl.load(reader)
    else:
        docs_by_id = read_ap.get_processed_docs()
        with open(docs_path, "wb") as writer:
            pkl.dump(docs_by_id, writer)
    """
    window_size = 5
    embedding_dim = 300
    docs_by_id = read_ap.get_processed_docs()
    #print('done')
    w2v = W2v(window_size, docs_by_id)
    w2v.train_nn(embedding_dim, window_size)
    print('done training')
    #w2v = W2v(window_size)
    #embedding = load_embedding()
    #print(weights.shape)
    #word = 'dog'
    #k = 10
    #similar = w2v.most_similar(word, k)
    #print(similar)
