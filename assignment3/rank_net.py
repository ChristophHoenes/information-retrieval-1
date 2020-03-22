import dataset
import ranking as rnk
import evaluate as evl

import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from time import time
from itertools import combinations, product

import pickle


class Rank_Net(nn.Module):
    def __init__(self, d_in, num_neurons=[200, 100], sigma=1.0, dropout=0.0, device='cpu'):
        assert isinstance(num_neurons, list) or isinstance(num_neurons, int), "num_neurons must be either an int (one layer) or a list of ints"
        if isinstance(num_neurons, int):
            num_neurons = [num_neurons]

        super(Rank_Net, self).__init__()
        self.d_in = d_in
        self.num_neurons = num_neurons
        self.num_neurons.insert(0, self.d_in)
        self.num_neurons.append(1)
        self.sigma = sigma
        self.dropout = dropout
        self.device = device
        self.model_id = np.random.randint(1000000, 9999999)
        layers = []
        for h, h_next in zip(num_neurons, num_neurons[1:]):
            layers.append(nn.Linear(h, h_next))
            #layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        #layers.pop()
        #layers.append(nn.ReLU6())
        #layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        self.layers.to(self.device)
        return self.layers(x)

    def train_sgd_speed4(self, data, lr=2e-3, batch_size=500, num_epochs=1, ndcg_convergence=0.95, eval_freq=1000):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            random_query_order = np.arange(num_queries)
            np.random.shuffle(random_query_order)
            all_loss = 0
            completed_batches = 0
            for qid in tqdm(range(num_queries)):
                x = torch.Tensor(data.train.query_feat(random_query_order[qid])).to(self.device)
                query_scores = self.layers(x)
                query_labels = data.train.query_labels(random_query_order[qid])

                # to catch cases with less than two documents (as no loss can be computed if there is no document pair)
                if len(query_labels) < 2:
                    continue

                score_combs = list(zip(*product(query_scores, repeat=2)))
                score_combs_i = torch.stack(score_combs[0]).squeeze()
                score_combs_j = torch.stack(score_combs[1]).squeeze()

                label_combs = np.array(list(product(query_labels, repeat=2)))
                label_combs_i = torch.from_numpy(label_combs[:, 0])
                label_combs_j = torch.from_numpy(label_combs[:, 1])

                loss = self.pair_cross_entropy_vectorized(score_combs_i, score_combs_j,
                                                            label_combs_i, label_combs_j)

                all_loss += loss

                if qid % batch_size == 0 or qid == num_queries-1:
                    completed_batches += 1
                    all_loss /= (batch_size if qid % batch_size == 0 else num_queries % batch_size)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if eval_freq != 0:
                        if qid % eval_freq == 0:
                            print('Loss epoch {}: {} after query {} of {} queries'.format(e, loss.item(), qid,
                                                                                          num_queries))
                            with torch.no_grad():
                                self.layers.eval()
                                x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                                validation_scores = self.layers(x).squeeze().numpy()
                            # eval_scores, eval_labels = self.prepare_for_evaluate(data.validation.label_vector,validation_scores)
                            # ndcg_result = evl.ndcg_at_k(eval_scores, eval_labels, 0)
                            ndcg_result = evl.evaluate(data.validation, validation_scores)['ndcg'][0]
                            print('NDCG score: {}'.format(ndcg_result))
                            if ndcg_result > ndcg_convergence:
                                converged = True
                                print(
                                    'Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence,
                                                                                                       e))
                                break
        return self.layers, self.evaluate(data.validation)['ndcg']

    def prepare_for_evaluate(self, labels, scores):
        n_docs = labels.shape[0]

        random_i = np.random.permutation(
            np.arange(scores.shape[0])
        )
        labels = labels[random_i]
        scores = scores[random_i]

        sort_ind = np.argsort(scores)[::-1]
        sorted_labels = labels[sort_ind]
        ideal_labels = np.sort(labels)[::-1]
        return sorted_labels, ideal_labels

    def train_sgd_speed5(self, data, lr=5e-5, num_epochs=1, ndcg_convergence=0.95, eval_freq=1000):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            random_query_order = np.arange(num_queries)
            np.random.shuffle(random_query_order)
            for qid in tqdm(range(num_queries)):
                x = torch.Tensor(data.train.query_feat(random_query_order[qid])).to(self.device)
                query_scores = self.layers(x)
                query_labels = data.train.query_labels(random_query_order[qid])

                # to catch cases with less than two documents (as no loss can be computed if there is no document pair)
                if len(query_labels) < 2:
                    continue

                score_combs = list(zip(*product(query_scores, repeat=2)))
                score_combs_i = torch.stack(score_combs[0]).squeeze()
                score_combs_j = torch.stack(score_combs[1]).squeeze()

                label_combs = np.array(list(product(query_labels, repeat=2)))
                label_combs_i = torch.from_numpy(label_combs[:, 0])
                label_combs_j = torch.from_numpy(label_combs[:, 1])

                loss = self.pair_cross_entropy_vectorized(score_combs_i, score_combs_j, label_combs_i, label_combs_j)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if eval_freq != 0:
                    if qid % eval_freq == 0:
                        print('Loss epoch {}: {} after query {} of {} queries'.format(e, loss.item(), qid, num_queries))
                        with torch.no_grad():
                            self.layers.eval()
                            x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                            validation_scores = self.layers(x).squeeze().numpy()
                        #eval_scores, eval_labels = self.prepare_for_evaluate(data.validation.label_vector,validation_scores)
                        #ndcg_result = evl.ndcg_at_k(eval_scores, eval_labels, 0)
                        ndcg_result = evl.evaluate(data.validation, validation_scores)['ndcg'][0]
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > ndcg_convergence:
                            converged = True
                            print('Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence, e))
                            break
        return self.layers, self.evaluate(data.validation)['ndcg']

    def pair_cross_entropy_vectorized(self, s_i, s_j, S_i, S_j):
        S_diff = S_i - S_j
        ones = torch.ones_like(S_diff)
        zeros = torch.zeros_like(S_diff)
        Sij = torch.where(S_diff > 0, ones, -ones)
        Sij = torch.where(S_diff == 0, zeros, Sij)
        sig_diff = self.sigma * (s_i - s_j)
        return torch.sum(0.5 * (1 - Sij) * sig_diff + torch.log(1 + torch.exp(-sig_diff)))

    def evaluate(self, data_fold, print_results=False):
        self.layers.eval()
        with torch.no_grad():
            x = torch.Tensor(data_fold.feature_matrix).to(self.device)
            validation_scores = self.layers(x).squeeze().numpy()
        results = evl.evaluate(data_fold, validation_scores, print_results=print_results)
        return results

    def save(self, path='./rank_net'):
        torch.save(self.state_dict(), path)

class Rank_Net_Sped_Up(Rank_Net):

    def train_sgd_speed4(self, data, lr=1e-3, batch_size=500, num_epochs=1, ndcg_convergence=0.95, eval_freq=0):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        prev_ndcg = 0
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            random_query_order = np.arange(num_queries)
            np.random.shuffle(random_query_order)
            # num_batches = num_queries/batch_size + (1 if num_queries % batch_size > 0 else 0)
            completed_batches = 0
            skipped = False
            for qid in tqdm(range(num_queries)):
                # s_i, e_i = data.train.query_range(random_query_order[qid])
                x = torch.Tensor(data.train.query_feat(random_query_order[qid])).to(self.device)
                query_scores = self.layers(x)
                # query_scores = all_scores[s_i:e_i]
                query_labels = data.train.query_labels(random_query_order[qid])

                # to catch cases with less than two documents (as no loss can be computed if there is no document pair)
                if len(query_labels) < 2:
                    if qid % batch_size == 1:
                        skipped = True
                    continue

                score_combs = list(zip(*product(query_scores.detach(), repeat=2)))
                score_combs_i = torch.stack(score_combs[0]).squeeze()
                score_combs_j = torch.stack(score_combs[1]).squeeze()

                label_combs = np.array(list(product(query_labels, repeat=2)))
                label_combs_i = torch.from_numpy(label_combs[:, 0])
                label_combs_j = torch.from_numpy(label_combs[:, 1])

                lambda_i = self.rank_net_loss(score_combs_i, score_combs_j,
                                                label_combs_i, label_combs_j)
                if qid == 0 or qid % batch_size == 1 or skipped:
                    batch_lambdas = lambda_i
                    batch_scores = query_scores
                    skipped = False
                else:
                    batch_lambdas = torch.cat((batch_lambdas, lambda_i), 0)
                    batch_scores = torch.cat((batch_scores, query_scores), 0)

                if qid % batch_size == 0 or qid == num_queries - 1:
                    completed_batches += 1
                    optimizer.zero_grad()
                    batch_scores.backward(batch_lambdas)
                    optimizer.step()

                    if eval_freq != 0:
                        if qid % eval_freq == 0:
                            print('Average Gradient epoch {}: {} after query {} of {} queries'.format(e, lambda_i, qid,
                                                                                                      num_queries))
                            # with torch.no_grad():
                            #     self.layers.eval()
                            #     x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                            #     validation_scores = self.layers(x)
                            ndcg_result = self.evaluate(data.validation)['ndcg'][0]
                            print('NDCG score: {}'.format(ndcg_result))
                            if ndcg_result > ndcg_convergence or ndcg_result - prev_ndcg < 0.001:
                                converged = True
                                print(
                                    'Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence,
                                                                                                       e))
                                break
                            else:
                                prev_ndcg = ndcg_result

                if not converged:
                    print('Done training for {} epochs'.format(num_epochs))
                return self.layers, self.evaluate(data.validation)['ndcg']

    def train_sgd_speed5(self, data, lr=5e-5, num_epochs=1, ndcg_convergence=0.95, eval_freq=7000):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        prev_ndcg = 0
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            random_query_order = np.arange(num_queries)
            np.random.shuffle(random_query_order)
            for qid in tqdm(range(num_queries)):
                x = torch.Tensor(data.train.query_feat(random_query_order[qid])).to(self.device)
                query_scores = self.layers(x)
                query_labels = data.train.query_labels(random_query_order[qid])

                # to catch cases with less than two documents (as no loss can be computed if there is no document pair)
                if len(query_labels) < 2:
                    continue

                score_combs = list(zip(*product(query_scores.detach(), repeat=2)))
                score_combs_i = torch.stack(score_combs[0]).squeeze()
                score_combs_j = torch.stack(score_combs[1]).squeeze()

                label_combs = np.array(list(product(query_labels, repeat=2)))
                label_combs_i = torch.from_numpy(label_combs[:, 0])
                label_combs_j = torch.from_numpy(label_combs[:, 1])

                lambda_i = self.rank_net_loss(score_combs_i, score_combs_j,
                                                label_combs_i, label_combs_j)

                optimizer.zero_grad()
                query_scores.backward(lambda_i)
                optimizer.step()


                if eval_freq != 0:
                    if qid % eval_freq == 0:
                        print('Average Gradient epoch {}: {} after query {} of {} queries'.format(e, lambda_i, qid,
                                                                                              num_queries))
                        # with torch.no_grad():
                        #     self.layers.eval()
                        #     x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                        #     validation_scores = self.layers(x)
                        ndcg_result = self.evaluate(data.validation)['ndcg'][0]
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > ndcg_convergence or ndcg_result-prev_ndcg < 0.001:
                            converged = True
                            print(
                                'Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence,
                                                                                                   e))
                            break
                        else:
                            prev_ndcg = ndcg_result

        if not converged:
            print('Done training for {} epochs'.format(num_epochs))
        return self.layers, self.evaluate(data.validation)['ndcg']

    def rank_net_loss(self, s_i, s_j, S_i, S_j):
        S_diff = S_i - S_j
        Sij = np.where(S_diff > 0, 1.0, -1.0)
        Sij = torch.from_numpy(np.where(S_diff == 0, 0.0, Sij))
        sig_diff = self.sigma * (s_i - s_j)
        num_docs = int(math.sqrt(s_i.shape[0]))
        lambda_ij = (self.sigma * (0.5 * (1 - Sij) - 1/(1 + torch.exp(sig_diff)))).view(num_docs,num_docs)
        return torch.sum(lambda_ij, dim=1).unsqueeze_(dim=1)

    def calculate_gradients(self, lambdas, model_output):
        model_output.backward(lambdas)

    def create_valid_pairs(self, query_scores, query_labels):
        score_combs = list(zip(*product(query_scores.detach(), repeat=2)))
        score_combs_i = torch.stack(score_combs[0]).squeeze()
        score_combs_j = torch.stack(score_combs[1]).squeeze()

        label_combs = np.array(list(product(query_labels, repeat=2)))
        label_combs_i = torch.from_numpy(label_combs[:, 0])
        label_combs_j = torch.from_numpy(label_combs[:, 1])
        return score_combs_i, score_combs_j, label_combs_i, label_combs_j

def hyperparameter_search():
    lrs = [1e-5, 1e-3, 1e-4]
    hidden_layers = [[200], [200, 100], [200,100,50]]
    irms = ['ndcg']
    sigmas = [0.5, 1.0, 2.0]
    best_ndcg = 0
    best_model = None
    best_ndcg2 = 0
    best_model2 = None
    for irm in irms:
        print(irm)
        for lr in lrs:
            for hidden_layer in hidden_layers:
                for sigma in sigmas:
                    model = Rank_Net(data.num_features,hidden_layer,sigma=sigma)
                    model2 = Rank_Net_Sped_Up(data.num_features,hidden_layer,sigma=sigma)
                    model, ndcg = model.train_sgd_speed5(data, lr=lr, num_epochs=2, eval_freq=0)
                    model2, ndcg2 = model2.train_sgd_speed5(data, lr=lr, num_epochs=2, eval_freq=0)
                    print(ndcg)
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_model = model
                        best_ndcg2 = ndcg2
                        best_model2 = model2
    print(best_ndcg)
    torch.save(best_model, './hp_search_best')
    print(best_ndcg2)
    torch.save(best_model2, './hp_search_best2')

def r(g, g_max=4):
    return (2**g-1 / 2**g_max)

def err(ranking_labels):
    p = 1.0
    ERR = 0.0
    for r in range(len(ranking_labels)):
        R = r(ranking_labels[r])
        ERR += p * R/(r+1)
        p *= 1-R
    return ERR


if __name__ == "__main__":
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    #hyperparameter_search()
    net = Rank_Net(data.num_features, sigma=1.0)
    start = time()
    net.train_sgd_speed4(data, num_epochs=1)
    end = time()
    print('Finished training in {} minutes'.format((end-start)/60))
    net.save(path='./rank_net'+str(net.model_id)+'.weights')
    final_test_results = net.evaluate(data.test,print_results=True)
    with open('eval'+str(net.model_id), 'wb') as f:
        pickle.dump(final_test_results, f)

    # net2 = Rank_Net_Sped_Up(data.num_features, sigma=1)
    # start = time()
    # net2.train_sgd_speed5(data, num_epochs=1)
    # end = time()
    # print('Finished training in {} minutes'.format((end - start) / 60))
    # net2.save(path='./rank_net' + str(net2.model_id) + '.weights')
    # final_test_results = net2.evaluate(data.test, print_results=True)
    # with open('eval' + str(net2.model_id), 'wb') as f:
    #     pickle.dump(final_test_results, f)

    # net = Rank_Net(data.num_features)
    # start = time()
    # net.train_bgd2_retain(data)
    # end = time()
    # print('Finished training in {} minutes'.format((end - start) / 60))
    # net.save(path='./rank_net' + str(net.model_id) + '.weights')
    # final_test_results = net.evaluate(data.test, print_results=True)
    # with open('eval' + str(net.model_id), 'wb') as f:
    #     pickle.dump(final_test_results, f)

