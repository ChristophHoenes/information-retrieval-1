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
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        #layers.pop()
        #layers.append(nn.ReLU6())
        #layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        self.layers.to(self.device)
        return self.layers(x)

    def train_sgd_speed(self, data, lr=1e-3, batch_size=1, num_epochs=1, ndcg_convergence=0.95, eval_freq=0):
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
                s_i, e_i = data.train.query_range(random_query_order[qid])
                query_scores = self.layers(data.train.feature_matrix[s_i:e_i])
                query_labels = data.train.query_labels(random_query_order[qid])

                # to catch cases with less than two documents (as no loss can be computed if there is no document pair)
                if len(query_labels) < 2:
                    continue

                score_combs = list(zip(*combinations(query_scores, 2)))
                score_combs_i = torch.stack(score_combs[0]).squeeze()
                score_combs_j = torch.stack(score_combs[1]).squeeze()

                label_combs = np.array(list(combinations(query_labels, 2)))
                label_combs_i = torch.from_numpy(label_combs[:, 0])
                label_combs_j = torch.from_numpy(label_combs[:, 1])

                loss = self.pair_cross_entropy_vectorized(score_combs_i, score_combs_j,
                                                          label_combs_i, label_combs_j)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if eval_freq != 0:
                    if qid % eval_freq == 0:
                        print('Average Loss epoch {}: {} after query {} of {} queries'.format(e, loss.item(), qid, num_queries))
                        with torch.no_grad():
                            self.layers.eval()
                            validation_scores = torch.round(self.layers(data.validation.feature_matrix))
                        ndcg_result = evl.ndcg_at_k(validation_scores.numpy(), data.validation.label_vector, 0)
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > ndcg_convergence:
                            converged = True
                            print('Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence, e))
                            break
        if not converged:
            print('Done training for {} epochs'.format(num_epochs))

    def train_sgd_speed2(self, data, lr=1e-5, batch_size=1, num_epochs=1, ndcg_convergence=0.95, eval_freq=0):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            #random_query_order = np.arange(num_queries)
            #np.random.shuffle(random_query_order)
            all_scores = self.layers(data.train.feature_matrix)
            all_loss = 0
            for qid in tqdm(range(num_queries)):
                s_i, e_i = data.train.query_range(qid)#random_query_order[qid])
                query_scores = all_scores[s_i:e_i]
                query_labels = data.train.query_labels(qid)#random_query_order[qid])

                # to catch cases with less than two documents (as no loss can be computed if there is no document pair)
                if len(query_labels) < 2:
                    continue

                score_combs = list(zip(*combinations(query_scores, 2)))
                score_combs_i = torch.stack(score_combs[0]).squeeze()
                score_combs_j = torch.stack(score_combs[1]).squeeze()

                label_combs = np.array(list(combinations(query_labels, 2)))
                label_combs_i = torch.from_numpy(label_combs[:, 0])
                label_combs_j = torch.from_numpy(label_combs[:, 1])

                loss = self.pair_cross_entropy_vectorized(score_combs_i, score_combs_j,
                                                          label_combs_i, label_combs_j)

                all_loss += loss

                if eval_freq != 0:
                    if qid % eval_freq == 0:
                        print('Average Loss epoch {}: {} after query {} of {} queries'.format(e, loss.item(), qid, num_queries))
                        with torch.no_grad():
                            self.layers.eval()
                            validation_scores = torch.round(self.layers(data.validation.feature_matrix))
                        ndcg_result = evl.ndcg_at_k(validation_scores.numpy(), data.validation.label_vector, 0)
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > ndcg_convergence:
                            converged = True
                            print('Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence, e))
                            break
            all_loss /= num_queries
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not converged:
            print('Done training for {} epochs'.format(num_epochs))

    def train_sgd_speed3(self, data, lr=5e-5, batch_size=1000, num_epochs=1, ndcg_convergence=0.95, eval_freq=0):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            random_query_order = np.arange(num_queries)
            np.random.shuffle(random_query_order)
            #num_batches = num_queries/batch_size + (1 if num_queries % batch_size > 0 else 0)
            all_scores = self.layers(data.train.feature_matrix)
            all_loss = 0
            completed_batches = 0
            for qid in tqdm(range(num_queries)):
                s_i, e_i = data.train.query_range(random_query_order[qid])
                query_scores = all_scores[s_i:e_i]
                query_labels = data.train.query_labels(random_query_order[qid])

                # to catch cases with less than two documents (as no loss can be computed if there is no document pair)
                if len(query_labels) < 2:
                    continue

                score_combs = list(zip(*combinations(query_scores, 2)))
                score_combs_i = torch.stack(score_combs[0]).squeeze()
                score_combs_j = torch.stack(score_combs[1]).squeeze()

                label_combs = np.array(list(combinations(query_labels, 2)))
                label_combs_i = torch.from_numpy(label_combs[:, 0])
                label_combs_j = torch.from_numpy(label_combs[:, 1])

                loss = self.pair_cross_entropy_vectorized(score_combs_i, score_combs_j,
                                                          label_combs_i, label_combs_j)

                all_loss += loss

                if qid % batch_size == 0 or qid == num_queries-1:
                    completed_batches += 1
                    all_loss /= (batch_size if qid % batch_size == 0 else num_queries % batch_size)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=(True if qid % batch_size == 0 else False))
                    optimizer.step()

                    if eval_freq != 0:
                        if completed_batches % eval_freq == 0:
                            print('Average Loss epoch {}: {} after query {} of {} queries'.format(e, loss.item(), qid, num_queries))
                            with torch.no_grad():
                                self.layers.eval()
                                validation_scores = torch.round(self.layers(data.validation.feature_matrix))
                            ndcg_result = evl.ndcg_at_k(validation_scores.numpy(), data.validation.label_vector, 0)
                            print('NDCG score: {}'.format(ndcg_result))
                            if ndcg_result > ndcg_convergence:
                                converged = True
                                print('Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence, e))
                                break

        if not converged:
            print('Done training for {} epochs'.format(num_epochs))

    def train_sgd_speed4(self, data, lr=1e-4, batch_size=500, num_epochs=1, ndcg_convergence=0.95, eval_freq=0):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            random_query_order = np.arange(num_queries)
            np.random.shuffle(random_query_order)
            #num_batches = num_queries/batch_size + (1 if num_queries % batch_size > 0 else 0)
            all_loss = 0
            completed_batches = 0
            for qid in tqdm(range(num_queries)):
                #s_i, e_i = data.train.query_range(random_query_order[qid])
                x = torch.Tensor(data.train.query_feat(random_query_order[qid])).to(self.device)
                query_scores = self.layers(x)
                #query_scores = all_scores[s_i:e_i]
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
                        if completed_batches % eval_freq == 0:
                            print('Average Loss epoch {}: {} after query {} of {} queries'.format(e, loss.item(), qid, num_queries))
                            with torch.no_grad():
                                self.layers.eval()
                                x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                                validation_scores = torch.round(self.layers(x))
                            ndcg_result = evl.ndcg_at_k(validation_scores.numpy(), data.validation.label_vector, 0)
                            print('NDCG score: {}'.format(ndcg_result))
                            if ndcg_result > ndcg_convergence:
                                converged = True
                                print('Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence, e))
                                break

        if not converged:
            print('Done training for {} epochs'.format(num_epochs))


    def train_sgd_speed5(self, data, lr=1e-4, num_epochs=1, ndcg_convergence=0.95, eval_freq=0):
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

                loss = self.pair_cross_entropy_vectorized(score_combs_i, score_combs_j,
                                                            label_combs_i, label_combs_j)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if eval_freq != 0:
                    if completed_batches % eval_freq == 0:
                        print('Average Loss epoch {}: {} after query {} of {} queries'.format(e, loss.item(), qid, num_queries))
                        with torch.no_grad():
                            self.layers.eval()
                            x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                            validation_scores = torch.round(self.layers(x))
                        ndcg_result = evl.ndcg_at_k(validation_scores.numpy(), data.validation.label_vector, 0)
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > ndcg_convergence:
                            converged = True
                            print('Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence, e))
                            break

        if not converged:
            print('Done training for {} epochs'.format(num_epochs))

    def calculate_gradients(self, loss, model_output):
        loss.backward()

    def rank_net_loss(self, s_i, s_j, S_i, S_j):
        S_diff = S_i - S_j
        Sij = np.where(S_diff > 0, 1.0, -1.0)
        Sij = torch.from_numpy(np.where(S_diff == 0, 0.0, Sij))
        sig_diff = self.sigma * (s_i - s_j)
        return torch.sum(0.5 * (1 - Sij) * sig_diff + torch.log(1 + torch.exp(-sig_diff)))

    def create_valid_pairs(self, query_scores, query_labels):
        score_combs = list(zip(*combinations(query_scores, 2)))
        score_combs_i = torch.stack(score_combs[0]).squeeze()
        score_combs_j = torch.stack(score_combs[1]).squeeze()

        label_combs = np.array(list(combinations(query_labels, 2)))
        label_combs_i = torch.from_numpy(label_combs[:, 0])
        label_combs_j = torch.from_numpy(label_combs[:, 1])
        return score_combs_i, score_combs_j, label_combs_i, label_combs_j

    def aggregate_loss(self, all_losses, loss):
        return all_losses + loss


    def pair_cross_entropy(self, s_i, s_j, S_i, S_j):
        Sij = 1.0 if S_i > S_j else (-1.0 if S_i < S_j else 0.0)
        sig_diff = self.sigma * (s_i - s_j)
        return 0.5 * (1-Sij) * sig_diff + torch.log(1 + torch.exp(-sig_diff))

    def pair_cross_entropy_vectorized(self, s_i, s_j, S_i, S_j):
        S_diff = S_i - S_j
        Sij = np.where(S_diff > 0, 1.0, -1.0)
        Sij = torch.from_numpy(np.where(S_diff == 0, 0.0, Sij))
        sig_diff = self.sigma * (s_i - s_j)
        return torch.sum(0.5 * (1 - Sij) * sig_diff + torch.log(1 + torch.exp(-sig_diff)))

    def evaluate(self, data_fold, print_results=False):
        self.layers.eval()
        with torch.no_grad():
            x = torch.Tensor(data_fold.feature_matrix).to(self.device)
            validation_scores = torch.round(self.layers(x)).squeeze().numpy()
        results = evl.evaluate(data_fold, validation_scores, print_results=print_results)
        return results

    def save(self, path='./rank_net'):
        torch.save(self.state_dict(), path)

class Rank_Net_Sped_Up(Rank_Net):

    def train_sgd_speed4(self, data, lr=1e-3, batch_size=500, num_epochs=1, ndcg_convergence=0.95, eval_freq=0):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
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
                        if completed_batches % eval_freq == 0:
                            print('Average Gradient epoch {}: {} after query {} of {} queries'.format(e, lambda_i, qid,
                                                                                                  num_queries))
                            with torch.no_grad():
                                self.layers.eval()
                                x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                                validation_scores = torch.round(self.layers(x))
                            ndcg_result = evl.ndcg_at_k(validation_scores.numpy(), data.validation.label_vector, 0)
                            print('NDCG score: {}'.format(ndcg_result))
                            if ndcg_result > ndcg_convergence:
                                converged = True
                                print(
                                    'Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence,
                                                                                                       e))
                                break

        if not converged:
            print('Done training for {} epochs'.format(num_epochs))

    def train_sgd_speed5(self, data, lr=5e-5, num_epochs=1, ndcg_convergence=0.95, eval_freq=0):
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
                    if completed_batches % eval_freq == 0:
                        print('Average Gradient epoch {}: {} after query {} of {} queries'.format(e, lambda_i, qid,
                                                                                              num_queries))
                        with torch.no_grad():
                            self.layers.eval()
                            x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                            validation_scores = torch.round(self.layers(x))
                        ndcg_result = evl.ndcg_at_k(validation_scores.numpy(), data.validation.label_vector, 0)
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > ndcg_convergence:
                            converged = True
                            print(
                                'Convergence criteria (NDCG of {}) reached after {} epochs'.format(ndcg_convergence,
                                                                                                   e))
                            break

        if not converged:
            print('Done training for {} epochs'.format(num_epochs))

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

    net = Rank_Net(data.num_features, sigma=1.0)
    start = time()
    net.train_sgd_speed5(data, num_epochs=1)
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

