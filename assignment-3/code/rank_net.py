import dataset
import evaluate as evl

import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from time import time
from itertools import product

import pickle


class Rank_Net(nn.Module):
    def __init__(self, d_in, num_neurons=[200, 100], sigma=1.0, dropout=0.0, device='cpu', model_id=None):
        assert isinstance(num_neurons, list) or isinstance(num_neurons, int), "num_neurons must be either an int (one layer) or a list of ints"
        if isinstance(num_neurons, int):
            num_neurons = [num_neurons]

        super(Rank_Net, self).__init__()
        self.d_in = d_in
        self.num_neurons = num_neurons.copy()
        self.num_neurons.insert(0, self.d_in)
        self.num_neurons.append(1)
        self.sigma = sigma
        self.dropout = dropout
        self.device = device
        if model_id is None:
            self.model_id = 'd_in_{}_num_ly_{}_sig_{}_{}'.format(d_in,len(self.num_neurons)-2,sigma,np.random.randint(1000000, 9999999))
        layers = []
        for h, h_next in zip(self.num_neurons, self.num_neurons[1:]):
            layers.append(nn.Linear(h, h_next))
            #layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        self.layers.to(self.device)
        return self.layers(x)

    def train_bgd(self, data, lr=5e-4, batch_size=500, num_epochs=1, eval_freq=1000):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        validation_results = []
        losses = []
        arrs = []
        best_ndcg = 0
        no_improvement = 0
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            random_query_order = np.arange(num_queries)
            np.random.shuffle(random_query_order)
            all_loss = 0
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

                loss = self.rank_net_loss(score_combs_i, score_combs_j,
                                                            label_combs_i, label_combs_j)

                all_loss += loss
                losses.append(loss.item()/(batch_size if qid % batch_size == 0 else num_queries % batch_size))

                if qid % batch_size == 0 or qid == num_queries-1:
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
                        val_result = evl.evaluate(data.validation, validation_scores)
                        arrs.append(val_result['arr'][0])
                        ndcg_result = val_result['ndcg'][0]
                        validation_results.append(ndcg_result)
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > best_ndcg:
                            best_ndcg = ndcg_result
                            no_improvement = 0
                        else:
                            no_improvement += 1
                        if no_improvement >= 8:
                            converged = True
                            print(
                                'Convergence criteria (NDCG of {}) reached after {} queries of epoch'.format(best_ndcg,
                                                                                                    qid + 1, e))
                            break

        print('Done training for {} epochs'.format(num_epochs))
        with open('valid_results_lr_' + str(lr)+'_'+self.model_id, 'wb') as f:
            pickle.dump(validation_results, f)
        with open('loss_results_lr_' + str(lr)+'_'+self.model_id, 'wb') as f:
             pickle.dump(losses, f)
        with open('arr_results_lr_' + str(lr)+'_'+self.model_id, 'wb') as f:
            pickle.dump(arrs, f)
        return self.layers, self.evaluate(data.validation)['ndcg']

    def train_sgd(self, data, lr=1e-5, num_epochs=1, eval_freq=1000):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        validation_results = []
        losses = []
        best_ndcg = 0
        no_improvement = 0
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

                loss = self.rank_net_loss(score_combs_i, score_combs_j, label_combs_i, label_combs_j)

                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if eval_freq != 0:
                    if qid % eval_freq == 0:
                        print('Loss epoch {}: {} after query {} of {} queries'.format(e, loss.item(), qid, num_queries))
                        with torch.no_grad():
                            self.layers.eval()
                            x = torch.Tensor(data.validation.feature_matrix).to(self.device)
                            validation_scores = self.layers(x).squeeze().numpy()
                        ndcg_result = evl.evaluate(data.validation, validation_scores)['ndcg'][0]
                        validation_results.append(ndcg_result)
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > best_ndcg:
                            best_ndcg = ndcg_result
                            no_improvement = 0
                        else:
                            no_improvement += 1
                        if no_improvement >= 8:
                            converged = True
                            print(
                                'Convergence criteria (NDCG of {}) reached after {} queries'.format(best_ndcg, qid + 1))
                            break

        print('Done training for {} epochs'.format(num_epochs))
        with open('valid_results_lr_'+ str(lr) +'_'+ self.model_id, 'wb') as f:
            pickle.dump(validation_results, f)
        with open('loss_results_lr_' + str(lr)+'_'+self.model_id, 'wb') as f:
            pickle.dump(losses, f)
        return self.layers, self.evaluate(data.validation)['ndcg']

    def rank_net_loss(self, s_i, s_j, S_i, S_j):
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

    def save(self, path='./rank_net_'):
        torch.save(self.state_dict(), path+self.model_id+'.weights')

class Rank_Net_Sped_Up(Rank_Net):

    def __init__(self, d_in, num_neurons=[200, 100], sigma=1.0, dropout=0.0, device='cpu', model_id=None):
        super(Rank_Net_Sped_Up, self).__init__(d_in, num_neurons, sigma, dropout, device, model_id)
        self.model_id = 'sped_up_'+self.model_id

    def train_bgd(self, data, lr=1e-3, batch_size=500, num_epochs=1, eval_freq=1000):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        validation_results = []
        losses = []
        arrs = []
        best_ndcg = 0
        no_improvement = 0
        converged = False
        for e in range(num_epochs):
            if converged:
                break
            random_query_order = np.arange(num_queries)
            np.random.shuffle(random_query_order)
            skipped = False
            for qid in tqdm(range(num_queries)):
                x = torch.Tensor(data.train.query_feat(random_query_order[qid])).to(self.device)
                query_scores = self.layers(x)
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
                    optimizer.zero_grad()
                    batch_scores.backward(batch_lambdas)
                    losses.append(batch_lambdas.mean().item())
                    optimizer.step()

                if eval_freq != 0:
                    if qid % eval_freq == 0:
                        val_result = self.evaluate(data.validation)
                        arrs.append(val_result['arr'][0])
                        ndcg_result = val_result['ndcg'][0]
                        validation_results.append(ndcg_result)
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > best_ndcg:
                            best_ndcg = ndcg_result
                            no_improvement = 0
                        else:
                            no_improvement += 1
                        if no_improvement >= 8:
                            converged = True
                            print(
                                'Convergence criteria (NDCG of {}) reached after {} queries'.format(best_ndcg,qid+1))
                            break

        print('Done training for {} epochs'.format(num_epochs))
        with open('valid_results_lr_'+ str(lr) +'_'+ self.model_id, 'wb') as f:
            pickle.dump(validation_results, f)
        with open('loss_results_lr_' + str(lr)+'_'+self.model_id, 'wb') as f:
             pickle.dump(losses, f)
        with open('arr_results_lr_' + str(lr)+'_'+self.model_id, 'wb') as f:
            pickle.dump(arrs, f)
        return self.layers, self.evaluate(data.validation)['ndcg']

    def train_sgd(self, data, lr=5e-5, num_epochs=1, eval_freq=1000):
        self.layers.train()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        num_queries = data.train.num_queries()
        validation_results = []
        losses = []
        best_ndcg = 0
        no_improvement = 0
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
                losses.append(lambda_i.mean().item())
                optimizer.step()


                if eval_freq != 0:
                    if qid % eval_freq == 0:
                        ndcg_result = self.evaluate(data.validation)['ndcg'][0]
                        validation_results.append(ndcg_result)
                        print('NDCG score: {}'.format(ndcg_result))
                        if ndcg_result > best_ndcg:
                            best_ndcg = ndcg_result
                            no_improvement = 0
                        else:
                            no_improvement += 1
                        if no_improvement >= 8:
                            converged = True
                            print(
                                'Convergence criteria (NDCG of {}) reached after {} queries'.format(best_ndcg, qid + 1))
                            break

        print('Done training for {} epochs'.format(num_epochs))
        with open('valid_results_lr_'+ str(lr) +'_'+ self.model_id, 'wb') as f:
            pickle.dump(validation_results, f)
        with open('loss_results_lr_' + str(lr)+'_'+self.model_id, 'wb') as f:
            pickle.dump(losses, f)
        return self.layers, self.evaluate(data.validation)['ndcg']

    def rank_net_loss(self, s_i, s_j, S_i, S_j):
        S_diff = S_i - S_j
        Sij = np.where(S_diff > 0, 1.0, -1.0)
        Sij = torch.from_numpy(np.where(S_diff == 0, 0.0, Sij))
        sig_diff = self.sigma * (s_i - s_j)
        num_docs = int(math.sqrt(s_i.shape[0]))
        lambda_ij = (self.sigma * (0.5 * (1 - Sij) - 1/(1 + torch.exp(sig_diff)))).view(num_docs,num_docs)
        return torch.sum(lambda_ij, dim=1).unsqueeze_(dim=1)

def hyperparameter_search():
    lrs = [2e-3, 1e-3, 5e-4]
    hidden_layers = [[200], [200, 100], [200,100,50]]
    irms = ['ndcg']
    sigmas = [1.0]

    best_ndcg = [0]
    best_model = None
    best_ndcg2 = [0]
    best_model2 = None
    for irm in irms:
        print(irm)
        for lr in lrs:
            for hidden_layer in hidden_layers:
                for sigma in sigmas:
                    model = Rank_Net(data.num_features,hidden_layer,sigma=sigma)
                    model2 = Rank_Net_Sped_Up(data.num_features,hidden_layer,sigma=sigma)
                    model, ndcg = model.train_bgd(data, lr=lr, num_epochs=3, eval_freq=100)
                    model2, ndcg2 = model2.train_bgd(data, lr=lr, num_epochs=3, eval_freq=100)
                    print(ndcg)
                    print(ndcg2)
                    if ndcg[0] > best_ndcg[0]:
                        best_ndcg = ndcg
                        best_model = model
                    print(ndcg2)
                    if ndcg2[0] > best_ndcg2[0]:
                        best_ndcg2 = ndcg2
                        best_model2 = model2
    print(best_ndcg)
    torch.save(best_model, './hp_search_best')
    print(best_ndcg2)
    torch.save(best_model2, './hp_search_best2')


if __name__ == "__main__":
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    #hyperparameter_search()
    net = Rank_Net(data.num_features, num_neurons=[200], sigma=1.0)
    start = time()
    net.train_bgd(data, lr=5e-4, eval_freq=50)
    end = time()
    print('Finished training in {} minutes'.format((end - start) / 60))
    net.save(path='./rank_net' + str(net.model_id) + '.weights')
    final_test_results = net.evaluate(data.test, print_results=True)
    with open('eval' + str(net.model_id), 'wb') as f:
        pickle.dump(final_test_results, f)

    net = Rank_Net_Sped_Up(data.num_features, num_neurons=[200], sigma=1.0)
    start = time()
    net.train_bgd(data, lr=1e-3, eval_freq=50)
    end = time()
    print('Finished training in {} minutes'.format((end-start)/60))
    net.save(path='./rank_net'+str(net.model_id)+'.weights')
    final_test_results = net.evaluate(data.test,print_results=True)
    with open('eval'+str(net.model_id), 'wb') as f:
        pickle.dump(final_test_results, f)

