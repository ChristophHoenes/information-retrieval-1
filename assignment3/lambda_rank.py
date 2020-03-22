import dataset
import ranking as rnk
import tqdm
import evaluate as evl
import numpy as np
import torch
import pickle

from collections import OrderedDict
from torch import nn


def delta_irm(irm, labels, scores, i, j, query_dcg=None):
    # sort labels by ranking score
    sorted_idx = (-scores).argsort()
    sorted_labels = labels[sorted_idx]
    # keep track of where i and j where sorted to
    new_idx_i = np.argwhere(sorted_idx == i)[0][0]
    new_idx_j = np.argwhere(sorted_idx == j)[0][0]
    # compute irm once before and once after swapping
    before = irm(sorted_labels, 0, query_dcg)
    sorted_labels[new_idx_i], sorted_labels[new_idx_j] = sorted_labels[new_idx_j], sorted_labels[new_idx_i]
    after = irm(sorted_labels, 0, query_dcg)

    return abs(before - after)



class LambdaRank:
    # Init function. Initializes the layers of the model
    def __init__(self, hidden_dims, input_size, lr, sigma):
        self.name = str(lr) + str(hidden_dims) + str(input_size)+ str(sigma)
        layers = OrderedDict()
        layers['layer0'] = nn.Linear(input_size, hidden_dims[0])
        layers['relu0'] = nn.ReLU()
        in_dim = hidden_dims[0]
        for i in range(1, len(hidden_dims)):
            layers['layer' + str(i)] = nn.Linear(in_dim, hidden_dims[i])
            layers['relu'+str(i)] = nn.ReLU()
            in_dim = hidden_dims[i]
        layers['layer'+str(len(hidden_dims))] = nn.Linear(in_dim, 1)
        layers['relu'+str(len(hidden_dims))] = nn.ReLU()
        self.model = nn.Sequential(layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, data, irm=evl.ndcg_speed, lr=1e-4, sigma=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        # more epochs take soooo long
        num_epochs = 1
        evaluate_every = 100
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # variables for early stopping
        max_iterations = 500 # maximum number of iterations the model's performance is allowed to not increase
        best_model = None
        best_model_irm = 0
        since_last_improvement = 0 # how many iterations have passed since last increase in performance
        stopped = False
        # variables to track ndcg and err development over training
        config_ndcgs = []
        errs = []

        for n in range(num_epochs):
            # goes sequentially through dataset, not batched
            for idx in tqdm.tqdm(range(data.train.num_queries())):
                self.model.zero_grad()
                # get feature vectors and labels for each query
                features = data.train.query_feat(idx)
                labels_numpy = data.train.query_labels(idx)
                features = torch.tensor(features).float().to(device)
                labels = torch.tensor(labels_numpy).float().to(device)
                # compute scores
                s = self.model(features)
                # make matrix of scores with all combinations with itself
                scores_repeated = s.repeat(1, s.shape[0])
                scores_repeated_t = s.t().repeat(s.shape[0], 1)
                # compute all differences s_i-s_j
                difference = scores_repeated - scores_repeated_t
                # do the same for the labels
                labels_repeated = labels.unsqueeze(1).repeat(1, s.shape[0])
                labels_repeated_t = labels.unsqueeze(0).repeat(s.shape, 1)
                labels_difference = labels_repeated - labels_repeated_t
                ones = torch.ones_like(labels_difference)
                zeros = torch.zeros_like(labels_difference)
                # use torch.where to get a matrix of S_ij
                s_ij = torch.where(labels_difference > 0, ones, -ones)
                s_ij = torch.where(labels_difference == 0, zeros, s_ij)
                # plug everything into the formula to compute the lambdas
                lambdas = sigma*(0.5*(torch.ones_like(s_ij)-s_ij)-(1 / (1 + torch.exp(sigma * difference))))
                deltas = torch.zeros_like(lambdas, requires_grad=False)
                # calculate the perfect dcg of the query here once to save time
                sorted_labels, _ = labels.sort(descending=True)
                query_dcg = evl.dcg_at_k(sorted_labels.cpu().numpy(), 0)
                s_detached = s.detach().squeeze().cpu().numpy()
                # calculate the change in ndcg/err for each combination i,j
                for i in range(lambdas.shape[0]):
                    for j in range(i + 1, lambdas.shape[1]):
                        deltas[i, j] = delta_irm(irm, labels_numpy, s_detached, i, j, query_dcg)
                        deltas[j, i] = deltas[i, j]
                # scale gradients by deltas
                lambdas = lambdas * deltas
                lambdas = lambdas.sum(1)
                # update weights
                s.backward(lambdas.unsqueeze(1))
                optimizer.step()

                # check the model's performance on validation set
                if idx % evaluate_every == 0:
                    result = self.eval_model()
                    ndcg = result['ndcg'][0]
                    err = result['err'][0]
                    config_ndcgs.append(ndcg)
                    errs.append(err)

                    # check for early stopping
                    if ndcg < best_model_irm:
                        since_last_improvement += evaluate_every
                        if since_last_improvement > max_iterations:
                            stopped = True
                            print("Reached Convergence!!!!!!!")
                            break
                    else:
                        since_last_improvement = 0
                        best_model = self.model.state_dict()
                        best_model_irm = ndcg
            if stopped:
                break
        # after training, eval onece more on validation AND on test set
        self.eval_model()
        torch.save(best_model, './best_lambda_rank_'+self.name)
        # final test set evaluation
        with torch.no_grad():
            features = data.test.feature_matrix
            features = torch.tensor(features).float().to(self.device)
            test_s = self.model(features)
        results = evl.evaluate(data.test, test_s.cpu().numpy().squeeze(), print_results=True)
        return best_model, best_model_irm, config_ndcgs, errs

    def eval_model(self):
        with torch.no_grad():
            features = data.validation.feature_matrix
            features = torch.tensor(features).float().to(self.device)
            validation_s = self.model(features)
            results = evl.evaluate(data.validation, validation_s.cpu().numpy().squeeze())
            return results


def hyperparameter_search():
    lrs = [1e-4, 1e-5, 1e-6]
    hidden_layers = [[200, 100], [200,100,50]]
    irm = evl.ndcg_speed
    best_ndcg = 0
    best_model = None
    ndcgs = []

    for lr in lrs:
        for hidden_layer in hidden_layers:
            model = LambdaRank(hidden_layer, data.num_features, lr)
            model, model_best_ndcg, model_ndcgs, model_best_err = model.train(data, irm, lr)
            ndcgs.append(model_ndcgs)
            print(model_best_ndcg)
            if model_best_ndcg > best_ndcg:
                best_ndcg = model_best_ndcg
                best_model = model
    print('BEST MODEL NDCG: ', best_ndcg)
    # save the training history and model
    with open('ndcgs.pkl', 'wb') as f:
        pickle.dump(ndcgs, f)
    torch.save(best_model, './hp_search_best')





if __name__ == '__main__':
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    #best_model = hyperparameter_search()

    best_lr = 1e-4
    best_hidden = [200, 100]
    best_sigma = 1
    irm = evl.ndcg_speed
    model = LambdaRank(best_hidden, data.num_features, best_lr, best_sigma)
    model, model_best_ndcg, model_ndcgs, model_errs = model.train(data, irm, best_lr, best_sigma)
    best_results = {'ndcg': model_ndcgs, 'err': model_errs}
    with open('best_results.pkl', 'wb') as f:
        pickle.dump(best_results, f)
    torch.save(model, './best_final')
