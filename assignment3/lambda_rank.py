import dataset
import ranking as rnk
import tqdm
import evaluate as evl
import numpy as np
import torch

from collections import OrderedDict
from torch import nn


def delta_irm(irm, labels, scores, i, j, query_dcg=None):
    sorted_idx = (-scores).argsort()
    sorted_labels = labels[sorted_idx]
    #sorted_scores = scores[sorted_idx]
    new_idx_i = np.argwhere(sorted_idx == i)[0][0]
    new_idx_j = np.argwhere(sorted_idx == j)[0][0]
    before = irm(sorted_labels, 0, query_dcg)
    sorted_labels[new_idx_i], sorted_labels[new_idx_j] = sorted_labels[new_idx_j], sorted_labels[new_idx_i]
    after = irm(sorted_labels, 0, query_dcg)

    return abs(before - after)



class LambdaRank:
    def __init__(self, hidden_dims, input_size, lr, irm):
        self.name = str(lr) + str(hidden_dims) + str(input_size) + str(irm)
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
        #self.device= 'cpu'

    def train(self, data, irm, lr):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = 'cpu'
        print(device)
        self.model = self.model.to(device)
        #lr = 1e-5
        num_epochs = 1
        sigma = 1
        evaluate_every = 100
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # variables for early stopping
        previous_value = 0
        max_iterations = 500
        best_model = None
        best_model_irm = 0
        since_last_improvement = 0
        stopped = False

        for n in range(num_epochs):
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

                if idx % evaluate_every == 0:
                    result = self.eval_model()
                    ndcg = result['ndcg'][0]
                    print(ndcg, best_model_irm)
                    # early stopping
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
                    #previous_value = ndcg
            self.eval_model()
            if stopped:
                break
        torch.save(best_model, './best_lambda_rank_'+self.name)
        return best_model, best_model_irm

    def eval_model(self):
        with torch.no_grad():
            features = data.validation.feature_matrix
            features = torch.tensor(features).float().to(self.device)
            validation_s = self.model(features)
            results = evl.evaluate(data.validation, validation_s.cpu().numpy().squeeze(), print_results=True)
            return results


def hyperparameter_search():
    lrs = [1e-4, 1e-5, 1e-6]
    hidden_layers = [[200, 100], [200,100,50]]
    irms = [evl.ndcg_speed]
    best_ndcg = 0
    best_model = None
    for irm in irms:
        print(irm)
        for lr in lrs:
            for hidden_layer in hidden_layers:
                model = LambdaRank(hidden_layer, data.num_features, lr, irm)
                model, ndcg = model.train(data, irm, lr)
                print(ndcg)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_model = model
    print(best_ndcg)
    torch.save(best_model, './hp_search_best')


if __name__ == '__main__':
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    best_model = hyperparameter_search()
