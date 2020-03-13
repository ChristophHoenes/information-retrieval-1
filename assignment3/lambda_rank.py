import dataset
import ranking as rnk
import tqdm
import evaluate as evl
import numpy as np
import torch

from collections import OrderedDict
from torch import nn

def err(labels, k=0):
    p = 1
    err = 0
    g_max = max(labels)
    for r in range(len(labels)):
        g = labels[r]
        R = (2**g-1)/(2**g_max)
        err += p * R / r
        p *= 1 - R
    return err

def delta_irm(ranking, i, irm):
    deltas = torch.zeros(ranking.shape[0])
    for j in range(ranking.shape[0]):
        labels_i = ranking
        tmp = ranking[i]
        ranking[i] = ranking[j]
        ranking[j] = tmp
        labels_j = ranking
        score1 = irm(labels_i, 0)
        score2 = irm(labels_j, 0)
        derr = abs(score1 - score2)
        deltas[j] = derr
    return deltas

class LambdaRank:
    def __init__(self, num_layers, hidden_dims, input_size):
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

    def train(self, data, irm):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        lr = 0.001
        num_epochs = 1
        sigma = 1
        evaluate_every = 10
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for n in range(num_epochs):
            for idx in tqdm.tqdm(range(data.train.num_queries())):
                self.model.zero_grad()
                # get feature vectors and labels for each query
                s_i, e_i = data.train.query_range(idx+1)
                input = torch.tensor(data.train.feature_matrix[s_i:e_i]).float().to(device)
                labels = torch.tensor(data.train.label_vector[s_i:e_i]).float().to(device)
                s = self.model(input).squeeze()
                lambdas = torch.zeros((input.shape[0], input.shape[0]))
                for i in range(input.shape[0]):
                    label_i = labels[i]
                    lambdas[i] = torch.where(labels > label_i, -torch.sigmoid(s[i]-s), torch.ones_like(s) - torch.sigmoid(s[i] - s))
                    lambdas[i, i] = torch.tensor(1/2) - torch.sigmoid(s[i]-s[i])
                    """
                        if i == j:
                            lambda_ij = torch.sigmoid(1/2 - (1 / (1 + torch.exp(torch.sigmoid(s[i] - s[j])))))
                        elif data.train.label_vector[i] > data.train.label_vector[j]:
                            lambda_ij = torch.sigmoid(-1/(1+torch.exp(torch.sigmoid(s[i]-s[j]))))
                        else:
                            lambda_ij = torch.sigmoid(1 - (1 / (1 + torch.exp(torch.sigmoid(s[i] - s[j])))))
                        """

                    deltas = delta_irm(labels, i, irm)
                    lambdas[i] = lambdas[i] * deltas
                lambdas_i = lambdas.sum(dim=1)
                #print(lambdas_i.shape)
                s.backward(lambdas_i)
                optimizer.step()
                if idx % evaluate_every == 0:
                    with torch.no_grad():
                        validation_scores = self.model(torch.tensor(data.validation.feature_matrix).float())
                        results = evl.evaluate(data.validation, validation_scores.squeeze().numpy(), print_results=True)
                        #print(results)
                        #query_ranking, query_inverted_ranking = rnk.rank_and_invert(s)



if __name__ == '__main__':
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    model = LambdaRank(3, [100, 200], input_size=data.num_features)
    model.train(data, irm=err)
