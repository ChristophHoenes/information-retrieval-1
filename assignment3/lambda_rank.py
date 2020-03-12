import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch

from collections import OrderedDict
from torch import nn

def err(labels):
    p = 1
    err = 0
    g_max = max(labels)
    for r in range(len(labels)):
        g = labels[r]
        R = (2**g-1)/(2**g_max)
        err += p * R / r
        p *= 1 - R
    return err

def delta_err(ranking, i, j):
    labels_i = ranking
    tmp = ranking[i]
    ranking[i] = ranking[j]
    ranking[j] = tmp
    labels_j = ranking
    err1 = err(labels_i)
    err2 = err(labels_j)
    derr = abs(err1-err2)
    return derr

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
        self.model = nn.Sequential(layers)

    def train(self, data):
        lr = 0.001
        num_epochs = 10
        batch_size = 100
        dataclass = dataset.DataClass(data.train.feature_matrix, data.train.label_vector)
        print(data.train.feature_matrix.shape)
        for i in range(num_epochs):
            input = dataclass.next_batch(100)
            self.model(input)




if __name__ == '__main__':
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    model = LambdaRank(3, [100, 200], input_size=data.num_features)
    model.train(data)
