import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch
from dataset import Dataclass
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
    def __init__(self, num_layers, hidden_dims):
        layers = OrderedDict()
        for i in range(num_layers):
            layers.update(('layer'+str(i), nn.Linear(len(input), hidden_dims[i])))
            layers.update(('relu'+str(i), nn.ReLU()))
        self.model = nn.Sequential(layers)

    def train(self):
        lr = 0.001
        data = dataset.get_dataset().get_data_folds()[0]
        data.read_data()
        print(data)




if __name__ == '__main__':
    model = LambdaRank(3, 100)
    model.train()
