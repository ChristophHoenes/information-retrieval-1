import pickle
from matplotlib import pyplot as plt
import numpy as np
from itertools import product
import os
from lambda_rank import LambdaRank
import torch
import dataset
import evaluate as evl

def plot_ndcg():
    lrs = [1e-4, 1e-5, 1e-6]
    hidden_layers = [[200, 100], [200,100,50]]
    models = list(product(lrs,hidden_layers))
    print(models)

    with open('ndcgs.pkl', 'rb') as f:
        ndcgs = pickle.load(f)
        print(len(ndcgs))
        plt.figure()
        for i, model_ndcgs in enumerate(ndcgs):
            x = np.arange(len(model_ndcgs))*100
            plt.plot(x, model_ndcgs, label = str(models[i]))
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('NDCG')
        plt.savefig('hp_tuning.png', bbox_inches='tight')

def plot_err():
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    files = os.listdir('.')
    errs = []
    lrs = [1e-4, 1e-5, 1e-6]
    hidden_layers = [[200, 100], [200, 100, 50]]
    models = list(product(lrs, hidden_layers))

    model1 = LambdaRank([200,100], 501, 1e-4, 1)
    model1.model.load_state_dict(torch.load('best_lambda_rank_0.0001[200, 100]501'))
    model1.model.eval()
    results = model1.eval_model()
    errs.append(results['err'][0])

    model2 = LambdaRank([200, 100, 50], 501, 1e-4, 1)
    model2.model.load_state_dict(torch.load('best_lambda_rank_0.0001[200, 100, 50]501'))
    model2.model.eval()
    results = model2.eval_model()
    errs.append(results['err'][0])

    model3 = LambdaRank([200, 100], 501, 1e-5, 1)
    model3.model.load_state_dict(torch.load('best_lambda_rank_1e-05[200, 100]501'))
    model3.model.eval()
    results = model3.eval_model()
    errs.append(results['err'][0])

    model4 = LambdaRank([200, 100, 50], 501, 1e-5, 1)
    model4.model.load_state_dict(torch.load('best_lambda_rank_1e-05[200, 100, 50]501'))
    model4.model.eval()
    results = model4.eval_model()
    errs.append(results['err'][0])

    model5 = LambdaRank([200, 100], 501, 1e-6, 1)
    model5.model.load_state_dict(torch.load('best_lambda_rank_1e-06[200, 100]501'))
    model5.model.eval()
    results = model5.eval_model()
    errs.append(results['err'][0])

    model6 = LambdaRank([200, 100, 50], 501, 1e-6, 1)
    model6.model.load_state_dict(torch.load('best_lambda_rank_1e-06[200, 100, 50]501'))
    model6.model.eval()
    results = model6.eval_model()
    errs.append(results['err'][0])
    print(errs)
    plt.figure()
    x = np.arange(6)
    plt.bar(x, errs)
    plt.xticks(x, models, rotation=45)
    plt.ylim(top=1)
    plt.xlabel('model')
    plt.ylabel('ERR')
    plt.savefig('hp_tuning_err.png', bbox_inches='tight')

def eval_sigma():
    sigmas = [0.5, 1, 2]
    with open('ndcgs_sigma.pkl', 'rb') as f:
        ndcgs = pickle.load(f)
    for i, model_ndcgs in enumerate(ndcgs):
        x = np.arange(len(model_ndcgs)) * 100
        plt.plot(x, model_ndcgs, label=str(sigmas[i]))
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('NDCG')
    plt.savefig('hp_tuning_sigma.png', bbox_inches='tight')
    plt.figure()

def eval_single_model():
    with open('best_results.pkl', 'rb') as f:
        results = pickle.load(f)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('NDCG')
    x = np.arange(len(results['ndcg'])) * 100
    ax1.plot(x, results['ndcg'], label='NDCG')
    ax2 = ax1.twinx()
    ax2.set_ylabel('ERR')  # we already handled the x-label with ax1
    ax2.plot(x, results['err'],label='ERR', color='red')
    ax1.legend(loc=4)
    ax2.legend(loc=0)
    fig.tight_layout()
    #fig.show()
    plt.savefig('best_model_plot.png', bbox_inches='tight')





if __name__ == '__main__':
    eval_single_model()
