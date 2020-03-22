
import pdb
import argparse
import numpy as np
import os
from pointwise import MLP
import os.path
import ranking as rnk
import evaluate as evl
import pickle as pkl

import matplotlib.pyplot as plt

import dataset
from dataset import DataClass

import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-3
N_HIDDENS_DEFAULT     = "50"
MAX_STEPS_DEFAULT     = 3000
BATCH_SIZE_DEFAULT    = 500
EVAL_FREQ_DEFAULT     = 100

CONVERGENCE = 5

FLAGS = None

def save_results(results):
    
    if not os.path.exists("results"):
        os.mkdir("results")
    hiddens = "-".join(FLAGS.n_hiddens.split(","))
    
    plt.plot(results["train"]["loss"], label="training loss")
    plt.plot(np.arange(len(results["validation"]["loss"]))*FLAGS.eval_freq, 
             results["validation"]["loss"], label="validation loss")
    plt.ylabel("Loss", fontsize=12)
    plt.xlabel("Step", fontsize=12)
    p = plt.plot(0)
    plt.twinx()
    plt.plot(np.arange(len(results["validation"]["ndcg"]))*FLAGS.eval_freq, 
             results["validation"]["ndcg"], color=p[0].get_color(), label="validation nDCG")
    plt.ylabel("nDCG", fontsize=12)
    plt.title(f"lr={FLAGS.learning_rate}, n_hiddens={FLAGS.n_hiddens}")
    plt.savefig(f"results/lr{FLAGS.learning_rate}_nhiddens{hiddens}.pdf")
    plt.close()
    
    with open(f"results/lr{FLAGS.learning_rate}_nhiddens{hiddens}.pkl", "wb") as f:
        pkl.dump(results, f)

def train():
    
#    np.random.seed(42)

    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.n_hiddens:
        n_hiddens = FLAGS.n_hiddens.split(",")
        n_hiddens = [int(n_hidden) for n_hidden in n_hiddens]
    else:
        n_hiddens = []

    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    max_steps = FLAGS.max_steps
    eval_freq = FLAGS.eval_freq

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print("Reading data...")
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    print("Done!")
    
    print("Preparing training...")
    #train set for batched training
    train  = DataClass(data.train.feature_matrix, data.train.label_vector, shuffle=True)
    
    #entire train set
    xtrain = torch.from_numpy(data.train.feature_matrix).float().to(device)
    ttrain = torch.from_numpy(data.train.label_vector).float().view(-1,1).to(device)
    
    #entire validation set
    xval = torch.from_numpy(data.validation.feature_matrix).float().to(device)
    tval = torch.from_numpy(data.validation.label_vector).float().view(-1,1).to(device)
    
    #entire test set
    xtest = torch.from_numpy(data.test.feature_matrix).float().to(device)
    ttest = torch.from_numpy(data.test.label_vector).float().view(-1,1).to(device)
  
    n_inputs = data.train.feature_matrix.shape[1]
    model = MLP(n_inputs, n_hiddens).to(device)
  
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    results = {"train":{"loss":[]}, "validation":{"loss":[],"ndcg":[]}, "test":{},
               "learning_rate":learning_rate, "n_hiddens":FLAGS.n_hiddens,
               "batch_size":batch_size, "eval_freq":eval_freq}
    print("Done!")
    
    print("Training...")
    for step in range(max_steps):
    
        optimizer.zero_grad()
    
        x, t = train.next_batch(batch_size)
        x = x.requires_grad_(True).to(device)
        t = t.to(device)
        
        y = model(x).to(device)
        loss = criterion(y, t)
    
        loss.backward() 
        optimizer.step()
        
        results["train"]["loss"].append(loss.detach())
        
        if step % eval_freq == 0 or step == max_steps-1:
            with torch.no_grad():
                yval = model(xval)
                val_results = evl.evaluate(data.validation, np.array(yval).squeeze(), print_results=False)
                val_ndcg, val_ndcg_std = val_results["ndcg"]
                lossval = criterion(yval.to(device), tval)
                results["validation"]["loss"].append(lossval)
                results["validation"]["ndcg"].append(val_ndcg) 
            if step == max_steps-1:
                print(f"[{train.epochs_completed}] {step+1}/{max_steps} | nDCG: {round(val_ndcg,3)}")
            else:
                print(f"[{train.epochs_completed}] {step}/{max_steps} | nDCG: {round(val_ndcg,3)}")
            #early stopping
            if len(results["validation"]["ndcg"]) > CONVERGENCE*2:
                prev = results["validation"]["ndcg"]
                if np.round(np.mean(prev[-2*CONVERGENCE:-CONVERGENCE]),3)==np.round(np.mean(prev[-CONVERGENCE:]),3):
                    print(f"Changes very small over previous {CONVERGENCE*2} iterations.")
                    break
    
    with torch.no_grad():
        ytest = model(xtest)
        yval = model(xval)
        test_results = evl.evaluate(data.test, np.array(ytest).squeeze(), print_results=False)
        test_ndcg, test_ndcg_std = test_results["ndcg"]
        results["test"]["ndcg"] = test_ndcg
        results["test"]["ndcg_std"] = test_ndcg_std
        results["test"]["scores"] = ytest
        results["validation"]["scores"] = yval
    print(f"Test nDCG: {round(test_ndcg,3)} +/- {round(test_ndcg_std,3)}")
    save_results(results)
    print("Done!")

    
def print_flags():
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():    
    print_flags()
    train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--n_hiddens', type = str, default = N_HIDDENS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()

    main()
    

    
    
