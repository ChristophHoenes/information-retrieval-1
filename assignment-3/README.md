## Assignment 3 ##
This is the readme for the third homework assignment.
### Pointwise ###
To train a pointwise model, run `train_pointwise.py`, e.g.:
<pre><code>python train_pointwise.py --learning_rate=5e-4 --n_hiddens=200,200 --max_steps=3000 --eval_freq=30 --batch_size=1000</code></pre>
Results are written to folder `results`. The results include losses over training, for both training and validation, and NDCG over training, and final test performance of the model.

### RankNet ###
To train the original RankNet and the sped-up version with the hyperparameters from the report simply run rank_net.py.
To change some parameters check the parameter list of the class and change the desired value in main.
If hyperparameter tuning should also be done, comment in the call for `hyperparameter_search()`

### LambdaRank ###
To start the fine-tuned model, simply run `lambda_net.py` as the main function.
If hyperparameter tuning should also be done, comment in the call for `hyperparameter_search()`
in the main function. `plot_lambdarank.py` was used to generate the plots in the report.
If you want to change the hyperparameters of the model, alter the respective variables at the beginning of the main.
