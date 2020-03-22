# Information Retrieval - Assignment 2
This README details how to run the code for the assignment. 


## Word2Vec
Word2Vec can be trained by running the word2vec.py file. I contains both the word2vec class and the code to run the queries.
Please note that the training is commented in and instead a python file with the trained weights is loaded to run the ranking.
If you want to train the model, uncomment the line and expect training times of 2-3 hours. Getting the most similar words can also be run by uncommenting the respective code line.
## Doc2Vec
Our Doc2Vec class can be found in doc2vec.py. If this file is run as main the ten most similar documents to a randomly sampled document are written to a json file.
Since the parameters for Doc2Vec had to be fine-tuned, the respective script can be found in config_d2v.py. CAUTION: Running this will train the model for all the different parameter options and takes a long time.
The evaluation of the rankings for Doc2Vec can be done with compare_configs.py.
## LSI
LSI implementation for retrieval can be found in lsi.py. Filepaths might be different than used, depending on the system, but it should work in most general cases. Trained models are left out, but results can be found in json files under /results and on the shared google folder as well as other files. Just calling the main of lsi.py trains BoW-LSI and TF-IDF-LSI models with topic numbers 10, 50, 100, 500, 1000 and 2000. These can be changed by adapting the values in topic_list in the main function. Creating an instance of class LSI also trains the model. evaluate.py can be used to evaluate the model. Results of report are generated in plot_results.ipynb 
## LDA
LDA implementation for retrieval can be found in lda.py. Examples of usage are in evaluation.ipynb, as well as general testing and evaluation. Filepaths might be different than used, depending on the system, but it should work in most general cases. Trained models are left out, but results can be found in json files under /results and on the shared google folder as well as other files. Training a new model is easily done by calling the model class first and then run model.train(args). LDA is used with BOW representation as this was said to be allowed in the canvas discussion.
