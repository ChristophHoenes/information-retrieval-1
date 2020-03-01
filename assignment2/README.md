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
