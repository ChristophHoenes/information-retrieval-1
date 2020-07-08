import numpy as np

def bow2tfidf(bow_vec, d):
    result = []
    bow_vec = [(id, np.log(1 + tf)) for (id, tf) in bow_vec]
    for (id, tf) in bow_vec:
        result.append((id, tf / d.dfs[id]))
    return result
