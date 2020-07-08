import os 
import pickle as pkl
import numpy as np
from gensim.models import LdaModel
from gensim.models import LdaMulticore
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import read_ap

def kl_divergence(p, q):
  p_ = p[p!=0]
  # +1e-6 in denom for stability
  return np.sum(p_ * np.log(p_ / (q[p!=0]+1e-6)))
    
    
class LDARetrieval():

    def __init__(self, docs, get_model=False, num_topics=10, passes=6, iterations=40, prep_search=False):
        
      fDICT = "./models/lda_dict.dat"

      fCORPUS = "./models/lda_corpus.dat"
      if os.path.exists(fDICT) and os.path.exists(fCORPUS):
          print("Loading corpus from disk...")
          with open(fDICT, "rb") as fp:
              self.dictionary = pkl.load(fp)
          with open(fCORPUS, "rb") as fp:
              self.corpus = pkl.load(fp)
      else:
          print("Processing documents...")
          doclist = [docs[doc] for doc in docs]  
          self.dictionary = Dictionary(doclist)
          self.dictionary.filter_extremes(no_below=400, no_above=0.333)
          self.corpus = [self.dictionary.doc2bow(doc) for doc in doclist]
          with open(fDICT, "wb") as fp:
              pkl.dump(self.dictionary, fp)
          with open(fCORPUS, "wb") as fp:
              pkl.dump(self.corpus, fp)
      if get_model:
          self.get_model(num_topics=num_topics, passes=passes, iterations=iterations, prep_search=prep_search, docs=docs)
    
    def train(self, num_topics, chunksize=10000, passes=6, iterations=40, eval_every=40):
      fmodel = f"./models/lda_{num_topics}top_{iterations}iter_{passes}pass"
#       logging.basicConfig(filename=fmodel + ".log",
#                     format="%(asctime)s:%(levelname)s:%(message)s",
#                     level=logging.INFO)
      
      temp = self.dictionary[0] 
      id2word = self.dictionary.id2token 
      model = LdaMulticore( corpus=self.corpus,
                            id2word=id2word,
                            chunksize=chunksize,
                            iterations=iterations,
                            num_topics=num_topics,
                            passes=passes,
                            eval_every=eval_every)
      model.save(fmodel + ".pt")
      self.model = model

#       p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
#       matches = [p.findall(l) for l in open(fmodel+'.log')]
#       matches = [m for m in matches if len(m) > 0]
#       tuples = [t[0] for t in matches]
#       perplexity = [float(t[1]) for t in tuples]
#       liklihood = [float(t[0]) for t in tuples]
#       iter = list(range(0,len(tuples)*10,10))
#       plt.plot(iter,liklihood,c="black")
#       plt.ylabel("log liklihood")
#       plt.xlabel("iteration")
#       plt.title("Topic Model Convergence")
#       plt.grid()
#       plt.savefig(fmodel + ".pdf") 
      
      return model

    def prepare_search(self, docs):
      fdocsearch = f"./models/docs_{self.model.num_topics}search.dat"
    
      if os.path.exists(fdocsearch):
        print("Loading docs for search from disk...")
        with open(fdocsearch, "rb") as fp:
          self.docvecs = pkl.load(fp)
      else:
        print("Preparing docs for search...")
        self.docvecs = {}
        for doc in docs:
          docvec = np.zeros(self.model.num_topics)
          doc_repr = self.dictionary.doc2bow(docs[doc])
          for i, frac in self.model[doc_repr]:
            docvec[i] = frac
          self.docvecs[doc] = docvec
        with open(fdocsearch, "wb") as fp:
          pkl.dump(self.docvecs, fp)

    def get_model(self, num_topics, passes=6, iterations=40, prep_search=False, docs=None):
      fname = f"./models/lda_{num_topics}top_{iterations}iter_{passes}pass"
      if not os.path.exists(fname + ".pt"):
        print("Model not found...")
        return None
      self.model = LdaModel.load(fname + ".pt")
      if prep_search:
        self.prepare_search(docs)
      return self.model

    def search(self, query):
        query_repr = self.dictionary.doc2bow(read_ap.process_text(query))
        qvec = np.zeros(self.model.num_topics)
        for i, frac in self.model[query_repr]:
          qvec[i] = frac

        results = {}
        for doc in self.docvecs:
          results[doc] = -kl_divergence(self.docvecs[doc], qvec)

        results = list(results.items())
        results.sort(key=lambda _: -_[1])
        return results