import warnings
warnings.filterwarnings('ignore')

from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

# download the model and return as object ready for use
model = api.load("glove-twitter-100")
print(model["policy"])