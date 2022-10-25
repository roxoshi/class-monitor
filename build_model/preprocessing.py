import re
import string
import logging
from build_model import config as cn
import gensim.downloader as api
import numpy as np
import pandas as pd

model = api.load("glove-twitter-200")

vec_len = 200

def cleantext(text_str: str) -> str:
    #will replace the html characters with " "
    text_str=re.sub('<.*?>', ' ', text_str)  
    text_str = text_str.translate(str.maketrans(' ',' ',string.punctuation)) # remove punctuations
    text_str = re.sub('[^a-zA-Z]',' ',text_str) # Only alphabets
    text_str = re.sub("\n"," ",text_str)
    text_str = re.sub(' +', ' ', text_str) # remove multiple spaces
    text_str = text_str.lower()
    # will split and join the words
    text_str = text_str.split()
    return text_str

def wordvecs(tweet: str) -> np.array:
    word_array = cleantext(tweet)
    sentence_vector = []
    for x in word_array:
        try:
            row_vec = model[x]
        except:
            row_vec = np.zeros(vec_len)
        assert row_vec.shape[0] == vec_len, "Word vecs not consistent"
        sentence_vector.append(row_vec.reshape(-1,vec_len))
        
        sentence_np = np.mean(sentence_vector, axis=0)    
    return sentence_np

def transform_dataframe(df: pd.DataFrame, colname: str='tweet') -> pd.DataFrame:
    df_features = pd.DataFrame(
        df['tweet']
        .apply(wordvecs)
        .apply(lambda x: x[0])
        .values.tolist())
    return df_features.fillna(0).astype('float')
    
