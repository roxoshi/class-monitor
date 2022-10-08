from preprocessing import cleantext
from tqdm import tqdm
import os
import time



def _file_to_corpus(fpath: str) -> list:
    li_tuples = []
    with open(fpath, 'r') as f:
        raw_txt = f.readlines()
    for row in raw_txt:
        row_list = row.split(",", maxsplit=3)
        li_tuples.append((row_list[1],row_list[2])) # tuple of (label, tweet)
    
    #print("Sample corpus tuple:\n",li_tuples)
    return li_tuples

def clean_tuple(tuple_list):
    return [(x[0],cleantext(x[1])) for x in tuple_list]

def word_freq_count(corpus:list,method:str='count') -> dict:
    """
    return a dictionary of all words in the corpus and their
    numeric weight according to the defined method
    args:
    corpus = list of tuples with label and documents
    method = count | frequency
    """
    #corpus = [x[1] for x in corpus]
    d_wordcount = {}
    total_words = 0
    for row in tqdm(corpus):
        word_list = row[1].split(" ")
        total_words += len(word_list)
        for word in word_list: # row[0] is label, row[1] is tweet
            if word not in d_wordcount.keys():
                d_wordcount[word] = 1
            d_wordcount[word] += 1
    if method == 'frequency':
        for k,v in d_wordcount.items():
            d_wordcount[k] = round(v/total_words,4)
    return d_wordcount

def document_to_vector(corpus: list) -> str:
    """
    Input: list of tuple like [(label, tweet),..,..,]
    Output: Returns path of csv with label as first column
        and feature vector
    """
    wordcount = word_freq_count(corpus)
    max_vec_len = 0
    output_vector_corpus = os.path.join('data','output_vector.txt')
    with open(output_vector_corpus,'w') as f:
        for idx,row in tqdm(enumerate(corpus)):
            if idx == 0: continue
            doc_text_vector = row[1].split(" ")
            doc_vector = [wordcount.get(x,0) for x in doc_text_vector]
            if len(doc_vector) > max_vec_len:
                max_vec_len = len(doc_vector)
            doc_vector_str = ",".join([str(x) for x in doc_vector])
            f.write(f"{row[0]},{doc_vector_str}\n")
    
    print("Max length of document vector is:", max_vec_len)
    return output_vector_corpus

def main():
    csv_path = os.path.join('data','train.csv')
    inp = csv_path
    for func in [_file_to_corpus, clean_tuple, document_to_vector]:
        inp = func(inp)
    
    return inp


        