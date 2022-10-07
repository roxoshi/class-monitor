from preprocessing import cleantext
from tqdm import tqdm
import os
import time


def _unquiwords(corpus:list,method:str='count') -> dict:
    """
    return a dictionary of all words in the corpus and their
    numeric weight according to the defined method
    args:
    corpus = list of documents
    method = count | frequency
    """
    #corpus = [x[1] for x in corpus]
    d_wordcount = {}
    total_words = 0
    for row in tqdm(corpus):
        word_list = row.split(" ")
        total_words += len(word_list)
        for word in word_list: # row[0] is label, row[1] is tweet
            if word not in d_wordcount.keys():
                d_wordcount[word] = 1
            d_wordcount[word] += 1
    if method == 'frequency':
        for k,v in d_wordcount.items():
            d_wordcount[k] = round(v/total_words,4)
    return d_wordcount

def _doc_to_vector(corpus: list) -> str:
    wordcount = _unquiwords(corpus)
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
    return [cleantext(x[1]) for x in tuple_list]

if __name__ == '__main__':
    csv_path = os.path.join('data','train.csv')
    inp = csv_path
    for func in [_file_to_corpus, clean_tuple, _doc_to_vector]:
        inp = func(inp)
    
    print("final output is: ", inp)


        