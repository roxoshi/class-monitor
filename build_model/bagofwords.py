import csv
from preprocessing import cleantext
from tqdm import tqdm
import os


def _unquiwords(corpus:list) -> dict:
    d_wordcount = {}
    for row in tqdm(corpus):
        for word in row[1]: # row[0] is label, row[1] is tweet
            if word not in d_wordcount.keys():
                d_wordcount[word] = 1
            d_wordcount[word] += 1

    return d_wordcount

def _doc_to_vector(corpus: list, wordcount: dict) -> list:
    output_vector_corpus = os.path.join('data','output_vector.txt')
    with open(output_vector_corpus,'w') as f:
        for idx,row in tqdm(enumerate(corpus)):
            if idx == 0: continue
            doc_text_vector = row[1].split(" ")
            doc_vector = [wordcount.get(x,0) for x in doc_text_vector]
            doc_vector_str = ",".join(doc_vector)
            print("row vector is: ",doc_vector_str)
            f.write(f"{row[0]},{','.join(doc_vector)}\n")
    
    print("Length of document vector is:", len(doc_vector))



def _file_to_corpus(fpath: str) -> list:
    li_tuples = []
    with open(fpath, 'r') as f:
        raw_txt = f.readlines()
    for row in raw_txt:
        row_list = row.split(",", maxsplit=3)
        li_tuples.append((row_list[1],cleantext(row_list[2]))) # tuple of (label, tweet)
    
    #print("Sample corpus tuple:\n",li_tuples)
    
    return li_tuples

if __name__ == '__main__':
    csv_path = os.path.join('data','train.csv')
    li_tuples = _file_to_corpus(csv_path)
    wordcount = _unquiwords(li_tuples)
    _doc_to_vector(li_tuples, wordcount)


        