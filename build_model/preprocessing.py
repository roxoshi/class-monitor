import re
import string
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.basicConfig(level=logging.DEBUG)


class CleanText:

    text_str = None

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

    def tokenizeit(text_str: str) -> list:
        return word_tokenize(text_str)

    def lemmatizeit(text_str: str) -> str:
        wnl = WordNetLemmatizer()
        return wnl.lemmatize(text_str)


class TransformText(CleanText):

    readcsv = str()

    def file_to_corpus(fpath: str) -> List[Tuple[int, list]]:
        li_tuples = []
        with open(fpath, 'r') as f:
            raw_txt = f.readlines()
        del raw_txt[0] # remove header
        for row in raw_txt:
            row_list = row.split(",", maxsplit=3)
            li_tuples.append((row_list[1],row_list[2])) # tuple of (label, tweet)
        
        #print("Sample corpus tuple:\n",li_tuples)
        return li_tuples

    def clean_tuple(tuple_list:List[Tuple[int, list]]) -> List[Tuple[int, list]]:
        return [(int(x[0]),CleanText.cleantext(x[1])) for x in tuple_list]

    def word_freq_count(corpus:List[Tuple[int, list]], method:str='frequency') -> Dict[str,float]:
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
        for row in corpus:
            total_words += len(row[1])
            for word in row[1]: # row[0] is label, row[1] is tweet
                if word not in d_wordcount.keys():
                    d_wordcount[word] = 1
                d_wordcount[word] += 1
        # new_dict = {}
        # i = 0
        # for k, v in sorted(d_wordcount.items(), key=lambda item: item[1]):
        #     if i == 500: break
        #     new_dict.update({k:v})
        #     i += 1

        # del d_wordcount
        if method == 'frequency':
            for k,v in d_wordcount.items():
                d_wordcount[k] = round(v/total_words,4)
        return d_wordcount

    def document_to_vector(corpus: List[Tuple[int, list]]) -> List[Tuple[int, list]]:
        """
        Input: list of tuple like [(label, tweet),..,..,]
        Output: Returns path of csv with label as first column
            and feature vector
        """
        corpus_vector = list()
        wordcount = TransformText.word_freq_count(corpus, method='frequency')
        for idx,row in tqdm(enumerate(corpus)):
            if idx == 0: continue
            doc_text_vector = row[1]
            doc_vector = []
            i = 0
            for k,v in wordcount.items():
                if i == 2000: break
                if k in doc_text_vector:
                    doc_vector.append(v)
                else:
                    doc_vector.append(-1)
                i += 1
            with open('data/output_vec.txt','a') as f:
                f.write(f"{row[0]},{doc_vector}\n")
            corpus_vector.append((row[0],doc_vector))
        return corpus_vector


    def run() -> List[Tuple[int, list]]:
        t = TransformText
        inputfile = t.readcsv
        outputfile = t.file_to_corpus(inputfile)
        outputfile = t.clean_tuple(outputfile)
        outputfile = t.document_to_vector(outputfile)
        return outputfile
    