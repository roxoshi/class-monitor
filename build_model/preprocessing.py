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
    max_vector_len = 0

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
        for row in tqdm(corpus):
            total_words += len(row[1])
            for word in row[1]: # row[0] is label, row[1] is tweet
                if word not in d_wordcount.keys():
                    d_wordcount[word] = 1
                d_wordcount[word] += 1
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
        wordcount = TransformText.word_freq_count(corpus)
        max_vec_len = 0
        for idx,row in tqdm(enumerate(corpus)):
            if idx == 0: continue
            doc_text_vector = row[1]
            doc_vector = [wordcount.get(x,0) for x in doc_text_vector]
            if len(doc_vector) > max_vec_len:
                max_vec_len = len(doc_vector)
            corpus_vector.append((row[0],doc_vector))
        TransformText.max_vector_len = max_vec_len
        return corpus_vector

    def vector_padding(corpus: List[Tuple[int, list]]) -> List[Tuple[int, list]]:
        corpus_new = []
        for idx, vec in enumerate(corpus):
            if len(vec[1]) < TransformText.max_vector_len:
                padding_length = (TransformText.max_vector_len - len(vec[1])) -1 
                corpus_new.append((vec[0], vec[1] + [-1.0] + [0.0]*padding_length))
            else:
                corpus_new.append(vec)
        return corpus_new

    def run() -> List[Tuple[int, list]]:
        t = TransformText
        inputfile = t.readcsv
        outputfile = t.file_to_corpus(inputfile)
        outputfile = t.clean_tuple(outputfile)
        outputfile = t.document_to_vector(outputfile)
        outputfile = t.vector_padding(outputfile)
        return outputfile
