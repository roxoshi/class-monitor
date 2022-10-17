import re
import string
import logging
import json
from textwrap import indent
from typing import Dict, List, Tuple
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from build_model import config as cn



class CleanText:

    @staticmethod
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
        return li_tuples

    def clean_tuple(tuple_list:List[Tuple[int, list]]) -> List[Tuple[int, list]]:
        return [(int(x[0]),CleanText.cleantext(x[1])) for x in tuple_list]

    def word_freq_count(corpus:List[Tuple[int, list]], method:str='frequency') -> str:
        """
        return a path to dictionary of all words in the corpus and their
        numeric weight according to the defined method.
        The dict looks like this Dict[str,float]
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
        if method == 'frequency':
            for k,v in d_wordcount.items():
                d_wordcount[k] = round(v/total_words,4)
        with open(cn.BOW_PATH, "w") as f:
            json.dump(d_wordcount,f,indent=4)
        return cn.BOW_PATH

    def document_to_vector(corpus: List[Tuple[int, list]]) -> List[Tuple[int, list]]:
        """
        Input: list of tuple like [(label, tweet),..,..,]
        Output: Returns path of csv with label as first column
            and feature vector
        """
        corpus_vector = list()
        
        try:
            wordcount = json.load(open(cn.BOW_PATH))
        except OSError as e:
            logging.info("BOW dictionary does not exist.")
            logging.info("Creating now.")
            TransformText.word_freq_count(corpus, method='frequency')
            wordcount = json.load(open(cn.BOW_PATH))

        for idx,row in tqdm(enumerate(corpus)):
            doc_text_vector = row[1]
            doc_vector = []
            i = 0
            for k,v in wordcount.items():
                if i == 1000: break
                if k in doc_text_vector:
                    doc_vector.append(v)
                else:
                    doc_vector.append(-1)
                i += 1
            with open('data/output_vec.txt','a') as f:
                f.write(f"{row[0]},{doc_vector}\n")
            corpus_vector.append((row[0],doc_vector))
        return corpus_vector


    def fit_transform() -> List[Tuple[int, list]]:
        t = TransformText
        inputfile = t.readcsv
        outputfile = t.file_to_corpus(inputfile)
        outputfile = t.clean_tuple(outputfile)
        outputfile = t.document_to_vector(outputfile)
        return outputfile
    
    def transform(input_tweet: str) -> list:
        t = TransformText
        t_input = [(0,input_tweet)]
        outputfile = t.clean_tuple(t_input)
        outputfile = t.document_to_vector(outputfile)
        return outputfile[0][1]

    