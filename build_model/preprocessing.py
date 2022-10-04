from lib2to3.pgen2 import token
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



def cleantext(text_str: str) -> str:
    #will replace the html characters with " "
    text_str=re.sub('<.*?>', ' ', text_str)  
    text_str = text_str.translate(str.maketrans(' ',' ',string.punctuation)) # remove punctuations
    text_str = re.sub('[^a-zA-Z]',' ',text_str) # Only alphabets
    text_str = re.sub("\n"," ",text_str)
    text_str = text_str.lower()
    # will split and join the words
    text_str=' '.join(text_str.split())
    return text_str

def tokenizeit(text_str: str) -> list:
    return word_tokenize(text_str)

def lemmatizeit(text_str: str) -> str:
    wnl = WordNetLemmatizer()
    return wnl.lemmatize(text_str)

if __name__ == "__main__":
    txt = "Hello12 My name is      Admiral! General! Aaladeen:)"
    clntxt = cleantext(txt)
    print(clntxt)
    tokentxt = tokenizeit(clntxt)
    print(tokentxt)
    lemtxt = [lemmatizeit(x) for x in tokentxt]
    print(lemtxt)
