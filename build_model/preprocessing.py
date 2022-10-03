import re
import string


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

if __name__ == "__main__":
    txt = "Hello12 My name is      Admiral! General! Aaladeen :)"
    print(cleantext(txt))
