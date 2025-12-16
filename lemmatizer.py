# lemmatizer.py
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextLemmatizer:
    @staticmethod
    def lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)



import re

def extract_emails(text):
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

def extract_phones(text):
    return re.findall(r"\+?\d[\d\s\-]{8,}", text)

def extract_address(text):
    if "Door No" in text:
        start = text.find("Door No")
        return text[start:start+150]
    return None
