#keybert_topic_extractor.py
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch

class KeyBERTTopicExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KeyBERT(model)

    def extract_main_topic(self, text, top_n=1):
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=top_n
        )
        return keywords[0][0] if keywords else ""

