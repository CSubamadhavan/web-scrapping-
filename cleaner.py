# cleaner.py
import re
import nltk
from nltk.corpus import stopwords
from typing import List, Tuple
from geotext import GeoText

nltk.download('stopwords')

class Cleaner:
    @staticmethod
    def clean(text: str) -> str:
        stop_words = set(stopwords.words('english'))

        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'\+?\d[\d\s\-]{7,}\d'
        url_pattern = r'https?://\S+|www\.\S+'

        preserved_matches = re.findall(f"{email_pattern}|{phone_pattern}|{url_pattern}", text)
        preserved_set = set(preserved_matches)

        text = text.lower()
        text = re.sub(r"[^\w\s@.:+/,-]", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        words = []
        for word in text.split():
            if word in preserved_set or word not in stop_words:
                words.append(word)

        return ' '.join(words)

    @staticmethod
    def extract_locations(text: str) -> List[Tuple[str, str]]:
        places = GeoText(text)
        locations = []

        for city in places.cities:
            locations.append((city, "CITY"))
        for country in places.countries:
            locations.append((country, "COUNTRY"))

        return locations

