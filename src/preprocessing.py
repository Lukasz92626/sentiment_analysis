import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pobranie niezbędnych zasobów NLTK
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Funkcja czyszcząca i normalizująca tekst recenzji"""
    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)