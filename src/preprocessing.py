import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Pobranie niezbędnych zasobów NLTK
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Funkcja czyszcząca i normalizująca tekst recenzji"""
    # Usuwanie znacznikow html
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')

    # Zamiana na małe litery
    text = text.lower()

    # Usuwanie wszystkiego poza literami
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Podział na słowa
    words = text.split()

    # Usuwanie stopwords
    words = [w for w in words if w not in stop_words]

    # Lematyzacja
    words = [lemmatizer.lemmatize(w) for w in words]

    # Połączenie z powrotem w tekst
    return " ".join(words)