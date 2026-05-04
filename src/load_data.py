import os
import pandas as pd

def load_imdb_dataset(data_dir):
    """Funkcja wczytująca zbiór danych IMDb Movie Reviews"""
    reviews = []
    sentiments = []

    for sentiment in ['pos', 'neg']:

        folder = os.path.join(data_dir, sentiment)

        for file in os.listdir(folder):

            file_path = os.path.join(folder, file)

            with open(file_path, encoding="utf-8") as f:
                review = f.read()

            reviews.append(review)
            sentiments.append(sentiment)

    df = pd.DataFrame({
        "review": reviews,
        "sentiment": sentiments
    })

    return df