from src.train_model import train_model
from src.preprocessing import preprocess_text
from src.visualization import generate_wordcloud
from src.load_data import load_imdb_dataset

def predict_review(review, model, vectorizer):
    """Funkcja do predykcji sentymentu pojedynczej recenzji"""
    clean_review = preprocess_text(review)

    vector = vectorizer.transform([clean_review])

    prediction = model.predict(vector)

    return prediction[0]

def main():
    model, vectorizer = train_model()

    print("\nExample prediction\n")

    review = "This movie was fantastic and the acting was brilliant!"

    sentiment = predict_review(review, model, vectorizer)

    print("Review:", review)

    print("Predicted sentiment:", sentiment)

    print("\nGenerating WordCloud...")

    df = load_imdb_dataset("data/aclImdb/train")

    generate_wordcloud(" ".join(df["review"]))

if __name__ == "__main__":
    main()