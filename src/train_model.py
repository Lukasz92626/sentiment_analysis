import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.load_data import load_imdb_dataset
from src.preprocessing import preprocess_text

def train_model():
    """Funkcja trenująca modele klasyfikacji sentymentu"""
    print("Loading dataset...")
    train_df = load_imdb_dataset("data/aclImdb/train")
    test_df = load_imdb_dataset("data/aclImdb/test")

    print("Cleaning text...")
    train_df["clean_review"] = train_df["review"].apply(preprocess_text)
    test_df["clean_review"] = test_df["review"].apply(preprocess_text)

    # Vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    # W sumie trzy metody oprócz Vectorizer

    X_train = vectorizer.fit_transform(train_df["clean_review"])
    X_test = vectorizer.transform(test_df["clean_review"])

    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)

    lr_model.fit(X_train, y_train)

    lr_predictions = lr_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_predictions)

    print("Logistic Regression Accuracy:", lr_accuracy)

    print("\nClassification Report\n")
    print(classification_report(y_test, lr_predictions))

    cm = confusion_matrix(y_test, lr_predictions)

    sns.heatmap(cm, annot=True, fmt='d')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Multinomial Naive Bayes
    print("\nTraining Naive Bayes...")
    nb_model = MultinomialNB()

    nb_model.fit(X_train, y_train)

    nb_predictions = nb_model.predict(X_test)

    nb_accuracy = accuracy_score(y_test, nb_predictions)
    print("Naive Bayes Accuracy:", nb_accuracy)

    # Dodać jeszcze LSTM, BERT

    print("\nSaving model...")

    pickle.dump(lr_model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return lr_model, vectorizer