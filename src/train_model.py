import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

    #print(train_df["sentiment"].unique())
    y_train = train_df["sentiment"].map({"pos": 1, "neg": 0})
    y_test = test_df["sentiment"].map({"pos": 1, "neg": 0})

    # Vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    # W sumie trzy metody oprócz Vectorizer

    X_train = vectorizer.fit_transform(train_df["clean_review"])
    X_test = vectorizer.transform(test_df["clean_review"])

    results = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    lr_predictions = lr_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_predictions)

    print("Logistic Regression Accuracy:", lr_accuracy)
    print(classification_report(y_test, lr_predictions))

    results["Logistic Regression"] = lr_accuracy

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

    results["Naive Bayes"] = nb_accuracy

    # SVM
    print("\nTraining SVM...")
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    print("SVM Accuracy:", svm_accuracy)

    results["SVM"] = svm_accuracy

    # Confusion Matrix
    best_classic_model = max(results, key=results.get)
    print(f"\nBest classic model: {best_classic_model}")

    if best_classic_model == "Logistic Regression":
        best_pred = lr_predictions
    elif best_classic_model == "Naive Bayes":
        best_pred = nb_predictions
    else:
        best_pred = svm_predictions

    cm = confusion_matrix(y_test, best_pred)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({best_classic_model})")
    plt.show()

    # Dodać jeszcze LSTM, BERT

    # LSTM
    print("\nPreparing data for LSTM...")

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_df["clean_review"])

    X_train_seq = tokenizer.texts_to_sequences(train_df["clean_review"])
    X_test_seq = tokenizer.texts_to_sequences(test_df["clean_review"])

    X_train_pad = pad_sequences(X_train_seq, maxlen=200)
    X_test_pad = pad_sequences(X_test_seq, maxlen=200)

    print("\nTraining LSTM...")

    lstm_model = Sequential([
        Embedding(input_dim=5000, output_dim=128),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    lstm_model.fit(X_train_pad, y_train, epochs=3, batch_size=64)

    lstm_acc = lstm_model.evaluate(X_test_pad, y_test)[1]
    print("LSTM Accuracy:", lstm_acc)

    results["LSTM"] = lstm_acc

    # BERT !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Comparison
    print("\n=== Model Comparison ===")
    for model, acc in results.items():
        print(f"{model}: {acc}")

    best_model_name = max(results, key=results.get)
    print(f"\nBEST MODEL OVERALL: {best_model_name}")

    print("\nSaving model...")

    if best_model_name == "Logistic Regression":
        pickle.dump(lr_model, open("model.pkl", "wb"))
    elif best_model_name == "Naive Bayes":
        pickle.dump(nb_model, open("model.pkl", "wb"))
    elif best_model_name == "SVM":
        pickle.dump(svm_model, open("model.pkl", "wb"))
    else:
        print("Skipping LSTM save (use model.save() if needed)")

    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("\nTraining completed successfully!")

    return lr_model, vectorizer