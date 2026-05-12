import pickle
import torch

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

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from src.load_data import load_imdb_dataset
from src.preprocessing import preprocess_text

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model():
    """Funkcja trenująca modele klasyfikacji sentymentu"""
    print("CUDA available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

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

    # BERT
    print("\nTraining BERT...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    model_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    model_bert.to(device)

    train_dataset = IMDBDataset(train_df["review"], y_train, tokenizer_bert)
    test_dataset = IMDBDataset(test_df["review"], y_test, tokenizer_bert)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    optimizer = AdamW(model_bert.parameters(), lr=2e-5)

    # Training loop
    model_bert.train()

    for epoch in range(2):
        print(f"Epoch {epoch + 1}")

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model_bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model_bert.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model_bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    bert_acc = accuracy_score(true_labels, predictions)

    print("BERT Accuracy:", bert_acc)
    print(classification_report(true_labels, predictions))

    results["BERT"] = bert_acc

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
    elif best_model_name == "LSTM":
        lstm_model.save("lstm_model.h5")
    elif best_model_name == "BERT":
        model_bert.save_pretrained("bert_model")
        tokenizer_bert.save_pretrained("bert_model")
    else:
        print("Unknown model selected")

    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("\nTraining completed successfully!")

    return lr_model, vectorizer