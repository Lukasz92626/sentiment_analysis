import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from transformers import BertForSequenceClassification, BertTokenizer

from src.train_model import train_model
from src.preprocessing import preprocess_text
from src.eda import run_eda

def predict_review(
    review: str,
    model,
    tokenizer_lstm=None,
    tokenizer_bert=None,
    maxlen: int = 200,
) -> str:
    """Predykcja sentymentu dla różnych typów modeli."""
    clean = preprocess_text(review)

    # sklearn Pipeline (LR, NB, SVM) — vectorizer wbudowany
    if isinstance(model, Pipeline):
        pred = model.predict([clean])[0]
        return "Positive" if pred == 1 else "Negative"

    # LSTM
    if isinstance(model, Sequential):
        if tokenizer_lstm is None:
            raise ValueError("tokenizer_lstm jest wymagany dla modelu LSTM.")
        seq = tokenizer_lstm.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        prob = model.predict(padded, verbose=0)[0][0]
        return "Positive" if prob > 0.5 else "Negative"

    # BERT
    if isinstance(model, BertForSequenceClassification):
        if tokenizer_bert is None:
            raise ValueError("tokenizer_bert jest wymagany dla modelu BERT.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        encoding = tokenizer_bert(
            clean,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            ).logits
        pred = torch.argmax(logits, dim=1).item()
        return "Positive" if pred == 1 else "Negative"

    raise TypeError(f"Nieobsługiwany typ modelu: {type(model)}")


def main() -> None:
    # EDA
    print("=" * 50)
    print("1: Exploratory Data Analysis")
    print("=" * 50)
    run_eda()

    print("=" * 50)
    print("2: Training")
    print("=" * 50)
    model, results = train_model()

    tokenizer_bert = None
    if isinstance(model, tuple):
        model, tokenizer_bert = model

    # Przykładowe predykcje
    examples = [
        "This movie was fantastic and the acting was brilliant!",
        "Terrible film, complete waste of time. Awful acting.",
        "I've never been so bored in my life. Absolute garbage.",
    ]

    print("\nExample predictions")
    for review in examples:
        sentiment = predict_review(review, model, tokenizer_bert=tokenizer_bert)
        print(f"  [{sentiment}] {review}")

if __name__ == "__main__":
    main()