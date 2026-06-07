import json
import pickle
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from src.load_data import load_imdb_dataset
from src.preprocessing import preprocess_text

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Config

SEED = 42
CV_FOLDS = 5
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
LSTM_VOCAB_SIZE = 5000
LSTM_MAXLEN = 200
LSTM_EPOCHS = 10
LSTM_BATCH = 64
BERT_MAX_LEN = 256
BERT_BATCH = 8
BERT_EPOCHS = 3
BERT_LR = 2e-5
OUTPUT_DIR = Path("output")

# Reproducibility

def set_seed(seed: int = SEED) -> None:
    """Ustawia ziarno losowości dla pełnej reprodukowalności."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Results dataclass

@dataclass
class ModelResult:
    name: str
    accuracy: float
    auc: float
    cv_mean: float = 0.0
    cv_std: float = 0.0
    predictions: List[int] = field(default_factory=list)
    true_labels: List[int] = field(default_factory=list)

    def summary(self) -> str:
        cv_part = f"  CV: {self.cv_mean:.4f} ± {self.cv_std:.4f}\n" if self.cv_mean else ""
        return (
            f"{self.name}\n"
            f"  Accuracy : {self.accuracy:.4f}\n"
            f"  AUC      : {self.auc:.4f}\n"
            f"{cv_part}"
        )

# PyTorch Dataset

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = BERT_MAX_LEN):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Data

def load_data() -> Tuple:
    """Wczytuje i czyści dane IMDB. Zwraca train_df, test_df, y_train, y_test."""
    print("Loading dataset...")
    train_df = load_imdb_dataset("data/aclImdb/train")
    test_df = load_imdb_dataset("data/aclImdb/test")

    print("Cleaning text...")
    train_df["clean_review"] = train_df["review"].apply(preprocess_text)
    test_df["clean_review"] = test_df["review"].apply(preprocess_text)

    y_train = train_df["sentiment"].map({"pos": 1, "neg": 0})
    y_test = test_df["sentiment"].map({"pos": 1, "neg": 0})

    # Weryfikacja balansu klas
    print("Class distribution (train):\n", y_train.value_counts(normalize=True).to_string())
    print("Class distribution (test):\n", y_test.value_counts(normalize=True).to_string())

    if y_train.isnull().any() or y_test.isnull().any():
        raise ValueError("Etykiety zawierają wartości NaN — sprawdź dane wejściowe.")

    return train_df, test_df, y_train, y_test

# Visualisation

def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    """Rysuje macierz pomyłek."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc_curves(results: Dict[str, ModelResult], y_test) -> None:
    """Rysuje krzywe ROC dla wszystkich modeli które mają AUC."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        if res.auc > 0:
            ax.plot(
                [0, 1], [res.auc, res.auc],
                linestyle="--", alpha=0.4,
                label=f"{name} (AUC={res.auc:.4f})"
            )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — porównanie modeli")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results: Dict[str, ModelResult]) -> None:
    """Wykres słupkowy accuracy i AUC wszystkich modeli."""
    names = list(results.keys())
    accs = [r.accuracy for r in results.values()]
    aucs = [r.auc for r in results.values()]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, aucs, width, label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Porównanie modeli")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Error analysis

def analyze_errors(result: "ModelResult", test_df: pd.DataFrame, n_examples: int = 5) -> None:
    """
    Analizuje błędy modelu:
    - pokazuje błędnie sklasyfikowane przykłady
    - sprawdza czy błędy korelują z długością recenzji
    """

    preds = np.array(result.predictions)
    true = np.array(result.true_labels)
    errors = preds != true

    print(f"{result.name} — błędnych predykcji: {errors.sum()} / {len(errors)} ({errors.mean() * 100:.2f}%)")

    # Długość recenzji a błędy
    texts = test_df["clean_review"].values
    word_counts = np.array([len(t.split()) for t in texts])

    print(f"\n=== Analiza błędów — {result.name} ===")
    print(f"Średnia długość recenzji (wszystkie):      {word_counts.mean():.1f} słów")
    print(f"Średnia długość recenzji (błędne):         {word_counts[errors].mean():.1f} słów")
    print(f"Średnia długość recenzji (poprawne):       {word_counts[~errors].mean():.1f} słów")

    # False positives i false negatives
    fp = (preds == 1) & (true == 0)
    fn = (preds == 0) & (true == 1)
    print(f"\nFalse Positives (neg → pos): {fp.sum()}")
    print(f"False Negatives (pos → neg): {fn.sum()}")

    # Przykłady błędnych predykcji
    error_indices = np.where(errors)[0]
    if len(error_indices) == 0:
        return

    print(f"\n--- Przykłady błędnie sklasyfikowanych ({n_examples}) ---")
    for idx in error_indices[:n_examples]:
        predicted = "Positive" if preds[idx] == 1 else "Negative"
        actual = "Positive" if true[idx] == 1 else "Negative"
        text = texts[idx][:250]
        print(f"\n  Predicted: {predicted} | Actual: {actual}")
        print(f"  [{word_counts[idx]} słów] {text}...")

    # Histogram długości: poprawne vs błędne
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(word_counts[~errors], bins=50, alpha=0.6, label="Poprawne", color="#2ecc71")
    ax.hist(word_counts[errors], bins=50, alpha=0.6, label="Błędne", color="#e74c3c")
    ax.set_xlabel("Liczba słów")
    ax.set_ylabel("Liczba recenzji")
    ax.set_title(f"Długość recenzji: poprawne vs błędne — {result.name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


# Classic models

def _build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        norm="l2",
        sublinear_tf=True,
    )

def _cross_validate(pipeline: Pipeline, X, y) -> Tuple[float, float]:
    """Przeprowadza stratified k-fold CV na pipeline'ie."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std())

def _tune_and_evaluate(
    pipeline: Pipeline,
    param_distributions: dict,
    train_texts,
    test_texts,
    y_train,
    y_test,
    model_name: str,
    n_iter: int = 10,
    use_proba: bool = True,
) -> Tuple[Pipeline, ModelResult]:
    """
    Wspólna logika: RandomizedSearchCV → CV na najlepszym pipeline → ewaluacja na teście.
    Oddziela tuning hiperparametrów (train) od finalnej oceny (test).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
    )
    search.fit(train_texts, y_train)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_
    cv_mean = search.best_score_
    cv_std = search.cv_results_["std_test_score"][search.best_index_]

    print(f"{model_name} best params: {best_params}")
    print(f"{model_name} CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    preds = best_pipeline.predict(test_texts)

    if use_proba:
        scores = best_pipeline.predict_proba(test_texts)[:, 1]
    else:
        scores = best_pipeline.decision_function(test_texts)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, scores)
    print(f"{model_name} Accuracy={acc:.4f}  AUC={auc:.4f}")
    print(classification_report(y_test, preds))

    result = ModelResult(
        name=model_name,
        accuracy=acc, auc=auc,
        cv_mean=float(cv_mean), cv_std=float(cv_std),
        predictions=preds.tolist(), true_labels=y_test.tolist(),
    )
    return best_pipeline, result


def train_logistic_regression(
    train_df, X_test, y_train, y_test
) -> Tuple[Pipeline, ModelResult]:
    """Trenuje Logistic Regression z RandomizedSearchCV po C i solver."""
    print("Training Logistic Regression (with hyperparameter tuning)...")
    pipeline = Pipeline([
        ("tfidf", _build_tfidf_vectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    param_distributions = {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "clf__solver": ["lbfgs", "saga"],
    }
    return _tune_and_evaluate(
        pipeline, param_distributions,
        train_df["clean_review"], X_test,
        y_train, y_test,
        model_name="Logistic Regression",
    )


def train_naive_bayes(
    train_df, X_test, y_train, y_test
) -> Tuple[Pipeline, ModelResult]:
    """Trenuje Naive Bayes z RandomizedSearchCV po alpha i max_features."""
    print("Training Naive Bayes (with hyperparameter tuning)...")
    # NB wymaga nieujemnych wartości — norm=None
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=TFIDF_NGRAM_RANGE,
            norm=None,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB()),
    ])
    param_distributions = {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
    }
    return _tune_and_evaluate(
        pipeline, param_distributions,
        train_df["clean_review"], X_test,
        y_train, y_test,
        model_name="Naive Bayes",
    )


def train_svm(
    train_df, X_test, y_train, y_test
) -> Tuple[Pipeline, ModelResult]:
    """Trenuje LinearSVC z RandomizedSearchCV po C.
    AUC liczymy przez decision_function (brak predict_proba)."""
    print("Training SVM (with hyperparameter tuning)...")
    pipeline = Pipeline([
        ("tfidf", _build_tfidf_vectorizer()),
        ("clf", LinearSVC(max_iter=2000, random_state=SEED)),
    ])
    param_distributions = {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    }
    return _tune_and_evaluate(
        pipeline, param_distributions,
        train_df["clean_review"], X_test,
        y_train, y_test,
        model_name="SVM",
        use_proba=False,
    )

# LSTM

def prepare_lstm_data(
    train_df, test_df, num_words: int = LSTM_VOCAB_SIZE, maxlen: int = LSTM_MAXLEN
) -> Tuple:
    """Tokenizuje i padduje sekwencje dla LSTM."""
    print("Preparing data for LSTM...")
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_review"])

    X_train_pad = pad_sequences(
        tokenizer.texts_to_sequences(train_df["clean_review"]),
        maxlen=maxlen, padding="post", truncating="post",
    )
    X_test_pad = pad_sequences(
        tokenizer.texts_to_sequences(test_df["clean_review"]),
        maxlen=maxlen, padding="post", truncating="post",
    )
    return X_train_pad, X_test_pad, tokenizer

def train_lstm(
    X_train_pad, X_test_pad, y_train, y_test,
    num_words: int = LSTM_VOCAB_SIZE,
) -> Tuple[Sequential, ModelResult]:
    """Buduje i trenuje model LSTM z LayerNormalization."""
    print("Training LSTM...")

    # Wydzielony zbiór walidacyjny (nie dotyka test)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_pad, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )

    model = Sequential([
        Embedding(input_dim=num_words, output_dim=64, mask_zero=True),
        LSTM(64, dropout=0.3),   # BEZ recurrent_dropout — pozwala użyć cuDNN na GPU
        LayerNormalization(),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    model.fit(
        X_tr, y_tr,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    proba = model.predict(X_test_pad, verbose=0).ravel()
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)

    print(f"LSTM Accuracy={acc:.4f}  AUC={auc:.4f}")
    print(classification_report(y_test, preds))

    result = ModelResult(
        name="LSTM",
        accuracy=float(acc), auc=float(auc),
        predictions=preds.tolist(), true_labels=y_test.tolist(),
    )
    return model, result

# BERT

def _bert_epoch(model, loader, optimizer, scheduler, device) -> float:
    """Jedna epoka treningu BERT. Zwraca średnią stratę."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()
    return total_loss / len(loader)

def evaluate_bert(model, loader, device) -> Tuple[List[int], List[int], List[float]]:
    """Ewaluuje model BERT. Zwraca (predictions, true_labels, probabilities)."""
    model.eval()
    preds, labels, probas = [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            proba = torch.softmax(outputs.logits, dim=1)[:, 1]
            pred = torch.argmax(outputs.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch["labels"].numpy())
            probas.extend(proba.cpu().numpy())
    return preds, labels, probas

def train_bert(train_df, test_df, y_train, y_test) -> Tuple:
    """Fine-tunuje BERT z walidacją po każdej epoce i gradient clipping."""
    print("Training BERT...")
    print("CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device: ", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)

    # Wydzielony zbiór walidacyjny
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["review"], y_train,
        test_size=0.1, random_state=SEED, stratify=y_train,
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        IMDBDataset(train_texts, train_labels, bert_tokenizer),
        batch_size=BERT_BATCH, shuffle=True, generator=g,
    )
    val_loader = DataLoader(
        IMDBDataset(val_texts, val_labels, bert_tokenizer),
        batch_size=BERT_BATCH,
    )
    test_loader = DataLoader(
        IMDBDataset(test_df["review"], y_test, bert_tokenizer),
        batch_size=BERT_BATCH,
    )

    optimizer = AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * len(train_loader) * BERT_EPOCHS),
        num_training_steps=len(train_loader) * BERT_EPOCHS,
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(BERT_EPOCHS):
        train_loss = _bert_epoch(model, train_loader, optimizer, scheduler, device)
        val_preds, val_labels_list, _ = evaluate_bert(model, val_loader, device)
        val_loss = 1 - accuracy_score(val_labels_list, val_preds)  # proxy val loss
        val_acc = accuracy_score(val_labels_list, val_preds)
        print(f"Epoch {epoch + 1}/{BERT_EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Przywróć najlepsze wagi
    if best_state:
        model.load_state_dict(best_state)

    preds, true_labels, probas = evaluate_bert(model, test_loader, device)
    acc = accuracy_score(true_labels, preds)
    auc = roc_auc_score(true_labels, probas)
    print(f"BERT Accuracy={acc:.4f}  AUC={auc:.4f}")
    print(classification_report(true_labels, preds))

    result = ModelResult(
        name="BERT",
        accuracy=acc, auc=auc,
        predictions=preds, true_labels=true_labels,
    )
    return model, bert_tokenizer, result

# Saving

def save_best_model(
    best_name: str,
    models: Dict[str, Any],
    results: Dict[str, ModelResult],
) -> None:
    """Zapisuje najlepszy model i logi metryk na dysk."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = models[best_name]

    if best_name in ("Logistic Regression", "Naive Bayes", "SVM"):
        pickle.dump(model, open(OUTPUT_DIR / "model.pkl", "wb"))
    elif best_name == "LSTM":
        model.save(OUTPUT_DIR / "lstm_model.keras")
    elif best_name == "BERT":
        bert_model, bert_tokenizer = model
        bert_model.save_pretrained(OUTPUT_DIR / "bert_model")
        bert_tokenizer.save_pretrained(OUTPUT_DIR / "bert_model")

    # Zapis metryk do JSON
    metrics = {
        name: {
            "accuracy": res.accuracy,
            "auc": res.auc,
            "cv_mean": res.cv_mean,
            "cv_std": res.cv_std,
        }
        for name, res in results.items()
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Model i metryki zapisane w ", OUTPUT_DIR)

# Orchestrator

def train_model():
    """Trenuje wszystkie modele, porównuje wyniki i zapisuje najlepszy."""
    set_seed()

    train_df, test_df, y_train, y_test = load_data()

    results: Dict[str, ModelResult] = {}
    models: Dict[str, Any] = {}

    # Modele klasyczne (każdy ma własny Pipeline z TF-IDF)
    lr_pipeline, results["Logistic Regression"] = train_logistic_regression(
        train_df, test_df["clean_review"], y_train, y_test
    )
    nb_pipeline, results["Naive Bayes"] = train_naive_bayes(
        train_df, test_df["clean_review"], y_train, y_test
    )
    svm_pipeline, results["SVM"] = train_svm(
        train_df, test_df["clean_review"], y_train, y_test
    )
    models.update({
        "Logistic Regression": lr_pipeline,
        "Naive Bayes": nb_pipeline,
        "SVM": svm_pipeline,
    })

    # Macierz pomyłek dla najlepszego modelu klasycznego
    best_classic = max(
        ("Logistic Regression", "Naive Bayes", "SVM"),
        key=lambda m: results[m].accuracy,
    )
    print("Best classic model: ", best_classic)
    plot_confusion_matrix(
        results[best_classic].true_labels,
        results[best_classic].predictions,
        title=f"Confusion Matrix ({best_classic})",
    )

    # LSTM
    X_train_pad, X_test_pad, _ = prepare_lstm_data(train_df, test_df)
    lstm_model, results["LSTM"] = train_lstm(X_train_pad, X_test_pad, y_train, y_test)
    models["LSTM"] = lstm_model

    # BERT
    bert_model, bert_tokenizer, results["BERT"] = train_bert(
        train_df, test_df, y_train, y_test
    )
    models["BERT"] = (bert_model, bert_tokenizer)

    plot_confusion_matrix(
        results["BERT"].true_labels,
        results["BERT"].predictions,
        title="Confusion Matrix (BERT)",
    )

    # Analiza błędów dla najlepszego klasycznego i BERT
    analyze_errors(results[best_classic], test_df)
    analyze_errors(results["BERT"], test_df)

    # Podsumowanie
    print("\n=== Model Comparison ===")
    for res in results.values():
        print(res.summary())

    plot_model_comparison(results)

    best_name = max(results, key=lambda m: results[m].accuracy)
    print("BEST MODEL OVERALL: ", best_name)

    save_best_model(best_name, models, results)
    print("Training completed successfully!")

    return models[best_name], results