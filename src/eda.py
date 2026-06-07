import logging
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from src.load_data import load_imdb_dataset
from src.preprocessing import preprocess_text

log = logging.getLogger(__name__)

# 1. Rozkład klas

def plot_class_distribution(y_train: pd.Series, y_test: pd.Series) -> None:
    """Sprawdza i wizualizuje balans klas w obu zbiorach."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, y, title in zip(axes, [y_train, y_test], ["Train", "Test"]):
        counts = y.value_counts()
        ax.bar(["Negative (0)", "Positive (1)"], counts.values, color=["#e74c3c", "#2ecc71"])
        ax.set_title(f"Class distribution — {title}")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 100, f"{v}\n({v/len(y)*100:.1f}%)", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    print("Train balance:", y_train.value_counts(normalize=True).to_dict())
    print("Test  balance:", y_test.value_counts(normalize=True).to_dict())


# 2. Długość recenzji

def plot_review_length_distribution(train_df: pd.DataFrame, y_train: pd.Series) -> None:
    """Rozkład długości recenzji (w słowach) per klasa."""
    train_df = train_df.copy()
    train_df["word_count"] = train_df["clean_review"].str.split().str.len()
    train_df["char_count"] = train_df["clean_review"].str.len()
    train_df["sentiment"] = y_train.values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sentiment, label, color in [(1, "Positive", "#2ecc71"), (0, "Negative", "#e74c3c")]:
        subset = train_df[train_df["sentiment"] == sentiment]["word_count"]
        axes[0].hist(subset, bins=60, alpha=0.6, label=label, color=color)
        axes[1].hist(
            train_df[train_df["sentiment"] == sentiment]["char_count"],
            bins=60, alpha=0.6, label=label, color=color,
        )

    axes[0].set_title("Rozkład długości recenzji (słowa)")
    axes[0].set_xlabel("Liczba słów")
    axes[0].set_ylabel("Liczba recenzji")
    axes[0].legend()

    axes[1].set_title("Rozkład długości recenzji (znaki)")
    axes[1].set_xlabel("Liczba znaków")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("\n=== Statystyki długości (słowa) ===")
    print(train_df.groupby("sentiment")["word_count"].describe().round(1))

    # Ważne dla LSTM — ile recenzji przekracza maxlen=200
    over_200 = (train_df["word_count"] > 200).mean() * 100
    over_500 = (train_df["word_count"] > 500).mean() * 100
    print(f"\nRecenzji > 200 słów: {over_200:.1f}% (obcinane przez LSTM padding)")
    print(f"Recenzji > 500 słów: {over_500:.1f}%")


# 3. Najczęstsze słowa per klasa

def plot_top_words(train_df: pd.DataFrame, y_train: pd.Series, top_n: int = 20) -> None:
    """Najczęstsze słowa w recenzjach pozytywnych i negatywnych."""
    stopwords = set(STOPWORDS)
    train_df = train_df.copy()
    train_df["sentiment"] = y_train.values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, sentiment, label, color in [
        (axes[0], 1, "Positive", "#2ecc71"),
        (axes[1], 0, "Negative", "#e74c3c"),
    ]:
        text = " ".join(train_df[train_df["sentiment"] == sentiment]["clean_review"])
        words = [w for w in text.lower().split() if w not in stopwords and len(w) > 2]
        counter = Counter(words)
        top_words, counts = zip(*counter.most_common(top_n))

        ax.barh(list(reversed(top_words)), list(reversed(counts)), color=color, alpha=0.8)
        ax.set_title(f"Top {top_n} słów — {label}")
        ax.set_xlabel("Liczba wystąpień")

    plt.tight_layout()
    plt.show()


# 4. WordCloud pos vs neg

def plot_wordclouds(train_df: pd.DataFrame, y_train: pd.Series) -> None:
    """WordCloud dla recenzji pozytywnych i negatywnych osobno."""
    stopwords = set(STOPWORDS)
    train_df = train_df.copy()
    train_df["sentiment"] = y_train.values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, sentiment, label, colormap in [
        (axes[0], 1, "Positive reviews", "Greens"),
        (axes[1], 0, "Negative reviews", "Reds"),
    ]:
        text = " ".join(train_df[train_df["sentiment"] == sentiment]["clean_review"])
        wc = WordCloud(
            width=700, height=400,
            background_color="white",
            stopwords=stopwords,
            max_words=150,
            colormap=colormap,
        ).generate(text)
        ax.imshow(wc)
        ax.axis("off")
        ax.set_title(label, fontsize=14)

    plt.tight_layout()
    plt.show()


# 5. Analiza OOV dla LSTM

def analyze_oov(train_df: pd.DataFrame, test_df: pd.DataFrame, vocab_size: int = 5000) -> None:
    """
    Szacuje odsetek słów Out-Of-Vocabulary dla różnych rozmiarów słownika.
    Ważne dla doboru LSTM_VOCAB_SIZE.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df["clean_review"])

    total_vocab = len(tokenizer.word_index)
    print(f"\n=== Analiza OOV ===")
    print(f"Całkowity słownik (train): {total_vocab:,} unikalnych słów")

    vocab_sizes = [1000, 2000, 5000, 10000, 20000, total_vocab]
    oov_rates = []

    for size in vocab_sizes:
        tok = Tokenizer(num_words=size, oov_token="<OOV>")
        tok.fit_on_texts(train_df["clean_review"])
        seqs = tok.texts_to_sequences(test_df["clean_review"])
        oov_count = sum(1 for seq in seqs for t in seq if t == 1)  # 1 = OOV token index
        total_tokens = sum(len(seq) for seq in seqs)
        oov_rate = oov_count / total_tokens * 100
        oov_rates.append(oov_rate)
        print(f"  vocab_size={size:>6,}  OOV rate: {oov_rate:.2f}%")

    plt.figure(figsize=(8, 4))
    plt.plot(vocab_sizes[:-1], oov_rates[:-1], marker="o", color="#3498db")
    plt.axvline(x=5000, color="red", linestyle="--", label="Obecne ustawienie (5000)")
    plt.xlabel("Rozmiar słownika")
    plt.ylabel("OOV rate (%)")
    plt.title("OOV rate vs rozmiar słownika (LSTM)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 6. Przykładowe recenzje

def show_sample_reviews(train_df: pd.DataFrame, y_train: pd.Series, n: int = 3) -> None:
    """Wyświetla przykładowe recenzje z obu klas."""
    train_df = train_df.copy()
    train_df["sentiment"] = y_train.values
    train_df["word_count"] = train_df["clean_review"].str.split().str.len()

    print("\n=== Przykładowe recenzje POZYTYWNE ===")
    for _, row in train_df[train_df["sentiment"] == 1].sample(n, random_state=42).iterrows():
        print(f"[{row['word_count']} słów] {row['clean_review'][:300]}...\n")

    print("=== Przykładowe recenzje NEGATYWNE ===")
    for _, row in train_df[train_df["sentiment"] == 0].sample(n, random_state=42).iterrows():
        print(f"[{row['word_count']} słów] {row['clean_review'][:300]}...\n")


# Orchestrator

def run_eda() -> tuple:
    """
    Uruchamia pełną analizę EDA.
    Zwraca (train_df, test_df, y_train, y_test) do dalszego użycia.
    """
    print("Loading data for EDA...")
    train_df = load_imdb_dataset("data/aclImdb/train")
    test_df = load_imdb_dataset("data/aclImdb/test")

    train_df["clean_review"] = train_df["review"].apply(preprocess_text)
    test_df["clean_review"] = test_df["review"].apply(preprocess_text)

    y_train = train_df["sentiment"].map({"pos": 1, "neg": 0})
    y_test = test_df["sentiment"].map({"pos": 1, "neg": 0})

    print("\n── 1. Rozkład klas ──")
    plot_class_distribution(y_train, y_test)

    print("\n── 2. Długość recenzji ──")
    plot_review_length_distribution(train_df, y_train)

    print("\n── 3. Najczęstsze słowa ──")
    plot_top_words(train_df, y_train)

    print("\n── 4. WordCloud ──")
    plot_wordclouds(train_df, y_train)

    print("\n── 5. Analiza OOV ──")
    analyze_oov(train_df, test_df)

    print("\n── 6. Przykładowe recenzje ──")
    show_sample_reviews(train_df, y_train)

    return train_df, test_df, y_train, y_test