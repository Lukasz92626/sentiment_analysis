from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def generate_wordcloud(
        texts,
        sentiment_labels=None,
        max_words: int = 200,
        figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Generuje chmurę słów z listy/serii tekstów.
    Jeśli podano sentiment_labels, rysuje osobne chmury dla pos/neg.
    """
    stopwords = set(STOPWORDS)

    if sentiment_labels is not None:
        pos_text = " ".join(texts[sentiment_labels == 1])
        neg_text = " ".join(texts[sentiment_labels == 0])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        for ax, text, title, color in [
            (ax1, pos_text, "Positive reviews", "Greens"),
            (ax2, neg_text, "Negative reviews", "Reds"),
        ]:
            wc = WordCloud(
                width=600, height=400,
                background_color="white",
                stopwords=stopwords,
                max_words=max_words,
                colormap=color,
            ).generate(text)
            ax.imshow(wc)
            ax.axis("off")
            ax.set_title(title)
    else:
        full_text = " ".join(texts)
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            stopwords=stopwords,
            max_words=max_words,
        ).generate(full_text)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc)
        ax.axis("off")
        ax.set_title("WordCloud of Movie Reviews")

    fig.tight_layout()
    plt.show()
    return fig