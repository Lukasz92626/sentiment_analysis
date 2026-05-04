from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text):
    """Funkcja generująca chmurę słów z podanego tekstu"""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(text)

    plt.figure(figsize=(10, 5))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("WordCloud of Movie Reviews")

    plt.show()