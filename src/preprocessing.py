import re
import pandas as pd

import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


def preprocess_text(text, lemmatization=False):
    """Lowercase text, remove punctuation and numbers, and optionally lemmatize."""

    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Keep alphabetic characters and whitespace
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Optional lemmatization
    if lemmatization:
        words = text.split()

        words = [
            lemmatizer.lemmatize(word)
            for word in words
        ]

        text = " ".join(words)

    return text


def preprocess_reviews(reviews, lemmatization=False):
    """Preprocess a collection of reviews."""

    return reviews.apply(
        lambda text: preprocess_text(
            text,
            lemmatization=lemmatization
        )
    )