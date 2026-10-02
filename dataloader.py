import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def load_data(file_path="./data/raw/all_data.csv"):
    data = pd.read_csv(file_path)

    # Keep only English reviews
    data = data[data["Review_Language"] == "English"].copy()

    # Combine the two review fields
    data["Review"] = (
        data["Upside_Review"].fillna("") + " " +
        data["Downside_Review"].fillna("")
    )

    return data


def preprocess_text(text):
    """Lowercase text and remove punctuation and numbers."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    return text


def split_data(data, test_size=0.25, random_state=1):
    """Split data into stratified training and test sets."""

    X = data["Review"]
    y = data["source"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def create_bag_of_words(X_train_text, X_test_text, use_stop_words=True, top_n=500):
    """Create a bag-of-words representation using the top N words."""

    # Load NLTK English stopwords
    stop_words = nltk.corpus.stopwords.words("english")

    vectorizer = CountVectorizer(
        max_features=top_n,
        stop_words=(stop_words if use_stop_words else None)
    )

    # Fit only on training data
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Calculate frequencies of selected words
    word_frequencies = X_train.sum(axis=0).A1

    frequency_df = pd.DataFrame({
        "word": vectorizer.get_feature_names_out(),
        "frequency": word_frequencies
    }).sort_values(
        by="frequency",
        ascending=False
    ).reset_index(drop=True)

    return X_train, X_test, vectorizer, frequency_df



def dataloader(use_stop_words=True, top_n=500):
    data = load_data()

    # Preprocess reviews
    data["Review"] = data["Review"].apply(preprocess_text)

    # Split data
    X_train_text, X_test_text, y_train, y_test = split_data(data)

    # Create BoW using only top N features
    X_train, X_test, vectorizer, frequency_df = create_bag_of_words(
        X_train_text,
        X_test_text,
        use_stop_words=use_stop_words,
        top_n=top_n
    )

    return X_train, X_test, y_train, y_test, vectorizer, frequency_df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer, frequency_df = dataloader(
        use_stop_words=True,
        top_n=500
    )

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    frequency_df.to_csv(
        "./data/word_frequencies.csv",
        index=False
    )

    print("\nTop 20 most frequent words:")
    print(frequency_df.head(20))