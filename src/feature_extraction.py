import pandas as pd
import nltk

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.feature_selection import SelectKBest, chi2


def get_stop_words():
    """Return English stopwords while preserving negations."""

    stop_words = nltk.corpus.stopwords.words("english")

    negations = {"no", "not"}

    return [
        word for word in stop_words
        if word not in negations
    ]


def create_vectorizer(params):
    """Create the configured CountVectorizer or TfidfVectorizer."""

    stop_words = get_stop_words()

    vectorizer_class = (
        TfidfVectorizer
        if params["TF-IDF"]
        else CountVectorizer
    )

    return vectorizer_class(
        max_features=(
            None
            if params["Chi2"]
            else params["Top N words"]
        ),
        min_df=params["Min doc frequency"],
        max_df=params["Max doc frequency"],
        stop_words=(
            stop_words
            if params["Use stop words"]
            else None
        ),
        ngram_range=params["N-grams"]
    )


def select_features(X_train, X_test, y_train, params):
    """Optionally select features using chi-squared scores."""

    if not params["Chi2"]:
        return X_train, X_test, None

    if y_train is None:
        raise ValueError(
            "y_train is required when Chi2=True."
        )

    k = min(
        params["Top N words"],
        X_train.shape[1]
    )

    selector = SelectKBest(
        score_func=chi2,
        k=k
    )

    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )

    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector


def create_frequency_dataframe(
    X_train,
    vectorizer,
    selector,
    params
):
    """Create a dataframe containing feature statistics."""

    feature_names = vectorizer.get_feature_names_out()

    if selector is not None:
        selected_indices = selector.get_support(indices=True)

        feature_names = feature_names[selected_indices]

        chi2_scores = selector.scores_[selected_indices]

        frequencies = (
            X_train[:, selected_indices]
            .sum(axis=0)
            .A1
        )

        frequency_df = pd.DataFrame({
            "word": feature_names,
            "frequency": frequencies,
            "chi2_score": chi2_scores
        }).sort_values(
            by="chi2_score",
            ascending=False
        ).reset_index(drop=True)

    else:
        frequencies = X_train.sum(axis=0).A1

        frequency_df = pd.DataFrame({
            "word": feature_names,
            "frequency": frequencies
        }).sort_values(
            by="frequency",
            ascending=False
        ).reset_index(drop=True)

    return frequency_df


def create_bag_of_words(
    X_train_text,
    X_test_text,
    params,
    y_train=None
):
    """Create count or TF-IDF features with optional chi-squared selection."""

    # Create and fit vectorizer using training text only
    vectorizer = create_vectorizer(params)

    X_train_counts = vectorizer.fit_transform(
        X_train_text
    )

    X_test_counts = vectorizer.transform(
        X_test_text
    )

    # Optional feature selection
    X_train, X_test, selector = select_features(
        X_train_counts,
        X_test_counts,
        y_train,
        params
    )

    # Generate feature statistics
    frequency_df = create_frequency_dataframe(
        X_train_counts,
        vectorizer,
        selector,
        params
    )

    return (
        X_train,
        X_test,
        vectorizer,
        frequency_df,
        selector
    )