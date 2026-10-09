from src.data_loader import load_data
from src.preprocessing import preprocess_reviews
from src.data_split import split_data
from src.feature_extraction import create_bag_of_words


def dataloader(params):
    """Load, preprocess, split, and vectorize the review dataset."""

    # Load data
    data = load_data()

    # Preprocess reviews
    data["Review"] = preprocess_reviews(
        data["Review"],
        lemmatization=params["Lemmatization"]
    )

    # Split data
    (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ) = split_data(data)

    # Extract features
    (
        X_train,
        X_test,
        vectorizer,
        frequency_df,
        selector
    ) = create_bag_of_words(
        X_train_text,
        X_test_text,
        params,
        y_train=y_train
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer,
        frequency_df,
        selector
    )