from sklearn.model_selection import train_test_split


def split_data(data, test_size=0.25, random_state=1):
    """Split reviews and labels into stratified training and test sets."""

    X = data["Review"]
    y = data["source"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )