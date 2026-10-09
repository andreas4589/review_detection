import pandas as pd


def load_data(file_path="./data/raw/all_data.csv"):
    """Load English reviews and combine the review fields."""

    data = pd.read_csv(file_path)

    # Keep only English reviews
    data = data[data["Review_Language"] == "English"].copy()

    # Combine the two review fields
    data["Review"] = (
        data["Upside_Review"].fillna("") + " " +
        data["Downside_Review"].fillna("")
    )

    return data