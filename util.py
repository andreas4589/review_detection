import os
from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score


def get_model(nr, X_train, y_train):
    if nr == 0:
        from models.multi_naive_bayes import train_naive_bayes
        return train_naive_bayes(X_train, y_train), "naive_bayes"
    elif nr == 1:
        from models.logistic_regression import train_logistic_regression
        return train_logistic_regression(X_train, y_train), "logistic_regression"
    elif nr == 2:
        from models.classification_tree import train_decision_tree
        return train_decision_tree(X_train, y_train), "decision_tree"
    elif nr == 3:
        from models.random_forest import train_random_forest
        return train_random_forest(X_train, y_train), "random_forest"
    elif nr == 4:
        from models.gradient_boost import train_gradient_boosting
        return train_gradient_boosting(X_train, y_train), "gradient_boosting"
    else:
        raise ValueError("MODEL must be between 0 and 4")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": model.score(X_test, y_test),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def save_results(model_name, use_stop_words, top_n, num_features, results):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"./results/{model_name}"

    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/{timestamp}.txt", "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Date: {timestamp}\n\n")

        file.write("Parameters:\n")
        file.write(f"\tUse stop words: {use_stop_words}\n")
        file.write(f"\tTop N words: {top_n}\n")
        file.write(f"\tNumber of features: {num_features}\n\n")

        file.write("Results:\n")
        file.write(f"\tAccuracy:  {results['accuracy']:.4f}\n")
        file.write(f"\tPrecision: {results['precision']:.4f}\n")
        file.write(f"\tRecall:    {results['recall']:.4f}\n")
        file.write(f"\tF1 Score:  {results['f1']:.4f}\n")
