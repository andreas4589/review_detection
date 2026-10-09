import os
from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score


def get_model(nr, X_train, y_train):
    if nr == 0:
        from models.multi_naive_bayes import train_naive_bayes

        model, params = train_naive_bayes(X_train, y_train)
        return model, "naive_bayes", params

    elif nr == 1:
        from models.logistic_regression import train_logistic_regression

        model, params = train_logistic_regression(X_train, y_train)
        return model, "logistic_regression", params

    elif nr == 2:
        from models.classification_tree import train_decision_tree

        model, params = train_decision_tree(X_train, y_train)
        return model, "decision_tree", params

    elif nr == 3:
        from models.random_forest import train_random_forest

        model, params = train_random_forest(X_train, y_train)
        return model, "random_forest", params

    elif nr == 4:
        from models.gradient_boost import train_gradient_boosting

        model, params = train_gradient_boosting(X_train, y_train)
        return model, "gradient_boosting", params

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


def save_results(
    model_name,
    PARAMS,
    num_features,
    params,
    results
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"./results/{model_name}"

    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/{timestamp}.txt", "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Date: {timestamp}\n\n")

        file.write("Parameters:\n")
        for key, value in PARAMS.items():
            file.write(f"\t{key}: {value}\n")
        file.write(f"\tNumber of features: {num_features}\n")

        for name, value in params.items():
            file.write(f"\t{name}: {value}\n")

        file.write("\nResults:\n")
        file.write(f"\tAccuracy:  {results['accuracy']:.4f}\n")
        file.write(f"\tPrecision: {results['precision']:.4f}\n")
        file.write(f"\tRecall:    {results['recall']:.4f}\n")
        file.write(f"\tF1 Score:  {results['f1']:.4f}\n")
