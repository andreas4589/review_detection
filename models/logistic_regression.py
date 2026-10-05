from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        random_state=1
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_