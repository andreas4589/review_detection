from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        random_state=1,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [250, 500, 750, 1000],
        "max_features": ["sqrt", "log2", 0.1, 0.25],
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

    final_model = grid_search.best_estimator_

    return final_model, grid_search.best_params_
