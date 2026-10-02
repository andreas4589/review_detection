from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def train_gradient_boosting(X_train, y_train, scoring="accuracy"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        "n_estimators": [100, 200, 400, 700],   # number of boosting rounds (trees)
        "learning_rate": [0.03, 0.01], # shrinkage per tree
        "max_depth": [2, 3, 4],            # depth of each tree
        "subsample": [0.8, 1.0],           # <1.0 = stochastic gradient boosting
    }

    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,   # use "f1_macro" if your classes are imbalanced
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print(f"Best CV {scoring}: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_