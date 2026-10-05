import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=1)

    param_grid = {
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.0, 0.1, 0.2],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_