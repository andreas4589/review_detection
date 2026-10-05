import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_decision_tree(X_train, y_train, scoring="accuracy"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Candidate pruning strengths from the cost-complexity path.
    # Drop the last alpha (it prunes the tree down to just the root)
    # and thin the list so the grid stays small.
    path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
    alphas = np.unique(path.ccp_alphas[:-1])
    if len(alphas) > 20:
        alphas = np.quantile(alphas, np.linspace(0, 1, 20))

    param_grid = {
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": [None, "balanced"],
        "ccp_alpha": alphas,
    }

    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,   # use "f1_macro" if your classes are imbalanced
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_