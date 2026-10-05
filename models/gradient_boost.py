from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
            random_state=1
        )

    param_grid = {
        "n_estimators": [100, 200, 400, 700],
        "learning_rate": [0.02, 0.03, 0.01], 
        "max_depth": [2, 3, 4, 5],            
        "subsample": [0.8, 0.9],           
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