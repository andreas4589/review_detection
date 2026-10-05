from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'force_alpha': [True, False],
        'fit_prior': [True, False],
        'class_prior': [None, [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]
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