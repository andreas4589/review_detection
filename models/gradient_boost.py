from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model
