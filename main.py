from dataloader import dataloader
from models.logistic_regression import train_logistic_regression
from sklearn.metrics import precision_score, recall_score, f1_score


if __name__ == "__main__":
    USE_STOP_WORDS = True
    TOP_N = 500
    
    X_train, X_test, y_train, y_test, vectorizer, frequency_df = dataloader(
        use_stop_words=USE_STOP_WORDS,
        top_n=TOP_N
    )
    
    print("Parameters:")
    print(f"Use stop words: {USE_STOP_WORDS}")
    print(f"Top N words: {TOP_N}")
    
    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    print("\nTop 5 words by frequency:")
    print(frequency_df.head(5))

    print("\nTraining model...")
    model = train_logistic_regression(X_train, y_train)
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
