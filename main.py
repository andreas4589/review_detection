from dataloader import dataloader
from util import get_model, evaluate_model, save_results

if __name__ == "__main__":
    USE_STOP_WORDS = True
    TOP_N = 500

    # 0: Naive Bayes, 1: Logistic Regression, 2: Decision Tree, 3: Random Forest, 4: Gradient Boosting
    MODEL = 1

    X_train, X_test, y_train, y_test, vectorizer, frequency_df = dataloader(
        use_stop_words=USE_STOP_WORDS,
        top_n=TOP_N
    )

    print("Params:")
    print(f"Use stop words: {USE_STOP_WORDS}")
    print(f"Top N words: {TOP_N}")

    print("\nTraining shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nTop 5 words by frequency:")
    print(frequency_df.head(5))

    print("\nTraining model...")
    model, model_name = get_model(MODEL, X_train, y_train)

    print("\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test)

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")

    save_results(
        model_name,
        USE_STOP_WORDS,
        TOP_N,
        X_train.shape[1],
        results
    )
    print(f"\nResults saved in ./results/{model_name}/")
