from src.dataloader import dataloader
from src.util import get_model, evaluate_model, save_results

if __name__ == "__main__":
    
    PARAMS = {
        "Use stop words": True,
        "Lemmatization": False,
        "Top N words": 2000, # None: off
        "Min doc frequency": 1, # 1: off
        "Max doc frequency": 1.0, # 1.0: off
        "Chi2": False, # False: off,
        "TF-IDF": False, # False: off
        "N-grams": (1, 1) # (1, 1): off
    }

    # 0: Naive Bayes, 1: Logistic Regression,
    # 2: Decision Tree, 3: Random Forest, 4: Gradient Boosting
    MODEL = 4

    X_train, X_test, y_train, y_test, vectorizer, frequency_df, selector = dataloader(
        PARAMS
    )

    print("Params:")
    for key, value in PARAMS.items():
        print(f"{key}: {value}")
        
    if selector is not None:
        selected_features = vectorizer.get_feature_names_out()[selector.get_support()]
        print("Number of selected features:", len(selected_features))
        print("Selected features:", selected_features)
    
    print("\nTraining shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nTop 5 words by frequency:")
    print(frequency_df.head(5))

    print("\nTraining model...")
    model, model_name, params = get_model(
        MODEL,
        X_train,
        y_train
    )

    print("\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test)

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")

    save_results(
        model_name,
        PARAMS,
        X_train.shape[1],
        params,
        results
    )

    print(f"\nResults saved in ./results/{model_name}/")
