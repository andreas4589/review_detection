print(type(vectorizer).__name__)          # CountVectorizer or TfidfVectorizer?
print(X_train.shape, X_test.shape)        # second number should equal top_n
print(vectorizer.get_feature_names_out()[:30])

# Do stop words actually change the vocabulary?
_, _, _, _, v_on, _  = dataloader.dataloader(use_stop_words=True,  top_n=500)
_, _, _, _, v_off, _ = dataloader.dataloader(use_stop_words=False, top_n=500)
on, off = set(v_on.get_feature_names_out()), set(v_off.get_feature_names_out())
print("Words only without stop-word removal:", sorted(off - on)[:20])