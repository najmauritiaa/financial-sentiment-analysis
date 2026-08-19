import numpy as np

def predict_svm(text_input, model, vectorizer=None):
    """
    TF-IDF / BoW → pakai vectorizer
    Word2Vec / FastText → input sudah vector
    """

    # CASE 1: TFIDF / BoW (text input)
    if vectorizer is not None:
        text_vector = vectorizer.transform([text_input])

    # CASE 2: Word2Vec / FastText (already vector)
    else:
        text_vector = np.array(text_input)

        # safety check
        if len(text_vector.shape) == 1:
            text_vector = text_vector.reshape(1, -1)

    # prediction
    prediction = model.predict(text_vector)[0]

    # probability (SVM must be trained with probability=True)
    probabilities = model.predict_proba(text_vector)[0]

    return prediction, probabilities