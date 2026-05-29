import numpy as np

def predict_svm(text, model, vectorizer=None):
    # TF-IDF / BoW
    if vectorizer is not None:
        text_vector = vectorizer.transform([text])
    # Word2Vec / FastText
    else:
        text_vector = np.array(text).reshape(1, -1)
    # prediction
    prediction = model.predict(text_vector)[0]
    # probability
    probabilities = model.predict_proba(text_vector)[0]
    return prediction, probabilities
