import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from utils.preprocessing import preprocess_text
from utils.predict_svm import predict_svm
from utils.predict_lstm import predict_lstm
from utils.embedding_utils import document_vector

# PAGE CONFIG
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="centered"
)

# TITLE
st.title("📈 Financial Sentiment Analysis")
st.markdown(
    """
    Aplikasi analisis sentimen keuangan menggunakan:

    - BoW + SVM
    - TF-IDF + SVM
    - Word2Vec + SVM
    - FastText + SVM
    - Word2Vec + LSTM
    - FastText + LSTM
    """
)

# LOAD LABEL ENCODER
label_encoder = joblib.load(
    'models/label_encoder.pkl'
)

# MODEL SELECTION
model_option = st.selectbox(
    "Pilih Model",
    [
        "BoW + SVM",
        "TF-IDF + SVM",
        "Word2Vec + SVM",
        "FastText + SVM",
        "Word2Vec + LSTM",
        "FastText + LSTM"
    ]
)

# TEXT INPUT
user_input = st.text_area(
    "Masukkan Kalimat Financial",
    height=150,
    placeholder="Example: The company reported strong quarterly earnings growth"
)

# PREDICT BUTTON
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning(
            "Silakan masukkan teks terlebih dahulu"
        )
    else:
        # PREPROCESSING
        clean_text = preprocess(user_input)
        st.subheader("Preprocessing Result")
        st.code(clean_text)
        # BOW + SVM
        if model_option == "BoW + SVM":
            vectorizer = joblib.load(
                'models/bow_vectorizer.pkl'
            )
            model = joblib.load(
                'models/bow_svm.pkl'
            )
            sentiment, probabilities = predict_svm(
                clean_text,
                model,
                vectorizer
            )
        # TF-IDF + SVM
        elif model_option == "TF-IDF + SVM":
            vectorizer = joblib.load(
                'models/tfidf_vectorizer.pkl'
            )
            model = joblib.load(
                'models/tfidf_svm.pkl'
            )
            sentiment, probabilities = predict_svm(
                clean_text,
                model,
                vectorizer
            )
        # WORD2VEC + SVM
        elif model_option == "Word2Vec + SVM":
            # load pretrained word2vec
            w2v_model = KeyedVectors.load_word2vec_format(
                'models/word2vec.bin',
                binary=True
            )
            # load svm
            model = joblib.load(
                'models/w2v_svm.pkl'
            )
            # tokenize
            tokens = clean_text.split()
            # document vector
            doc_vector = document_vector(
                tokens,
                w2v_model
            )
            sentiment, probabilities = predict_svm(
                doc_vector,
                model
            )
        # FASTTEXT + SVM                                 
        elif model_option == "FastText + SVM":
            # load pretrained fasttext
            ft_model = load_facebook_vectors(
                'models/fasttext.bin'
            )
            # load svm
            model = joblib.load(
                'models/ft_svm.pkl'
            )
            # tokenize
            tokens = clean_text.split()
            # document vector
            doc_vector = document_vector(
                tokens,
                ft_model
            )
            sentiment, probabilities = predict_svm(
                doc_vector,
                model
            )
        # WORD2VEC + LSTM                           
        elif model_option == "Word2Vec + LSTM":
            tokenizer = joblib.load(
                'models/w2v_tokenizer.pkl'
            )
            model = tf.keras.models.load_model(
                'models/w2v_lstm.keras'
            )
            sentiment, probabilities = predict_lstm(
                clean_text,
                tokenizer,
                model,
                label_encoder
            )
        # FASTTEXT + LSTM
        elif model_option == "FastText + LSTM":
            tokenizer = joblib.load(
                'models/ft_tokenizer.pkl'
            )
            model = tf.keras.models.load_model(
                'models/ft_lstm.keras'
            )
            sentiment, probabilities = predict_lstm(
                clean_text,
                tokenizer,
                model,
                label_encoder
            )
  
        # OUTPUT
        st.subheader("Prediction Result")
        st.success(
            f"Predicted Sentiment: {sentiment.upper()}"
        )
      
        # CONFIDENCE SCORE
        st.subheader("Confidence Score")

        # label untuk svm
        if model_option in [
            "BoW + SVM",
            "TF-IDF + SVM",
            "Word2Vec + SVM",
            "FastText + SVM"
        ]:
            labels = [
                'negative',
                'neutral',
                'positive'
            ]
        # label untuk lstm
        else:
            labels = label_encoder.classes_
        prob_df = pd.DataFrame({
            'Sentiment': labels,
            'Probability': (
                probabilities * 100
            ).round(2)
        })

        st.dataframe(prob_df)
      
        st.bar_chart(
            prob_df.set_index('Sentiment')
        )
