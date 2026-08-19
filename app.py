import streamlit as st
import skops.io as sio
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import altair as alt
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from utils.preprocessing import preprocess
from utils.predict_svm import predict_svm
from utils.predict_lstm import predict_lstm
from utils.document_vector import document_vector

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Financial Sentiment Analysis")

st.markdown("""
Aplikasi analisis sentimen keuangan menggunakan:
- BoW + SVM
- TF-IDF + SVM
- Word2Vec + SVM
- FastText + SVM
- Word2Vec + LSTM
- FastText + LSTM
""")

# =========================
# MAX LENGTH
# =========================
max_len = 100

# =========================
# MODEL SELECTION
# =========================
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

# =========================
# TEXT INPUT
# =========================
user_input = st.text_area(
    "Masukkan Kalimat Finansial",
    height=150,
    placeholder="Example: The company reported strong quarterly earnings growth"
)

# =========================
# PREDICTION
# =========================
if st.button("Prediksi Sentimen"):

    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu")

    else:

        # =========================
        # PREPROCESSING
        # =========================
        clean_text = preprocess(user_input)
        st.subheader("Hasil Preprocessing")
        st.code(clean_text)

        # =====================================================
        # BOW + SVM
        # =====================================================
        if model_option == "BoW + SVM":

            unknown_types = sio.get_untrusted_types(
                file="models/bow_vectorizer.skops"
            )
            vectorizer = sio.load(
                "models/bow_vectorizer.skops",
                trusted=unknown_types
            )

            unknown_types = sio.get_untrusted_types(
                file="models/bow_svm.skops"
            )
            print(unknown_types)
            model = sio.load(
                "models/bow_svm.skops",
                trusted=unknown_types
            )

            sentiment, probabilities = predict_svm(
                clean_text,
                model,
                vectorizer
            )

        # =====================================================
        # TF-IDF + SVM
        # =====================================================
        elif model_option == "TF-IDF + SVM":

            unknown_types = sio.get_untrusted_types(
                file="models/tfidf_vectorizer.skops"
            )
            vectorizer = sio.load(
                "models/tfidf_vectorizer.skops",
                trusted=unknown_types
            )

            unknown_types = sio.get_untrusted_types(
                file="models/tfidf_svm.skops"
            )
            print(unknown_types)
            model = sio.load(
                "models/tfidf_svm.skops",
                trusted=unknown_types
            )

            sentiment, probabilities = predict_svm(
                clean_text,
                model,
                vectorizer
            )

        # =====================================================
        # WORD2VEC + SVM
        # =====================================================
        elif model_option == "Word2Vec + SVM":

            # Load embedding dictionary
            data = np.load(
                "models/w2v_embedding_dict.npz",
                allow_pickle=False
            )
            w2v_dict = {
                word: vector
                for word, vector in zip(
                    data["words"],
                    data["vectors"]
                )
            }
            
            unknown_types = sio.get_untrusted_types(
                file="models/w2v_svm.skops"
            )
            print(unknown_types)
            model = sio.load(
                "models/w2v_svm.skops",
                trusted=unknown_types
            )

            tokens = clean_text.split()
            
            # Average Embedding
            doc_vector = document_vector(
                tokens,
                w2v_dict
            )
            
            # Shape (1, 300)
            doc_vector = np.asarray(
                doc_vector,
                dtype=np.float32
            ).reshape(1, -1)
            
            sentiment, probabilities = predict_svm(
                doc_vector,
                model
            )

        # =====================================================
        # FASTTEXT + SVM
        # =====================================================
        elif model_option == "FastText + SVM":
            
            data = np.load(
                "models/ft_embedding_dict.npz",
                allow_pickle=False
            )
            ft_dict = {
                word: vector
                for word, vector in zip(
                    data["words"],
                    data["vectors"]
                )
            }
            
            unknown_types = sio.get_untrusted_types(
                file="models/ft_svm.skops"
            )
            print(unknown_types)
            model = sio.load(
                "models/ft_svm.skops",
                trusted=unknown_types
            )
            
            # Sentence embedding (average pooling)
            tokens = clean_text.split()

            doc_vector = document_vector(
                tokens,
                ft_dict
            )
            
            # Shape (1, 300)
            doc_vector = np.asarray(
                doc_vector,
                dtype=np.float32
            ).reshape(1, -1)
            
            sentiment, probabilities = predict_svm(
                doc_vector,
                model
            )
        
        # =====================================================
        # WORD2VEC + LSTM
        # =====================================================
        elif model_option == "Word2Vec + LSTM":

            # LOAD TOKENIZER
            with open(
                "models/w2v_tokenizer.json",
                "r"
            ) as f:
                tokenizer = tokenizer_from_json(
                    f.read()
                )

            # LOAD MODEL
            model = tf.keras.models.load_model(
                "models/w2v_lstm.keras"
            )

            # LOAD LABELS
            with open(
                "models/label_classes.json",
                "r"
            ) as f:
                label_classes = json.load(f)

            # PREDICTION
            sentiment, probabilities = predict_lstm(
                clean_text,
                tokenizer,
                model,
                label_classes,
                max_len
            )

        # =====================================================
        # FASTTEXT + LSTM
        # =====================================================
        elif model_option == "FastText + LSTM":

            # LOAD TOKENIZER
            with open(
                "models/ft_tokenizer.json",
                "r"
            ) as f:
                tokenizer = tokenizer_from_json(
                    f.read()
                )

            # LOAD MODEL
            model = tf.keras.models.load_model(
                "models/ft_lstm.keras"
            )
            
            # LOAD LABELS
            with open(
                "models/label_classes.json",
                "r"
            ) as f:
                label_classes = json.load(f)

            # PREDICTION
            sentiment, probabilities = predict_lstm(
                clean_text,
                tokenizer,
                model,
                label_classes,
                max_len
            )

        # =========================
        # OUTPUT
        # =========================
        st.subheader("Hasil Prediksi")

        st.success(
            f"Predicted Sentiment: {sentiment.upper()}"
        )

        # =========================
        # CONFIDENCE SCORE
        # =========================
        st.subheader("Confidence Score")

        # LABELS
        if "SVM" in model_option:

            labels = [
                "negative",
                "neutral",
                "positive"
            ]

        else:

            with open(
                "models/label_classes.json",
                "r"
            ) as f:

                labels = json.load(f)


        # DATAFRAME
        prob_df = pd.DataFrame({
            "Sentiment": labels,
            "Probability": (probabilities * 100).round(2)
        })
        
        st.dataframe(prob_df)

        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Sentiment:N",
                    sort=["negative", "neutral", "positive"],
                    axis=alt.Axis(
                        labelAngle=0,
                        title="Sentiment"
                    )
                ),
                y=alt.Y(
                    "Probability:Q",
                    title="Probability (%)",
                    scale=alt.Scale(domain=[0, 100])
                ),
                color=alt.Color(
                    "Sentiment:N",
                    scale=alt.Scale(
                        domain=["negative", "neutral", "positive"],
                        range=["#EF4444", "#FACC15", "#22C55E"]
                    ),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip(
                        "Sentiment:N",
                        title="Sentiment"
                    ),
                    alt.Tooltip(
                        "Probability:Q",
                        title="Probability",
                        format=".2f"
                    )
                ]
            )
            .properties(
                height=350
            )
        )
        st.altair_chart(chart, use_container_width=True)
