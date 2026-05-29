import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50

def predict_lstm(text, tokenizer, model, label_encoder):
    # text to sequence
    sequence = tokenizer.texts_to_sequences([text])
    # padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post'
    )
    # prediction
    prediction = model.predict(
        padded,
        verbose=0
    )
    # ambil index probabilitas tertinggi
    predicted_index = np.argmax(prediction)
    # decode label
    sentiment = label_encoder.inverse_transform(
        [predicted_index]
    )[0]
    # probability
    probabilities = prediction[0]
    return sentiment, probabilities
