import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_lstm(
    text,
    tokenizer,
    model,
    label_classes,
    max_len
):

    # TEXT TO SEQUENCE
    sequence = tokenizer.texts_to_sequences([text])

    print("TEXT:", text)
    print("SEQUENCE:", sequence)

    # PADDING
    padded = pad_sequences(
        sequence,
        maxlen=max_len,
        padding='pre',
        truncating='pre'
    )

    print("PADDED SHAPE:", padded.shape)
    print("PADDED:", padded)

    # RAW PREDICTION
    prediction = model.predict(
        padded,
        verbose=0
    )

    print("RAW PREDICTION:", prediction)

    predicted_index = np.argmax(prediction[0])

    sentiment = label_classes[predicted_index]

    probabilities = prediction[0]

    return sentiment, probabilities