import numpy as np

def document_vector(tokens, model):
    valid_words = [
        word for word in tokens
        if word in model.key_to_index
    ]
    if len(valid_words) == 0:
        return np.zeros(model.vector_size)
    return np.mean(
        model[valid_words],
        axis=0
    )
