import numpy as np

def document_vector(tokens, embedding_dict):

    vectors = []

    for word in tokens:

        if word in embedding_dict:
            vectors.append(embedding_dict[word])

        elif word.lower() in embedding_dict:
            vectors.append(embedding_dict[word.lower()])

    if len(vectors) == 0:
        return np.zeros(300, dtype=np.float32)

    return np.mean(vectors, axis=0)