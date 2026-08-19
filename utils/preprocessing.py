import re

def preprocess(text):
    # lowercasing
    text = text.lower()
    # normalize contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # remove hashtags
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    # Remove punctuation & numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text