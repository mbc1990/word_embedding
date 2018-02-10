"""
Testing out Keras's word embedding feature
"""

import csv
from nltk import word_tokenize
from keras.models import Sequential
from keras.layers import Embedding, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 

def get_documents():
    docs = []
    with open('raw_dump.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            text = r[3]
            docs.append(text)
    return docs


def make_embedding(texts):
    MAX_NUM_WORDS = 10000
    MAX_SEQUENCE_LENGTH = 500
    EMBEDDING_VECTOR_LENGTH = 300
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Map of string -> integer representation 
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    model = Sequential()
    model.add(Embedding(len(word_index), EMBEDDING_VECTOR_LENGTH, input_length=MAX_SEQUENCE_LENGTH))
    model.compile('rmsprop', 'mse')

    # Embedding matrix
    output_array = model.predict(data)
    return output_array


def main():
    docs = get_documents()
    embedding = make_embedding(docs)
    print embedding

if __name__ == "__main__":
    main()
