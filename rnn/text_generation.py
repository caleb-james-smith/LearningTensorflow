# text_generation.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.callbacks import Callback
from random import randint

with open('sonnets.txt', 'r') as in_file:
    corpus = in_file.read()

# get unique characters
chars = list(set(corpus))
#print(chars)

data_size, vocab_size = len(corpus), len(chars)
#print(data_size, vocab_size)

idx_to_char = {i:c for i,c in enumerate(chars)}
char_to_idx = {c:i for i,c in enumerate(chars)}

#method: given a sentence (50 chars), predict next char
sentence_length = 50
sentences       = []
next_chars      = []
for i in range(data_size - sentence_length):
    sentences.append(corpus[i:i+sentence_length])
    next_chars.append(corpus[i+sentence_length])

num_sentences = len(sentences)
#print(num_sentences)

# use one-hot character encoding

x = np.zeros((num_sentences, sentence_length, vocab_size), dtype=np.bool)
y = np.zeros((num_sentences, vocab_size), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x[i, j, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# create model
model = Sequential()
model.add(LSTM(256, input_shape=(sentence_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x, y, epochs=2, batch_size=256)


