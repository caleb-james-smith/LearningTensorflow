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



