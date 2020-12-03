# text_generation.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.callbacks import Callback
from random import randint

# manage GPU memory
def gpu_allow_mem_grow():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("GPU list: {0}".format(gpus))
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("ERROR: gpu_mem_grow failed: ",e)

gpu_allow_mem_grow()

#data_file = 'data/sonnets.txt'
#data_file = 'data/kernel.c'
data_file = 'data/dissertation_zinv.tex'

with open(data_file, 'r') as in_file:
    corpus = in_file.read()

# get unique characters
chars = list(set(corpus))
print("chars: {0}".format(chars))

data_size, vocab_size = len(corpus), len(chars)
print("data size: {0}, vocab size: {1}".format(data_size, vocab_size))

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
print("num sentences: {0}".format(num_sentences))

# use one-hot character encoding

x = np.zeros((num_sentences, sentence_length, vocab_size), dtype=np.bool)
y = np.zeros((num_sentences, vocab_size), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x[i, j, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1


def sample_from_model(model, sample_length=100):
    # starting seed
    seed = randint(0, data_size - sentence_length)
    seed_sentence = corpus[seed: seed + sentence_length]
    x_pred = np.zeros((1, sentence_length, vocab_size), dtype=np.bool)
    for j, char in enumerate(seed_sentence):
        x_pred[0, j, char_to_idx[char]] = 1
    # generate text
    generated_text = ''
    for i in range(sample_length):
        prediction = np.argmax(model.predict(x_pred))
        generated_text += idx_to_char[prediction]
        # remove first character from x_pred and add new character to x_pred
        activation = np.zeros((1, 1, vocab_size), dtype=np.bool)
        activation[0, 0, prediction] = 1
        x_pred = np.concatenate((x_pred[:, 1:, :], activation), axis=1)

    return generated_text

def show_generated_text(generated_text):
    print('\nGenerated Text')
    print('-' * 32)
    print(generated_text)

class SamplerCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        generated_text = sample_from_model(self.model)
        show_generated_text(generated_text)

# create model
sampler_callback = SamplerCallback()
model = Sequential()
model.add(LSTM(256, input_shape=(sentence_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.fit(x, y, epochs=20, batch_size=256, callbacks=[sampler_callback])
model.fit(x, y, epochs=20, batch_size=256, callbacks=[sampler_callback])

generated_text = sample_from_model(model, sample_length=1000)
show_generated_text(generated_text)

