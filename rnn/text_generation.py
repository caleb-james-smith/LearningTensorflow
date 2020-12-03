# text_generation.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.callbacks import Callback
from tools import gpu_allow_mem_grow, show_generated_text, sample_from_model

# manage GPU memory
gpu_allow_mem_grow()

model_name = "dissertation"

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

class SamplerCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        # show generated text
        generated_text = sample_from_model(self.model, 100, corpus, data_size, sentence_length, vocab_size, char_to_idx, idx_to_char)
        show_generated_text(generated_text)
        # save model
        this_epoch = epoch + 1
        if (this_epoch % 5 == 0):
            self.model.save("models/{0}_{1:03d}.h5".format(model_name, this_epoch))

# create model
sampler_callback = SamplerCallback()
model = Sequential()
model.add(LSTM(256, input_shape=(sentence_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.fit(x, y, epochs=20, batch_size=256, callbacks=[sampler_callback])
model.fit(x, y, epochs=1, batch_size=256, callbacks=[sampler_callback])

generated_text = sample_from_model(model, 1000, corpus, data_size, sentence_length, vocab_size, char_to_idx, idx_to_char)
show_generated_text(generated_text)

