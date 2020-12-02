# movie_sentiment_analysis.py
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

VOCAB_SIZE  = 5000
INDEX_FROM  = 3
MAX_SEQ_LEN = 128
EMBEDDING_DIM = 64

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE, index_from=INDEX_FROM)

word_to_idx = imdb.get_word_index()
idx_to_word = {v+INDEX_FROM:k for k,v in word_to_idx.items()}
idx_to_word[0] = '<PAD>'
idx_to_word[1] = '<START>'
idx_to_word[2] = '<UNK>'

# print for testing
#print(x_train[0])
#print(' '.join(idx_to_word[idx] for idx in x_train[0]))
#print(word_to_idx)
#print("training size: {0}".format(len(x_train)))
#print("testing size: {0}".format(len(x_test)))
#print("dictionary size: {0}".format(len(word_to_idx)))
#print(10*'-')
#for i in range(10):
#    print(len(x_train[i]))
#
#print(' '.join(idx_to_word[idx] for idx in x_train[0]))

x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQ_LEN)
x_test  = sequence.pad_sequences(x_test,  maxlen=MAX_SEQ_LEN)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN))
model.add(LSTM(128))
#model.add(LSTM(128, return_sequences=True))
#model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir='sentiment_logs')

model.fit(x_train, y_train, epochs=2, batch_size=256, validation_split=0.2, callbacks=[tensorboard])

score = model.evaluate(x_test, y_test)
print("Accuracy: {0:.4f}".format(score[1]))


