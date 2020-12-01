# movie_sentiment_analysis.py
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

VOCAB_SIZE = 5000
INDEX_FROM = 3

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE, index_from=INDEX_FROM)

word_to_idx = imdb.get_word_index()
idx_to_word = {v+INDEX_FROM:k for k,v in word_to_idx.items()}
idx_to_word[0] = '<PAD>'
idx_to_word[1] = '<START>'
idx_to_word[2] = '<UNK>'

print(x_train[0])
print(' '.join(idx_to_word[idx] for idx in x_train[0]))
#print(word_to_idx)
#print("training size: {0}".format(len(x_train)))
#print("testing size: {0}".format(len(x_test)))
#print("dictionary size: {0}".format(len(word_to_idx)))

#for i in range(10):
#    print("{0} : {1}".format(i, idx_to_word[i]))


