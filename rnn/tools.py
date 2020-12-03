# tools.py
import numpy as np
import tensorflow as tf
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

def show_generated_text(generated_text):
    print('\nGenerated Text')
    print('-' * 32)
    print(generated_text)

def sample_from_model(model, sample_length, corpus, data_size, sentence_length, vocab_size, char_to_idx, idx_to_char):
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

