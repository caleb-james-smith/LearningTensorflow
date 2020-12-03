# tools.py

import tensorflow as tf

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

