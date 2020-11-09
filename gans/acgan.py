from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Activation, Embedding, multiply, UpSampling2D, Conv2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import math
import sys
import numpy as np

(X_train, y_train), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train[..., np.newaxis]
y_train = y_train.reshape(-1, 1)

NUM_CLASSES = 10
#NUM_EPOCHS = 50
NUM_EPOCHS = 1
BATCH_SIZE = 256
BATCH_COUNT = math.ceil(X_train.shape[0] / BATCH_SIZE)
NOISE_DIM = 100
HALF_BATCH = int(BATCH_SIZE / 2)

adam = Adam(lr=2e-4, beta_1=0.5)
losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

noise = Input(shape=(NOISE_DIM,))
label = Input(shape=(1,), dtype='int32')
label_embedding = Flatten()(Embedding(NUM_CLASSES, NOISE_DIM)(label))
generator_input = multiply([noise, label_embedding])

x = Dense(7 * 7 * 128)(generator_input)
x = Reshape((7, 7, 128))(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = UpSampling2D()(x)
x = Conv2D(64, kernel_size=(5, 5), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = UpSampling2D()(x)
generator_output = Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh')(x)

generator = Model([noise, label], generator_output)

generator.compile(loss='binary_crossentropy', optimizer=adam)

img = Input(shape=(28, 28, 1))
x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1))(img)
x = LeakyReLU(0.2)(x)
x = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = Flatten()(x)
validity = Dense(1, activation='sigmoid')(x)
class_label = Dense(NUM_CLASSES + 1, activation='softmax')(x)
# another class for fake class

discriminator = Model(img, [validity, class_label])

discriminator.compile(loss=losses, optimizer=adam)

discriminator.trainable = False
noise = Input(shape=(NOISE_DIM,))
label = Input(shape=(1,), dtype='int32')
generated_img = generator([noise, label])
valid, target_label = discriminator(generated_img)
combined = Model([noise, label], [valid, target_label])
combined.compile(loss=losses, optimizer=adam)

def save_imgs(epoch, num_examples=100):
    noise = np.random.normal(0, 1, size=[num_examples, NOISE_DIM])
    sampled_labels = np.random.randint(0, 10, num_examples).reshape(-1, 1)

    generated_imgs = generator.predict([noise, sampled_labels])
    generated_imgs = generated_imgs.reshape(num_examples, 28, 28)

    plt.figure(figsize=(10,10))
    for i in range(num_examples):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_imgs[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/acgan_generated_image_epoch_{0}.png'.format(epoch + 1))

for epoch in range(NUM_EPOCHS):
    epoch_d_loss = 0.
    epoch_g_loss = 0.

    for step in range(BATCH_COUNT):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # randomly select half a batch of real images
        idx = np.random.randint(0, X_train.shape[0], HALF_BATCH)
        imgs = X_train[idx]

        # generate half a batch of fake images
        noise = np.random.normal(0, 1, (HALF_BATCH, NOISE_DIM))
        sampled_labels = np.random.randint(0, 10, HALF_BATCH).reshape(-1, 1)
        generated_imgs = generator.predict([noise, sampled_labels])

        # use one-sided label smoothing
        real_valid_y = np.ones((HALF_BATCH, 1)) * 0.9
        fake_valid_y = np.zeros((HALF_BATCH, 1))

        real_class_labels = y_train[idx]
        # real classes are [0, NUM_CLASSES-1]
        fake_class_labels = NUM_CLASSES * np.ones(HALF_BATCH).reshape(-1, 1)

        # train on real and fake images
        d_loss_real = discriminator.train_on_batch(imgs, [real_valid_y, real_class_labels])
        d_loss_fake = discriminator.train_on_batch(generated_imgs, [fake_valid_y, fake_class_labels])
        # use combined loss
        d_loss = 0.5 * (np.array(d_loss_real) + np.array(d_loss_fake))

        epoch_d_loss += np.mean(d_loss)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
        real_valid_y = np.ones((BATCH_SIZE, 1))
        sampled_labels = np.random.randint(0, 10, BATCH_SIZE).reshape(-1, 1)
        g_loss = combined.train_on_batch([noise, sampled_labels], [real_valid_y, sampled_labels])

        # fold loss into cumulative moving average
        # # use combined loss
        epoch_g_loss += np.mean(g_loss)

    print("%d [D loss: %f] [G loss: %f]" % ((epoch + 1), epoch_d_loss / BATCH_COUNT, epoch_g_loss / BATCH_COUNT))

    if (epoch + 1) % 10 == 0:
        generator.save('models/acgan_generator_{0}.h5'.format(epoch + 1))
        save_imgs(epoch)
