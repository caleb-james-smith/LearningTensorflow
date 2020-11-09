from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Activation, Embedding, multiply, UpSampling2D, Conv2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import math
import sys
import numpy as np

(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train[..., np.newaxis]

#NUM_EPOCHS = 50
NUM_EPOCHS = 1
BATCH_SIZE = 256
BATCH_COUNT = math.ceil(X_train.shape[0] / BATCH_SIZE)
NOISE_DIM = 100
HALF_BATCH = int(BATCH_SIZE / 2)

adam = Adam(lr=2e-4, beta_1=0.5)

generator = Sequential()
generator.add(Dense(7 * 7 * 128, input_shape=(NOISE_DIM,)))
generator.add(Reshape((7, 7, 128)))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
# automatically doubles activation map size
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)
# be careful about the resulting generated image's dimensions

discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator.trainable = False
gan_input = Input(shape=(NOISE_DIM,))
generated_img = generator(gan_input)
gan_output = discriminator(generated_img)
combined = Model(gan_input, gan_output)
combined.compile(loss='binary_crossentropy', optimizer=adam)

def save_imgs(epoch, num_examples=100):
    noise = np.random.normal(0, 1, size=[num_examples, NOISE_DIM])
    generated_imgs = generator.predict(noise)
    generated_imgs = generated_imgs.reshape(num_examples, 28, 28)

    plt.figure(figsize=(10,10))
    for i in range(num_examples):
        plt.subplot(10, 10, i + 1)
        plt.imshow(generated_imgs[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_image_epoch_{0}.png'.format(epoch + 1))

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
        generated_imgs = generator.predict(noise)

        # use one-sided label smoothing
        real_y = np.ones((HALF_BATCH, 1)) * 0.9
        fake_y = np.zeros((HALF_BATCH, 1))

        # train on real and fake images
        d_loss_real = discriminator.train_on_batch(imgs, real_y)
        d_loss_fake = discriminator.train_on_batch(generated_imgs, fake_y)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        epoch_d_loss += d_loss

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
        real_y = np.ones((BATCH_SIZE, 1))
        g_loss = combined.train_on_batch(noise, real_y)

        # fold loss into cumulative moving average
        epoch_g_loss += g_loss

    print("%d [D loss: %f] [G loss: %f]" % ((epoch + 1), epoch_d_loss / BATCH_COUNT, epoch_g_loss / BATCH_COUNT))

    if (epoch + 1) % 10 == 0:
        generator.save('models/dcgan_generator_{0}.h5'.format(epoch + 1))
        save_imgs(epoch)
