from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import math

# load data
(x_train, _), (_, _) = mnist.load_data()

# constants
MID_POINT   = 255.0 / 2.0           # midpoint of pixel values used for normalization
PIX_LENGTH  = 28                    # images are 28 x 28 pixels
GEN_LENGTH  = int(PIX_LENGTH/4)     # length used for generator
NUM_PIXELS  = PIX_LENGTH ** 2       # total number of pixels in image
#NUM_EPOCHS  = 50
NUM_EPOCHS  = 1
BATCH_SIZE  = 256
HALF_BATCH  = int(BATCH_SIZE / 2)
NOISE_DIM   = 100                   # dimension of noise vector
TRAIN_SIZE  = x_train.shape[0]      # number of training data points
BATCH_COUNT = math.ceil(TRAIN_SIZE / float(BATCH_SIZE))

# format data
# normalize input to be in the range [1, 1]
x_train = (x_train.astype(np.float32) - MID_POINT) / MID_POINT
# reshape data using data size
x_train = x_train[..., np.newaxis]

# optimizer
adam = Adam(lr=2e-4, beta_1=0.5)

print("length for generator: {0}".format(GEN_LENGTH))
print("vector length: {0}".format(GEN_LENGTH * GEN_LENGTH * 128))

# --- generator --- #
# input is vector (1D): length is 7 * 7 * 128
# output is image (2D): dimensions are 28 x 28 x 1 
# to go from vector to image, use reshape (opposite of flatten)
# after each leaky relu activation function, apply batch normalization
generator = Sequential()
# start with vector
generator.add(Dense(GEN_LENGTH * GEN_LENGTH * 128, input_shape=(NOISE_DIM,)))
# reshape vector into 7 x 7 x 128 image (7x7 image, 128 filters)
generator.add(Reshape((GEN_LENGTH, GEN_LENGTH, 128)))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
# UpSampling2D() doubles image dimensions by default to get 14 x 14 x 128
generator.add(UpSampling2D())
# use Conv2D() to get to 14 x 14 x 64 (reduce filters by factor of 2)
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
# UpSampling2D() doubles image dimensions by default to get 28 x 28 x 64
generator.add(UpSampling2D())
# use Conv2D() to get to 28 x 28 x 1
# last layer: use tanh as input range is [-1, 1]
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
# compile model
generator.compile(loss='binary_crossentropy', optimizer=adam)

# --- discriminator --- #
# standard CNN
# input is 28 x 28 x 1 input (same dimensions as output of generator)
# output is single node with value in range [0, 1]
# avoid sparse operations
# use stride instead of max pooling
# use leaky relu instead of relu
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding='same', input_shape=(PIX_LENGTH, PIX_LENGTH, 1)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
# last layer: use sigmoid as discriminator outputs probability [0, 1] for fake/real
discriminator.add(Dense(1, activation='sigmoid'))
# compile model
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# --- combined model --- #
# freeze discriminator when training generator
discriminator.trainable = False
gan_input = Input(shape=(NOISE_DIM,))
generated_img = generator(gan_input)
gan_output = discriminator(generated_img)
combined = Model(gan_input, gan_output)
combined.compile(loss='binary_crossentropy', optimizer=adam)


# function to save images after given training epoch 
def save_imgs(epoch, num_examples=100):
    noise = np.random.normal(0, 1, (num_examples, NOISE_DIM))
    generated_imgs = generator.predict(noise)
    generated_imgs = generated_imgs.reshape(num_examples, PIX_LENGTH, PIX_LENGTH)

    # plot images
    plt.figure(figsize=(10, 10))
    for i in range(num_examples):
        plt.subplot(10, 10, i + 1)
        plt.imshow(generated_imgs[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_epoch_{0:03d}.png'.format(epoch))


# training
def train(num_epochs):
    # loop over epochs
    for epoch in range(1, num_epochs + 1):
        discriminator_loss_epoch = 0.0
        generator_loss_epoch     = 0.0
        
        for batch in range(BATCH_COUNT):
            # --- train discriminator --- #
            # - generator is frozen, modify parameters for discriminator
            
            # randomly select half a batch of real training images 
            indices = np.random.randint(0, TRAIN_SIZE, HALF_BATCH)
            images  = x_train[indices]
            # generate half a batch of fake images
            noise = np.random.normal(0, 1, (HALF_BATCH, NOISE_DIM))
            generated_imgs = generator.predict(noise)
            # to label real images, use 0.9 instead of 1.0 for label smoothing
            real_y = 0.9 * np.ones((HALF_BATCH, 1))
            fake_y =       np.zeros((HALF_BATCH, 1))
            
            # train_on_batch() updates model parameters
            discriminator_loss_real  =  discriminator.train_on_batch(images, real_y)
            discriminator_loss_fake  =  discriminator.train_on_batch(generated_imgs, fake_y)
            discriminator_loss_batch =  0.5 * (discriminator_loss_real + discriminator_loss_fake)
            discriminator_loss_epoch += discriminator_loss_batch
            
            # --- train generator --- #
            # - discriminator is frozen, modify parameters for generator
            noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
            real_y = np.ones((BATCH_SIZE, 1))
            # train_on_batch() updates model parameters
            generator_loss_batch =  combined.train_on_batch(noise, real_y)
            generator_loss_epoch += generator_loss_batch
    
    
        print("epoch: {0}, discriminator loss: {1}, generator loss: {2}".format(epoch, discriminator_loss_epoch / BATCH_COUNT, generator_loss_epoch / BATCH_COUNT))
        if (epoch % 10 == 0):
            generator.save('models/dcgan_generator_{0:03d}.h5'.format(epoch))
            save_imgs(epoch)


if __name__ == "__main__":
    train(NUM_EPOCHS)

