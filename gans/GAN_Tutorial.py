from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import math
import os

# problem on mac regarding OpenMP:
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
# https://github.com/dmlc/xgboost/issues/1715
# fix:
# As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results.
# For more information, please see http://openmp.llvm.org/
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# load data
(x_train, _), (_, _) = mnist.load_data()

# constants
MID_POINT   = 255.0 / 2.0           # midpoint of pixel values used for normalization
PIX_LENGTH  = 28                    # images are 28 x 28 pixels
NUM_PIXELS  = PIX_LENGTH ** 2       # total number of pixels in image
NUM_EPOCHS  = 100
BATCH_SIZE  = 256
HALF_BATCH  = int(BATCH_SIZE / 2)
NOISE_DIM   = 100                   # dimension of noise vector
TRAIN_SIZE  = x_train.shape[0]      # number of training data points
BATCH_COUNT = math.ceil(TRAIN_SIZE / float(BATCH_SIZE))

# format data
# normalize input to be in the range [1, 1]
x_train = (x_train.astype(np.float32) - MID_POINT) / MID_POINT
# reshape data using data size
x_train = x_train.reshape(TRAIN_SIZE, -1)

# optimizer
adam = Adam(lr=2e-4, beta_1=0.5)

# --- generator --- #
generator = Sequential()
# alternate dense layers and leaky relu activation function
generator.add(Dense(256, input_shape=(NOISE_DIM,)))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
# last layer: use tanh as input range is [-1, 1]
generator.add(Dense(NUM_PIXELS, activation='tanh'))
# compile model
generator.compile(loss='binary_crossentropy', optimizer=adam)

# --- discriminator --- #
discriminator = Sequential()
discriminator.add(Dense(512, input_shape=(NUM_PIXELS,)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
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
    noise = np.random.normal(0, 1, size=[num_examples, NOISE_DIM])
    generated_imgs = generator.predict(noise)
    generated_imgs = generated_imgs.reshape(num_examples, PIX_LENGTH, PIX_LENGTH)

    # plot images
    plt.figure(figsize=(10, 10))
    for i in range(num_examples):
        plt.subplot(10, 10, i + 1)
        plt.imshow(generated_imgs[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_generated_epoch_{0:03d}.png'.format(epoch))


# training
def train(num_epochs):
    # make directories if they do not exist
    dir_list = ["images", "models"]
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
    # loop over epochs
    for epoch in range(1, num_epochs + 1):
        discriminator_loss_epoch = 0.0
        generator_loss_epoch     = 0.0
        
        for batch in range(BATCH_COUNT):
            # --- train discriminator --- #
            # - generator is frozen, modify parameters for discriminator
            
            # randomly select training images 
            indices = np.random.randint(0, TRAIN_SIZE, HALF_BATCH)
            images  = x_train[indices]
            # generate equal number of images
            noise = np.random.normal(0, 1, size=[HALF_BATCH, NOISE_DIM])
            generated_imgs = generator.predict(noise)
            # to label real images, use 0.9 instead of 1.0 for label smoothing
            real_y = 0.9 * np.ones((HALF_BATCH, 1))
            fake_y =       np.zeros((HALF_BATCH, 1))
            
            # train_on_batch() updates model parameters
            discriminator_loss_real  =  discriminator.train_on_batch(images, real_y)
            discriminator_loss_fake  =  discriminator.train_on_batch(generated_imgs, fake_y)
            discriminator_loss_batch =  np.average(discriminator_loss_real + discriminator_loss_fake)
            discriminator_loss_epoch += discriminator_loss_batch
            
            # --- train generator --- #
            # - discriminator is frozen, modify parameters for generator
            noise = np.random.normal(0, 1, size=[BATCH_SIZE, NOISE_DIM])
            real_y = np.ones((BATCH_SIZE, 1))
            # train_on_batch() updates model parameters
            generator_loss_batch =  combined.train_on_batch(noise, real_y)
            generator_loss_epoch += generator_loss_batch
    
    
        print("epoch: {0}, discriminator loss: {1}, generator loss: {2}".format(epoch, discriminator_loss_epoch / BATCH_COUNT, generator_loss_epoch / BATCH_COUNT))
        if (epoch % 5 == 0):
            generator.save('models/gan_generator_{0:03d}.h5'.format(epoch))
            save_imgs(epoch)


if __name__ == "__main__":
    train(NUM_EPOCHS)


