from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Embedding, multiply, UpSampling2D, Conv2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import math

# load data
(x_train, y_train), (_, _) = mnist.load_data()

# constants
MID_POINT   = 255.0 / 2.0           # midpoint of pixel values used for normalization
PIX_LENGTH  = 28                    # images are 28 x 28 pixels
GEN_LENGTH  = int(PIX_LENGTH/4)     # length used for generator
NUM_PIXELS  = PIX_LENGTH ** 2       # total number of pixels in image
NUM_CLASSES = 10
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
y_train = y_train.reshape(-1, 1)

# optimizer
adam = Adam(lr=2e-4, beta_1=0.5)
# loss functions for generator/discriminator (real/fake) and classifcation into number categories
# here sparse_categorical_crossentropy means that each image corresponds to exactly one class
losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
noise = Input(shape=(NOISE_DIM,))
# not using "one-hot" for labels... one number instead
label = Input(shape=(1,), dtype='int32')
# combine noise and label: turn label into same dimension as noise
label_embedding = Flatten()(Embedding(NUM_CLASSES, NOISE_DIM)(label)) 
generator_input = multiply([noise, label_embedding])

# --- generator --- #
x = Dense(GEN_LENGTH * GEN_LENGTH * 128)(generator_input)
x = Reshape((GEN_LENGTH, GEN_LENGTH, 128))(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = UpSampling2D()(x)
x = Conv2D(64, kernel_size=(5,5), padding='same')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = UpSampling2D()(x)
generator_output = Conv2D(1, kernel_size=(5,5), padding='same', activation='tanh')(x)

generator = Model([noise, label], generator_output)
generator.compile(loss='binary_crossentropy', optimizer=adam)

# --- discriminator --- #
# validity: real vs. fake
# classification: assign category (class label), including extra category for fake data
# since there are two different discriminator outputs, we use a functional structure instead of using Sequential()
img = Input(shape=(PIX_LENGTH,PIX_LENGTH,1))
x   = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same', input_shape=(PIX_LENGTH,PIX_LENGTH,1))(img)
x   = LeakyReLU(0.2)(x)
x   = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same')(x)
x   = LeakyReLU(0.2)(x)
x   = Flatten()(x)
# branch into validity and classification 
validity = Dense(1, activation='sigmoid')(x)
class_label = Dense(NUM_CLASSES + 1, activation='softmax')(x)
# input is image, output is validity and class label
discriminator = Model(img, [validity, class_label])
discriminator.compile(loss=losses, optimizer=adam)

# freeze discriminator and train generator
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
    generated_imgs = generated_imgs.reshape(num_examples, PIX_LENGTH, PIX_LENGTH)

    plt.figure(figsize=(10,10))
    for i in range(num_examples):
        plt.subplot(10,10, i + 1)
        plt.imshow(generated_imgs[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.save_fig('images/acgan_generated_image_epoch_{0:03d}.png'.format(epoch))


# training
def train(num_epochs):
    for epoch in range(1, num_epochs + 1):
        discriminator_loss_epoch = 0.0
        generator_loss_epoch     = 0.0

        for batch in range(BATCH_COUNT):
            # --- train discriminator --- #
            indices = np.random.randint(0, TRAIN_SIZE, HALF_BATCH)
            images  = x_train[indices]
            
            noise = np.random.normal(0, 1, size=[HALF_BATCH, NOISE_DIM])
            sampled_labels = np.random.randint(0, 10, HALF_BATCH).reshape(-1, 1)
            generated_imgs = generator.predict([noise, sampled_labels]) 

            # validity: real vs. fake
            real_valid_y = 0.9 * np.ones((HALF_BATCH, 1))
            fake_valid_y =       np.zeros((HALF_BATCH, 1))
            # classification: correct class labels
            real_class_labels = y_train[indices] 
            fake_class_labels = NUM_CLASSES * np.ones(HALF_BATCH).reshape(-1, 1)
            # train
            discriminator_loss_real = discriminator.train_on_batch(images, [real_valid_y, real_class_labels])
            discriminator_loss_fake = discriminator.train_on_batch(generated_imgs, [fake_valid_y, fake_class_labels])
            discriminator_loss_batch = 0.5 * (np.array(discriminator_loss_real) + np.array(discriminator_loss_fake))
            discriminator_loss_epoch += np.mean(discriminator_loss_batch)

            # --- train generator --- #
            noise = np.random.normal(0, 1, size=[HALF_BATCH, NOISE_DIM])
            sampled_labels = np.random.randint(0, 10, HALF_BATCH).reshape(-1, 1)
            real_valid_y = np.ones((HALF_BATCH, 1))
            generator_loss_batch =  combined.train_on_batch([noise, sampled_labels], [real_valid_y, sampled_labels])
            generator_loss_epoch += np.mean(generator_loss_batch)
            
        print("epoch: {0}, discriminator loss: {1}, generator loss: {2}".format(epoch, discriminator_loss_epoch / BATCH_COUNT, generator_loss_epoch / BATCH_COUNT))
        if (epoch % 10 == 0):
            generator.save('models/dcgan_generator_{0:03d}.h5'.format(epoch))
            save_imgs(epoch)



if __name__ == "__main__":
    train(NUM_EPOCHS)

