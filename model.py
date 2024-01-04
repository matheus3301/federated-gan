import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from keras.optimizers import Adam
from keras import initializers

initializer = initializers.RandomNormal(mean=0.0, stddev=0.02)
latent_dim = 100

generator = Sequential()
generator.add(Dense(256, input_dim=latent_dim, kernel_initializer=initializer))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(512, kernel_initializer=initializer))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(1024, kernel_initializer=initializer))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(784, activation='tanh', kernel_initializer=initializer))
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializer))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(512, kernel_initializer=initializer))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(256, kernel_initializer=initializer))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (
        x_train[idx * 6000 : (idx + 1) * 6000],
        y_train[idx * 6000 : (idx + 1) * 6000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )

