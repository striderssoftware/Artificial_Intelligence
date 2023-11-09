import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

print ("strider was here")

# Create a Graph of layers
encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

#  Create a (encoder) model using some of the layers
encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

#  Create a (decoder) model using some of the layers
decoder = keras.Model(encoder_output, decoder_output, name="decoder")
decoder.summary()

#  Create a (autoencoder) model using some of the layers
autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()

print ("ENDEEEeee")

