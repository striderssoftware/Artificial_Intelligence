#!pip install keras

import keras
from keras import layers
from keras.utils import plot_model

print ("strider was here")

num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

# Define 3 input layers or tensors
title_input_tensor = keras.Input(shape=(None,), name="title")  # Variable-length sequence of ints
body_input_tensor = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input_tensor = keras.Input(shape=(num_tags,), name="tags")  # Binary vectors of size `num_tags`

# Add layers to embed 2 of the tensors into vectors
title_features = layers.Embedding(num_words, 64)(title_input_tensor)
body_features = layers.Embedding(num_words, 64)(body_input_tensor)

# Add layers to reduce tensors into vectors
title_features = layers.LSTM(128)(title_features)
body_features = layers.LSTM(32)(body_features)

# Add a merge layer to concatenate the 3 layers, Note: this brings in the tag input tensor
x = layers.concatenate([title_features, body_features, tags_input_tensor])

# Add the 2 output or solution spaces
priority_pred = layers.Dense(1, name="priority")(x)
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input_tensor, body_input_tensor, tags_input_tensor],
    outputs=[priority_pred, department_pred],
)

print ("strider was here: ENDEEeeeee")

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
