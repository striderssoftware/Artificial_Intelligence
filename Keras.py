
import tensorflow as tf
import keras
from keras import layers


print ("strider was here")

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)


# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x))) 

model = keras.Sequential(
        [
            layers.Dense(2, activation="relu"),
            layers.Dense(3, activation="relu"),
            layers.Dense(4),
        ]
)

model.layers

model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

model.pop()
print(len(model.layers))  # 2

model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))

# Specify the input shape in advance
#https://www.tensorflow.org/guide/keras/sequential_model

print ("ENDeeeeeee")
