import tensorflow as tf
import matplotlib.pyplot as plt

print("Strider was here:", tf.__version__)

# The TRAINING DATA - This defines the problem and the solution space
# From a business view -  An Entire New Industry of Acquisition,
# Labeling, and Structure of this training data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_features = x_train.shape[1]
print (n_features)

#  Building Mechanism NOT an Algorithm
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),   # The input, in the case the image
      tf.keras.layers.Dense(128, activation='relu'),   # hidden
      tf.keras.layers.Dropout(.2),                     # hidden
      tf.keras.layers.Dense(10)                        # The output layer, the solution space
    ])
# The hidden layers are the structure that ALLOWS the
# algorithm to work.  The algorithm is "written" into the weights
# using backpropagation and cost functions

#  Training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
                            loss=loss_fn,
                            metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)  #epochs=25

print ("saving")

model.save("stridersMod.keras")

reconstructed_model = tf.keras.models.load_model("stridersMod.keras")
reconstructed_model.fit(x_train, y_train, epochs=1)

print ("endeeeee")
