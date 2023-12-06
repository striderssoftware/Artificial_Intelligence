import tensorflow as tf
import matplotlib.pyplot as plt

print("Strider was here:", tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_features = x_train.shape[1]
print (n_features)

model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),   # The input, in the case the image
      tf.keras.layers.Dense(128, activation='relu'),   # hidden
      tf.keras.layers.Dropout(.2),                     # hidden
      tf.keras.layers.Dense(10)                        # The output layer, the solution space
    ])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
                            loss=loss_fn,
                            metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)  #epochs=25

#  SAVING
#  Once one net/model has "learned" something that knowledge can be utilized by all net/models of a particular type
#  To Quote Neo,   "I know Kung Foo" 
#
# This will give you a .keras zip file that containes
# A Keras model consisting of multiple components:

# The architecture, or configuration, which specifies what layers the model contain, and how they're connected.
# A set of weights values (the "state of the model").
# An optimizer (defined by compiling the model).
# A set of losses and metrics (defined by compiling the model).
# The Keras API saves all of these pieces together in a unified format, marked by the .keras extension. This is a zip archive consisting of the following:

   # A JSON-based configuration file (config.json): Records of model, layer, and other trackables' configuration.
   # A H5-based state file, such as model.weights.h5 (for the whole model), with directory keys for layers and their weights.
   # A metadata file in JSON, storing things such as the current Keras version.
      
print ("saving")
model.save("stridersMod.keras")

reconstructed_model = tf.keras.models.load_model("stridersMod.keras")

probability_model = tf.keras.Sequential([
      model,
      tf.keras.layers.Softmax()
    ])

rec_probability_model = tf.keras.Sequential([
      reconstructed_model,
      tf.keras.layers.Softmax()
    ])

predResult = probability_model.predict(x_test)
rec_predResult = rec_probability_model.predict(x_test)

areEqual = tf.math.equal(predResult, rec_predResult)
print (areEqual)

print ("edndeeeeeeeee")
