import keras
from keras import layers
from keras.utils import plot_model
import tensorflow as tf

print ("strider was here")

inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)


# define some models using the Functional Graph
model = keras.Model(inputs, outputs, name="toy_resnet")
#model.summary()

block1Model = keras.Model(inputs, block_1_output, name="toy_resnetb1")

block2Model = keras.Model(block_1_output, block_2_output, name="toy_resnetb2")

block3Model = keras.Model(block_2_output, block_3_output, name="toy_resnetb3")

block4Model = keras.Model(block_3_output, outputs, name="toy_resnetb3")

# Train
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# compile some of the Models
model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )


block1Model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

block2Model.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["acc"],
        )

block3Model.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["acc"],
        )

block4Model.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["acc"],
        )


# Call fit on some of the Models Note: no training data for some
model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)

# Cannot call fit on this Model ONLY because there is no Training Data that has
# the right output size,
#block1Model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)

# Predict
modelOutput = model.predict(x_test[:1])

output = block1Model(x_test[:1])
output = block2Model(output)
output = block3Model(output)
output = block4Model(output)

areEqual = tf.math.equal(modelOutput, output)
print (areEqual)
 

print ("strider was here: ENDEEeeeee")

#plot_model(block1Model, show_shapes=True, show_layer_names=True, to_file='model.png')


