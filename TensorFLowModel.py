
import tensorflow as tf
import matplotlib.pyplot as plt

print("Strider was here:", tf.__version__)

# An Entire New Industry of Acquisition, Labeling, and Structure of training data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Building Mechanism NOT an Algorithm
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),   # The input, in the case the image
      tf.keras.layers.Dense(128, activation='relu'),   # hidden
      tf.keras.layers.Dropout(.2),                     # hidden
      tf.keras.layers.Dense(10)                        # The output layer, the solution space
    ])
# The hidden layers are the structure that ALLOWS the
# work to be done.  The work is done with Backpropagation, and
# Weights while training.

#  Training
predictions = model(x_train[2:3]).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
                            loss=loss_fn,
                            metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)  # Raise this

# Evaluation Metrics
evalResults = model.evaluate(x_test,y_test)

# Wrap the Model in a probability model for "results"
probability_model = tf.keras.Sequential([
      model,
      tf.keras.layers.Softmax()
    ])

# Query the Model
probResult = probability_model(x_test, training=False)   # Probability of Class Membership
predResult = probability_model.predict(x_test)           # Model Confidence of Class Membership

print ('\n\n')

# Map results into solution space
class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
for i, logits in enumerate(predResult):  # probResults, predResults
    class_idx = tf.math.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]

    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
    # output image
    if i > 10:
        break
    plt.figure()
    plt.imshow(x_test[i])
    plt.grid(False)
    plt.show()

    print ("all doneeeeeee")

