import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

print ("strider was here:START")

#  READ IN CSV VALUES

np.set_printoptions(precision=3, suppress=True)

#  Load data into a pandas DataFrame  NOTE: changed Age feature to AnyKey
abalone_train = pd.read_csv(
"https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
"Viscera weight", "Shell weight", "AnyKey"])  

abalone_train.head()

# pop (remove) the Predict variable from the features
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('AnyKey')

abalone_features.head()

abalone_features = np.array(abalone_features)

n_features = abalone_features.shape[1]

abalone_model = tf.keras.Sequential([
    layers.Input(shape=(n_features,))
    layers.Dense(256),
    layers.Dense(128),
    layers.Dense(1)
    ])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())


abalone_model.fit(abalone_features, abalone_labels, epochs=1)

#  NORMALIZE THE INPUTS

normalize = layers.Normalization()

normalize.adapt(abalone_features)

norm_abalone_model = tf.keras.Sequential([
      layers.Input(shape=(n_features,))
      normalize,
      layers.Dense(64),
      layers.Dense(1)
    ])

norm_abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                                                      optimizer = tf.keras.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)


#  MIXED DATA TYPES

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

# Perform a calculation using the input
result = 2*input + 1

# the result doesn't have a value
result

calc = tf.keras.Model(inputs=input, outputs=result)

print(calc(1).numpy())
print(calc(2).numpy())

inputs = {}

for name, column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32


    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}


x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

    
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

titanic_features_dict = {name: np.array(value)
                         for name, value in titanic_features.items()}


features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)

def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])


    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam())
    return model

titanic_model = titanic_model(titanic_preprocessing, inputs)
titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

titanic_model.save('test')
reloaded = tf.keras.models.load_model('test')

features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}

before = titanic_model(features_dict)
after = reloaded(features_dict)
assert (before-after)<1e-3
print(before)
print(after)


