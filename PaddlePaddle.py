!pip install paddlepaddle

import paddle as pad
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

print ("woo woo")

train_dataset = pad.vision.datasets.MNIST(mode='train')

train_data0 = np.array(train_dataset[0][0])
train_label_0 = np.array(train_dataset[0][1])

# Display the first image of the first batch
import matplotlib.pyplot as plt
plt.figure("Image") # the window title of the image
plt.figure(figsize=(2,2))
plt.imshow(train_data0, cmap=plt.cm.binary)
plt.axis('on') # Set the axis to off
plt.title('image') # Image title
plt.show()

print("shape of the image data and corresponding data:", train_data0.shape)
print("shape of the image label and corresponding data:", train_label_0.shape, train_label_0)
print("\n Print the first image of the first batch, and the number of the matched label is {}".format(train_label_0))

print ("woo woo:END")

