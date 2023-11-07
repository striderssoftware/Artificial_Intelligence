#!pip install MXNet

import mxnet
import matplotlib.pyplot as plt
from mxnet.gluon.data.vision import datasets

print ("strider was here", mxnet.__version__)

#  Get Data
mnist_train = datasets.FashionMNIST(train=True)

text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

X, y = mnist_train[0:10]
# plot images
_, figs = plt.subplots(1, X.shape[0], figsize=(15, 15))

for f,x,yi in zip(figs, X,y):
    # 3D->2D by removing the last channel dim
    f.imshow(x.reshape((28,28)).asnumpy())
    ax = f.axes
    ax.set_title(text_labels[int(yi)])
    ax.title.set_fontsize(14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

print ("all doneeee")

