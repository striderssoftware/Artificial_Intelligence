#!pip install MXNet

import mxnet as mnet
from mxnet import nd
from mxnet.gluon import nn

print ("strider was here", mnet.__version__)

#  Create a layer
layer = nn.Dense(2)
layer.initialize()
x = nd.random.uniform(-1,1,(3,4))
layer(x)
layer.weight.data()


# Create a Sequential model
net = nn.Sequential()

#  Add the layers explictly 
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Dense(120, activation="relu"),
                nn.Dense(84, activation="relu"),
                nn.Dense(10))

net.initialize()
# Input shape is (batch_size, color_channels, height, width)
x = nd.random.uniform(shape=(4,1,28,28))
y = net(x)
y.shape

print (net[0].weight.data().shape, net[6].bias.data().shape)

print ("all doneeeeeeee")


