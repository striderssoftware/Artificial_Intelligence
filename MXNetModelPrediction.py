#!pip install MXNet

import mxnet as mnet
from mxnet import nd
from mxnet.gluon import nn
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from IPython import display
import matplotlib.pyplot as plt
import time

print ("strider was here", mnet.__version__)


#  Get Training Data
mnist_train = datasets.FashionMNIST(train=True)

transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)])
mnist_train = mnist_train.transform_first(transformer)

batch_size = 256
train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer),
        batch_size=batch_size, num_workers=4)


#  Create a Model  # Add the layers explictly
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                        nn.MaxPool2D(pool_size=2, strides=2),
                        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                        nn.MaxPool2D(pool_size=2, strides=2),
                        nn.Flatten(),
                        nn.Dense(120, activation="relu"),
                        nn.Dense(84, activation="relu"),
                        nn.Dense(10))

net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# Check ouput with label
def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()


#  Train the Model
for epoch in range(2):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)
    # calculate validation accuracy
    for data, label in valid_data:
        valid_acc += acc(net(data), label)
    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data),
        valid_acc/len(valid_data), time.time()-tic))
    


#  Predict
mnist_valid = datasets.FashionMNIST(train=False)
X, y = mnist_valid[:10]
preds = []
for x in X:
    x = transformer(x).expand_dims(axis=0)
    pred = net(x).argmax(axis=1)
    preds.append(pred.astype('int32').asscalar())

    _, figs = plt.subplots(1, 10, figsize=(15, 15))
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    display.set_matplotlib_formats('svg')
    for f,x,yi,pyi in zip(figs, X, y, preds):
        f.imshow(x.reshape((28,28)).asnumpy())
        ax = f.axes
        ax.set_title(text_labels[yi]+'\n'+text_labels[pyi])
        ax.title.set_fontsize(14)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
                             

        
print ("all doneefe")

