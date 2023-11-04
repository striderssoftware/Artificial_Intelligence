#!pip install paddlepaddle

import paddle

print ("strider was here")

# Subclassing to Create a Model
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.flatten = paddle.nn.Flatten()
        self.liner = paddle.nn.Linear(10,3)
        
    def forward(self, inputs):
        y = self.flatten(inputs)
        return y

model = Model

print (model.sublayers())

# Add a new layer
newLayer = paddle.nn.Dropout()
model.add_sublayer("woo",newLayer)

print (model.sublayers())


print ("ENDeeeee")
