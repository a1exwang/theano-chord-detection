from network import Network
from layers import Relu, Softmax, Linear, Convolution, Pooling
from loss import CrossEntropyLoss
from optimizer import SGDOptimizer
from optimizer import AdagradOptimizer
from solve_net import solve_net
from mnist import load_mnist_for_cnn

import theano.tensor as T
import theano

theano.config.floatX = 'float32'

train_data, test_data, train_label, test_label = load_mnist_for_cnn('data')
model = Network()
model.add(Convolution('conv1', 5, 1, 8, 0.01))   # output size: N x 4 x 24 x 24
model.add(Relu('relu1'))
model.add(Pooling('pool1', 2))                  # output size: N x 4 x 12 x 12
model.add(Convolution('conv2', 3, 8, 12, 0.01))   # output size: N x 8 x 10 x 10
model.add(Relu('relu2'))
model.add(Pooling('pool2', 2))                  # output size: N x 8 x 5 x 5
model.add(Linear('fc3', 12 * 5 * 5, 10, 0.01))          # input reshaped to N x 200 in Linear layer
model.add(Softmax('softmax'))

loss = CrossEntropyLoss(name='xent')

# optim = SGDOptimizer(learning_rate=0.0001, weight_decay=0.005, momentum=0.9)
optim = AdagradOptimizer(learning_rate=0.0002, eps=1e-5)

input_placeholder = T.ftensor4('input')
label_placeholder = T.fmatrix('label')
model.compile(input_placeholder, label_placeholder, loss, optim)

solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=32, max_epoch=100, disp_freq=1000, test_freq=10000)
