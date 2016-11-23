import theano.tensor as T
import numpy as np
from utils import sharedX
import theano.tensor.nnet
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable

    def forward(self, inputs):
        pass

    def params(self):
        pass


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, inputs):
        # Your codes here
        return 0.5 * (T.abs_(inputs) + inputs)


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, inputs):
        # Your codes here
        return 1 / (1 + T.exp(-inputs))


class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)

    def forward(self, inputs):
        # Your codes here
        return theano.tensor.nnet.softmax(inputs)


class Linear(Layer):
    def __init__(self, name, inputs_dim, num_output, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.W = sharedX(np.random.randn(inputs_dim, num_output) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros((num_output)), name=name + '/b')

    def forward(self, inputs):
        # Your codes here
        batch_size = inputs.shape[0]
        n = T.prod(inputs.shape) / inputs.shape[0]
        inputs = T.reshape(inputs, [batch_size, n])
        return T.dot(inputs, self.W) + self.b

    def params(self):
        return [self.W, self.b]


class Convolution(Layer):
    def __init__(self, name, kernel_size, num_input, num_output, init_std):
        super(Convolution, self).__init__(name, trainable=True)
        W_shape = (num_output, num_input, kernel_size, kernel_size)
        self.W = sharedX(np.random.randn(*W_shape) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros((num_output)), name=name + '/b')

    def forward(self, inputs):
        # Your codes here
        result = conv2d(input=inputs, filters=self.W, border_mode='valid')
        s0, s2, s3 = result.shape[0], result.shape[2], result.shape[3]
        result += T.repeat(T.repeat(T.repeat(T.reshape(
            self.b, [1, self.b.shape[0], 1, 1]), s0, 0), s2, 2), s3, 3)
        return result

    def params(self):
        return [self.W, self.b]
        # return [self.W]


class Pooling(Layer):
    def __init__(self, name, kernel_size):
        super(Pooling, self).__init__(name)
        self.kernel_size = kernel_size

    def forward(self, inputs):
        # Your coders here
        return pool_2d(inputs,
                       ds=(self.kernel_size, self.kernel_size),
                       ignore_border=True,
                       mode='max')
