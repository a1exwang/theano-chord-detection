import theano.tensor as T
import theano.tensor.nnet.nnet


class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, inputs, labels):
        eps = 1e-9
        # return T.sum(theano.tensor.nnet.nnet.categorical_crossentropy(inputs, labels))
        return -T.sum(labels * T.log(inputs + eps) + (1-labels) * T.log(1-inputs + eps))
        # return (inputs - labels).norm(2)


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, inputs, labels):
        return T.sum((inputs - labels) * (inputs - labels))
