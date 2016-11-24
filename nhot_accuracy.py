import theano
import theano.tensor as T
import numpy as np


class NHotAcc(theano.Op):
    # Properties attribute
    __props__ = ()

    itypes = [T.fmatrix, T.fmatrix, T.ivector]
    otypes = [T.dscalar]

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        x = np.argsort(inputs_storage[0])
        y = np.argsort(inputs_storage[1])
        counts = inputs_storage[2]
        z = output_storage[0]
        acc = 0
        for index, n in enumerate(counts):
            xx = x[index, -n:]
            diff = set(list(xx)) - set(list(y[index, -n:]))
            correct_count = len(xx) - len(diff)
            acc += float(correct_count) / len(xx)
        z[0] = acc / len(counts)
