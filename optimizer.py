import theano.tensor as T
import theano.compile.sharedvalue;
from utils import sharedX


class SGDOptimizer(object):
    def __init__(self, learning_rate, weight_decay=0.005, momentum=0.9):
        self.lr = learning_rate
        self.wd = weight_decay
        self.mm = momentum

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            d = sharedX(p.get_value() * 0.0)
            new_d = self.mm * d - self.lr * (g + self.wd * p)
            updates.append((d, new_d))
            updates.append((p, p + new_d))

        return updates


class AdagradOptimizer(object):
    def __init__(self, learning_rate, eps=1e-8):
        self.lr = learning_rate
        self.eps = eps

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, dx in zip(params, grads):
            cache = sharedX(0)
            delta = self.lr * dx / (T.sqrt(cache) + self.eps)
            updates.append((cache, dx.norm(2)))
            updates.append((p, p - delta))

        return updates
