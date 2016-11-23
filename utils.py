import theano
import numpy as np
from datetime import datetime


def sharedX(X, name=None):
    return theano.shared(
        np.asarray(X, dtype=theano.config.floatX),
        name=name,
        borrow=True)


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print display_now + ' ' + msg
