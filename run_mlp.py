from network import Network
from layers import Relu, Sigmoid, Softmax, Linear
from loss import CrossEntropyLoss, EuclideanLoss
from optimizer import SGDOptimizer
from optimizer import AdagradOptimizer
from solve_net import solve_net
from maps_db import MapsDB

import theano.tensor as T

sample_count = 4000
count_bins = 88 * 20
dataset = MapsDB('../db',
                 freq_count=sample_count,
                 count_bins=count_bins)
model = Network()
model.add(Linear('fc1', dataset.get_vec_input_width(), dataset.get_label_width(), 0.001))
# model.add(Sigmoid('relu1'))
# model.add(Linear('fc2', 128, dataset.get_label_width(), 0.001))
model.add(Sigmoid('relu2'))

loss = CrossEntropyLoss(name='xent')
# loss = EuclideanLoss(name='r2')

optim = SGDOptimizer(learning_rate=0.000003, weight_decay=0, momentum=0)
# optim = AdagradOptimizer(learning_rate=0.01, eps=1e-8)

input_placeholder = T.fmatrix('input')
label_placeholder = T.fmatrix('label')
model.compile(input_placeholder, label_placeholder, loss, optim)

solve_net(model, dataset,
          batch_size=10, max_epoch=10, disp_freq=8, test_freq=50)
