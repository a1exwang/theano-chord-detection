from network import Network
from layers import Relu, Sigmoid, Softmax, Linear
from loss import CrossEntropyLoss, EuclideanLoss
from optimizer import SGDOptimizer, AdagradOptimizer
from maps_db import MapsDB
from theano import tensor as T


def network_setup(model_file_path=None):
    freq_count = 4000
    count_bins = 88 * 20
    dataset = MapsDB('../db',
                     freq_count=freq_count,
                     count_bins=count_bins,
                     batch_size=128,
                     start_time=0.5,
                     duration=0.5)
    model = Network()
    model.add(Linear('fc1', dataset.get_vec_input_width(), 2048, 0.001))
    model.add(Sigmoid('sigmoid1'))
    model.add(Linear('fc2', 2048, dataset.get_label_width(), 0.001))
    model.add(Softmax('softmax2'))

    loss = CrossEntropyLoss(name='xent')
    # loss = EuclideanLoss(name='r2')

    optim = SGDOptimizer(learning_rate=0.00001, weight_decay=0.005, momentum=0.9)
    # optim = AdagradOptimizer(learning_rate=0.001, eps=1e-6)

    input_placeholder = T.fmatrix('input')
    label_placeholder = T.fmatrix('label')
    label_active_size_placeholder = T.ivector('label_active_size')

    if model_file_path:
        model.loads(model_file_path)
    else:
        dataset.load_cache()

    model.compile(input_placeholder, label_placeholder, label_active_size_placeholder, loss, optim)
    return model, dataset, freq_count, count_bins

