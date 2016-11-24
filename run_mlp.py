from network import Network
from layers import Relu, Sigmoid, Softmax, Linear
from loss import CrossEntropyLoss, EuclideanLoss
from optimizer import SGDOptimizer, AdagradOptimizer
from solve_net import solve_net
from maps_db import MapsDB
from play import play
import signal
import sys

import theano.tensor as T

# Check parameters
if len(sys.argv) >= 2:
    if sys.argv[1] == 'train':
        train_or_play = True
        model_file_path = None
        play_file_name = None
    else:
        train_or_play = False
        model_file_path = sys.argv[2]
        play_file_name = sys.argv[3]
else:
    raise RuntimeError("Wrong argument")


# Initialize signal handler to save model at SIGINT
def signal_handler(sig, frame):
    print('Save model? [y/n]')
    save_or_not = raw_input()
    if save_or_not == 'y':
        model.dumps('model.bin')
    print('Exiting...')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

freq_count = 4000
count_bins = 88 * 20
start_time = 0.5
duration = 0.5
dataset = MapsDB('../db',
                 freq_count=freq_count,
                 count_bins=count_bins,
                 batch_size=20,
                 start_time=0.5,
                 duration=0.5)
model = Network()
model.add(Linear('fc1', dataset.get_vec_input_width(), 1024, 0.001))
model.add(Sigmoid('sigmoid1'))
model.add(Linear('fc2', 1024, dataset.get_label_width(), 0.001))
model.add(Softmax('softmax2'))

loss = CrossEntropyLoss(name='xent')
# loss = EuclideanLoss(name='r2')

optim = SGDOptimizer(learning_rate=0.00005, weight_decay=0.001, momentum=0.9)
# optim = AdagradOptimizer(learning_rate=0.01, eps=1e-8)

input_placeholder = T.fmatrix('input')
label_placeholder = T.fmatrix('label')
label_active_size_placeholder = T.ivector('label_active_size')
model.compile(input_placeholder, label_placeholder, label_active_size_placeholder, loss, optim)

if train_or_play:
    dataset.load_cache()

    solve_net(model, dataset,
              max_epoch=300, disp_freq=100, test_freq=1000)

    print('Save model? [y/n]')
    yes_or_no = raw_input()
    if yes_or_no == 'y':
        model.dumps('model.bin')
else:
    model.loads(model_file_path)
    dataset.load_cache()

    result = play(model, play_file_name, freq_count=freq_count, count_bins=count_bins, duration=duration)
    print(result)

