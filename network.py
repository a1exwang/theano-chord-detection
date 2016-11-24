import theano
import theano.tensor as T
from utils import LOG_INFO
from theano.misc.pkl_utils import dump, load
from nhot_accuracy import NHotAcc


class Network(object):
    def __init__(self):
        self.layer_list = []
        self.params = []
        self.num_layers = 0

    def add(self, layer):
        self.layer_list.append(layer)
        self.num_layers += 1
        if layer.trainable:
            self.params += layer.params()

    def dumps(self, file_path):
        with open(file_path, 'wb') as f:
            dump(self.params, f)

    def loads(self, file_path):
        with open(file_path, 'rb') as f:
            self.params = load(f)
        index = 0
        for layer in self.layer_list:
            if layer.trainable:
                for i in range(len(layer.params())):
                    layer.set_params(i, self.params[index])
                    index += 1

    def find_layer_by_name(self, name):
        for layer in self.layer_list:
            if layer.get_name() == name:
                return layer
        raise RuntimeError("Wrong layer name")

    def compile(self, input_placeholder, label_placeholder, label_active_size_placeholder, loss, optimizer):
        x = input_placeholder
        for k in range(self.num_layers):
            x = self.layer_list[k].forward(x)

        self.loss = loss.forward(x, label_placeholder)
        self.updates = optimizer.get_updates(self.loss, self.params)

        nhot_acc = NHotAcc()
        self.accuracy = nhot_acc(label_placeholder, x, label_active_size_placeholder)
        LOG_INFO('start compiling model...')
        self.train = theano.function(
            inputs=[input_placeholder, label_placeholder, label_active_size_placeholder],
            outputs=[self.loss, self.accuracy, x],
            updates=self.updates,
            allow_input_downcast=True)

        self.test = theano.function(
            inputs=[input_placeholder, label_placeholder, label_active_size_placeholder],
            outputs=[self.accuracy, self.loss],
            allow_input_downcast=True)

        self.predict = theano.function(
            inputs=[input_placeholder],
            outputs=[x],
            allow_input_downcast=True)

        LOG_INFO('model compilation done!')

