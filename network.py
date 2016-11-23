import theano
import theano.tensor as T
from utils import LOG_INFO
import pickle


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
        h = {}
        for layer in self.layer_list:
            h[layer.get_name()] = layer.params()
        bin_data = pickle.dumps(h)
        with open(file_path, 'w') as f:
            f.write(bin_data)

    def loads(self, file_path):
        with open(file_path, 'r') as f:
            h = pickle.loads(f.read())
            for name in h:
                self.find_layer_by_name(name).set_params(h[name])

    def find_layer_by_name(self, name):
        for layer in self.layer_list:
            if layer.get_name() == name:
                return layer
        raise RuntimeError("Wrong layer name")

    def compile(self, input_placeholder, label_placeholder, loss, optimizer):
        x = input_placeholder
        for k in range(self.num_layers):
            x = self.layer_list[k].forward(x)

        self.loss = loss.forward(x, label_placeholder)
        self.updates = optimizer.get_updates(self.loss, self.params)
        self.accuracy = T.mean(T.eq(T.argmax(x, axis=-1),
                               T.argmax(label_placeholder, axis=-1)))
        self.predict_val = T.argmax(x, axis=-1)
        self.true_val = T.argmax(label_placeholder, axis=-1)
        LOG_INFO('start compiling model...')
        self.train = theano.function(
            inputs=[input_placeholder, label_placeholder],
            outputs=[self.loss, self.accuracy, self.predict_val, self.true_val, x],
            updates=self.updates,
            allow_input_downcast=True)

        self.test = theano.function(
            inputs=[input_placeholder, label_placeholder],
            outputs=[self.accuracy, self.loss],
            allow_input_downcast=True)

        self.predict = theano.function(
            inputs=[input_placeholder],
            outputs=[x],
            allow_input_downcast=True)

        LOG_INFO('model compilation done!')

