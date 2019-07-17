import tensorflow as tf
import modules.layers_new as layers

class Backbone(object):
    def __init__(self, input_channels=3, output_size=256, scope='backbone'):
        self.input_channels = input_channels
        self.output_size    = output_size
        self.scope          = scope

        act = tf.contrib.keras.layers.LeakyReLU(0.2)

        with tf.variable_scope(scope):
            self.conv1 = layers.Conv2D(input_channels, nfilters=3*output_size,
                init='xavier', activation=act, scope='conv1')

            self.conv2 = layers.Conv2D(3*output_size, nfilters=2*output_size,
                init='xavier', activation=act, scope='conv2')

            self.conv3 = layers.Conv2D(2*output_size, nfilters=output_size,
                init='xavier', activation=act, scope='conv3')

    def __call__(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)

        return o3

class RPN(object):
    def __init__(self):
        pass
    def __call__(self, x):
        pass

class RCNN(object):
    def __init__(self):
        pass
    def __call__(self, x):
        pass

class MaskNN(object):
    def __init__(self):
        pass
    def __call__(self, x):
        pass

class Net(object):
    def __init__(self, input_dims, hidden_size):
        pass
