import tensorflow as tf
import modules.layers_new as layers

class Backbone(object):
    def __init__(self, input_channels=3, output_size=64, scope='backbone'):
        self.input_channels = input_channels
        self.output_size    = output_size
        self.scope          = scope

        self.act = tf.contrib.keras.layers.LeakyReLU(0.2)

        with tf.variable_scope(scope):
            self.conv1 = layers.Conv2D(input_channels, nfilters=3*output_size,
                init='xavier', activation=self.act, scope='conv1')

            self.conv2 = layers.Conv2D(3*output_size, nfilters=2*output_size,
                init='xavier', activation=self.act, scope='conv2')

            self.conv3 = layers.Conv2D(2*output_size, nfilters=output_size,
                init='xavier', activation=self.act, scope='conv3')

    def __call__(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)

        return o3

class RPN(object):
    def __init__(self, backbone, anchors, backbone_channels=64, window_size=20, hidden_size=256, scope='rpn'):
        '''
        anchors - 4xk array
        '''
        self.backbone          = backbone
        self.backbone_channels = backbone_channels
        self.window_size       = window_size
        self.hidden_size       = hidden_size
        self.num_boxes         = anchors.shape[0]
        self.anchors           = anchors
        self.anchors_tensor    = tf.constant(anchors, dtype=tf.float32)
        self.scope             = scope

        self.act = tf.contrib.keras.layers.LeakyReLU(0.2)

        with tf.variable_scope(scope):
            self.conv = layers.Conv2D(backbone_channels, dims=[window_size, window_size],
                nfilters=hidden_size, activation=self.act, scope='window_net')

            self.object_conv = layers.Conv2D(hidden_size, dims=[1, 1],
                nfilters=self.num_boxes, activation=tf.nn.sigmoid, scope='object_conv')

            self.box_conv = layers.Conv2D(hidden_size, dims=[1, 1],
                nfilters=4*self.num_boxes, activation=tf.nn.sigmoid, scope='box_conv')

            #anchor tensor?

    def __call__(self, x):
        o1 = self.backbone(x)
        o2 = self.conv(o1)

        obj = self.object_conv(o2)
        box = self.box_conv(o2)

        s = box.get_shape().as_list()

        shape = tf.constant([s[0],s[1],s[2],self.num_boxes,4])

        box = tf.reshape(box, shape=shape)

        return obj, box

    def get_box_tensor(self, x):
        obj, box = self(x)
        a = self.anchors_tensor

        bx = box[:,:,:,:,0]*a[:,2]+a[:,0]
        by = box[:,:,:,:,1]*a[:,3]+a[:,1]

        bw = tf.exp(box[:,:,:,:,2])*a[:,2]
        bh = tf.exp(box[:,:,:,:,3])*a[:,3]

        out_boxes = tf.stack([bx,by,bw,bh], axis=4)
        return out_boxes

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
