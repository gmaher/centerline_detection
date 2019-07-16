import tensorflow as tf

class Conv2D
    def __init__(self,s, dims=[3,3], nfilters=32, strides=[1,1],
           init=1e-3, padding='SAME', activation=tf.identity, scope='conv2d', reuse=False):
        """
        args:
            x, (tf tensor), tensor with shape (batch,width,height,channels)
            dims, (list), size of convolution filters
            filters, (int), number of filters used
            strides, (list), number of steps convolutions slide
            std, (float/string), std of weight initialization, 'xavier' for xavier
                initialization
            padding, (string), 'SAME' or 'VALID' determines if input should be padded
                to keep output dimensions the same or not
            activation, (tf function), tensorflow activation function, e.g. tf.nn.relu
            scope, (string), scope under which to store variables
            reuse, (boolean), whether we want to reuse variables that have already
                been created (i.e. reuse an earilier layer)
        returns:
            a, (tf tensor), the output of the convolution layer, has size
                (batch, new_width , new_height , filters)
        """
        self.padding    = padding
        self.scope      = scope
        self.activation = activation

        with tf.variable_scope(scope,reuse=reuse):
            shape = dims +[s[3],nfilters]

            if init=='xavier':
                init = np.sqrt(1.0/(dims[0]*dims[1]*s[3]))

            self.W = tf.Variable(tf.random_normal(shape=shape,stddev=init),
                name='W')
            self.b = tf.Variable(tf.zeros([nfilters]), name='b')

    def __call__(self, x):
        with tf.variable_scope(self.scope, reuse=False):
            o = tf.nn.convolution(x, self.W, padding, strides=strides)

            o = o+self.b

            a = self.activation(o)

        return a

class FullyConnected
    def __init__(self,s,output_units=100,activation=tf.identity,std=1e-3,
                      scope='fc',reuse=False):
        """
        args:
            x, (tf tensor), tensor with shape (batch,width,height,channels)
            std, (float/string), std of weight initialization, 'xavier' for xavier
                initialization
            output_units,(int), number of output units for the layer
            activation, (tf function), tensorflow activation function, e.g. tf.nn.relu
            scope, (string), scope under which to store variables
            reuse, (boolean), whether we want to reuse variables that have already
                been created (i.e. reuse an earilier layer)
        returns:
            a, (tf tensor), the output of the fullyConnected layer, has size
                (batch, output_units)
        """
        self.scope      = scope
        self.activation = activation
        with tf.variable_scope(scope,reuse=reuse):
            shape = [s[1],output_units]

            if std=='xavier':
                std = np.sqrt(1.0/(shape[0]))

            self.W = tf.get_variable('W',shape=shape,initializer=tf.random_normal_initializer(0.0,std))
            self.b = tf.get_variable("b",shape=shape[1],initializer=tf.constant_initializer(0.0))

    def __call__(self,x):
        with tf.variable_scope(self.scope,reuse=False):
            h = tf.matmul(x,self.W)+self.b
            a = self.activation(h)
        return a
