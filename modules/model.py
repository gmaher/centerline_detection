import tensorflow as tf
import modules.layers as tf_util
import numpy as np

def get_model(model_type):
    if model_type == "I2INetRegression":
        return I2INetReg
    else:
        raise RuntimeError("Unrecognized model type")

class Model(object):
    def __init__(self,config):
        self.config   = config

        self.build_model()

        self.configure_trainer()

        self.finalize()

    def finalize(self):
        pass

    def train_step(self,xb,yb):
        self.global_step = self.global_step+1

        if np.sum(np.isnan(xb)) > 0: return
        if np.sum(np.isnan(yb)) > 0: return

        self.sess.run(self.train,{self.x:xb,self.y:yb})

    def save(self):
        model_dir  = self.config['MODEL_DIR']
        model_name = self.config['MODEL_NAME']
        self.saver.save(
            self.sess,model_dir+'/{}'.format(model_name))

    def load(self, model_path=None):
        if model_path == None:
            model_dir  = self.config['MODEL_DIR']
            model_name = self.config['MODEL_NAME']
            model_path = model_dir + '/' + model_name
        self.saver.restore(self.sess, model_path)

    def predict(self,xb):
        return self.sess.run(self.yclass,{self.x:xb})

    def calculate_loss(self,xb,yb):
        return self.sess.run(self.loss,{self.x:xb,self.y:yb})

    def build_model(self):
        raise RuntimeError("Abstract not implemented")

    def build_loss(self):
        if not 'LOSS' in self.config:
            self.loss = tf.reduce_mean(
                   tf.nn.sigmoid_cross_entropy_with_logits(logits=self.yhat,labels=self.y))

        elif self.config['LOSS'] == 'BALANCED':
            self.loss = tf_util.class_balanced_sigmoid_cross_entropy(logits=self.yhat,label=self.y)

        elif self.config['LOSS'] == 'MASKED':
            self.loss = tf_util.masked_loss_2D(self.yhat,self.y)

        elif self.config['LOSS'] == 'MASKED_3D':
            self.loss = tf_util.masked_loss_3D(self.yhat,self.y)

    def configure_trainer(self):
        LEARNING_RATE = self.config["LEARNING_RATE"]
        self.global_step = tf.Variable(0, trainable=False)
        boundaries = [10000, 20000, 25000]
        values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/100, LEARNING_RATE/1000]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.train = self.opt.minimize(self.loss)

class I2INetReg(Model):
    def build_model(self):
        CROP_DIMS   = self.config['CROP_DIMS']
        C           = self.config['NUM_CHANNELS']
        LEAK        = self.config['LEAK']
        NUM_FILTERS = self.config['NUM_FILTERS']
        LAMBDA      = self.config['L2_REG']
        INIT        = self.config['INIT']

        NUM_POINTS  = self.config['NUM_CONTOUR_POINTS']+2

        leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

        self.x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None,NUM_POINTS],dtype=tf.float32)

        self.yclass,self.yhat,_,_ = tf_util.I2INet(self.x,nfilters=NUM_FILTERS,
            activation=leaky_relu,init=INIT)


        o = leaky_relu(self.yhat)

        if "POOL" in self.config:
            pool = self.config['POOL']
            o = tf.nn.pool(o,window_shape=[pool, pool],pooling_type="MAX",padding="VALID",strides=[pool, pool])

        s = o.get_shape().as_list()

        o_vec = tf.reshape(o,shape=[-1,s[1]*s[2]*s[3]])

        for i in range(self.config['FC_LAYERS']-1):
            if "HIDDEN_SIZES" in self.config:
                h = self.config['HIDDEN_SIZES'][i]
            else:
                h = self.config['HIDDEN_SIZE']

            o_vec = tf_util.fullyConnected(o_vec, h,
                leaky_relu, std=INIT, scope='fc_'+str(i))

        self.yhat = tf_util.fullyConnected(o_vec, NUM_POINTS,
            tf.nn.sigmoid, std=INIT, scope='fc_final')

        self.build_loss()

        self.saver = tf.train.Saver()

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
        self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def predict(self,xb):
        return self.sess.run(self.yhat,{self.x:xb})

    def finalize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

class ResNetReg(Model):
    def build_model(self):
        CROP_DIMS   = self.config['CROP_DIMS']
        C           = self.config['NUM_CHANNELS']
        LEAK        = self.config['LEAK']
        LAMBDA      = self.config['L2_REG']
        INIT        = self.config['INIT']

        NLAYERS     = int(self.config['NLAYERS']/2)
        NFILTERS_SMALL = self.config['NFILTERS_SMALL']
        NFILTERS_LARGE = self.config['NFILTERS_LARGE']

        NUM_POINTS  = self.config['NUM_CONTOUR_POINTS']+2

        leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

        self.x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None,NUM_POINTS],dtype=tf.float32)

        self.yclass,self.yhat,_,_ = tf_util.ResNet(self.x,
            nlayers_before=NLAYERS, nlayers_after=NLAYERS,
            nfilters_small=NFILTERS_SMALL, nfilters_large=NFILTERS_LARGE,
            output_filters=NFILTERS_LARGE, activation=leaky_relu, init=INIT)


        o = leaky_relu(self.yhat)

        if "POOL" in self.config:
            pool = self.config['POOL']
            o = tf.nn.pool(o,window_shape=[pool, pool],pooling_type="MAX",padding="VALID",strides=[pool, pool])

        s = o.get_shape().as_list()

        o_vec = tf.reshape(o,shape=[-1,s[1]*s[2]*s[3]])

        for i in range(self.config['FC_LAYERS']-1):
            if "HIDDEN_SIZES" in self.config:
                h = self.config['HIDDEN_SIZES'][i]
            else:
                h = self.config['HIDDEN_SIZE']

            o_vec = tf_util.fullyConnected(o_vec, h,
                leaky_relu, std=INIT, scope='fc_'+str(i))

        self.yhat = tf_util.fullyConnected(o_vec, NUM_POINTS,
            tf.nn.sigmoid, std=INIT, scope='fc_final')

        self.build_loss()

        self.saver = tf.train.Saver()

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
        self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def predict(self,xb):
        return self.sess.run(self.yhat,{self.x:xb})

    def finalize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
