from layers import *
from metrics import *
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.hid = None
        self.temp=None
        
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        # activations
        
        self.activations.append(self.inputs)
        #flag=1
        for layer in self.layers:
        #    if flag==1:
        #       hidden,self.temp = layer(self.activations[-1])
        #       self.activations.append(hidden)
        #       flag=flag+1
        #    else:
               hidden = layer(self.activations[-1])   
               self.activations.append(hidden)
        self.outputs=self.activations[-1]
        
        self.hid = self.activations[-2]
        
        # Store model variables for easy access
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        #print("variables:",variables)
        self.vars = {var.name: var for var in variables}
        
        
        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass
    
    def hidd(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GAutoencoder(Model,):
    def __init__(self, placeholders, size_u, size_v, latent_factor_num, **kwargs):
        super(GAutoencoder, self).__init__(**kwargs)
        
        self.adj = placeholders['adjacency_matrix']
        self.feature = placeholders['Feature_matrix']
        self.placeholders = placeholders
        self.size_u = size_u
        self.size_v = size_v
        self.latent_factor_num = latent_factor_num
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.inputs = [self.adj, self.feature]
        self.build()
    
    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers[1].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss += masked_accuracy(self.outputs, self.placeholders['labels'],self.placeholders['labels_mask'],self.placeholders['negative_mask'])
    def _accuracy(self):
#         self.accuracy = euclidean_loss(self.outputs, self.placeholders['labels'])
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],self.placeholders['labels_mask'],self.placeholders['negative_mask'])
    

    def _build(self):
        self.layers.append(Encoder(size1=self.size_u,
                                   size2=self.size_v,
                                   latent_factor_num=self.latent_factor_num,
                                   placeholders=self.placeholders
                                   ))

        self.layers.append(Decoder(size1=self.size_u,
                                   size2=self.size_v,
                                   latent_factor_num=self.latent_factor_num,
                                   ))
        
    def predict(self):
        return self.outputs