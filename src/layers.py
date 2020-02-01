from inits import *
import tensorflow as tf
from metrics import *
from crf import crf_layer
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dot(x, y, sparse=False):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
 
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Encoder(Layer):
    """Encoder layer."""
    def __init__(self,size1,size2,latent_factor_num,placeholders,act=tf.nn.relu, featureless=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.act = act
        self.featureless = featureless
        self.size1 = size1
        self.size2 = size2
        self.latent_factor_num = latent_factor_num
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weight1'] = glorot([size1[1]+size2[1],latent_factor_num])
            self.vars['weight2'] = glorot([size1[0]+size2[0],latent_factor_num])
    
            self.vars['weight3'] = glorot([latent_factor_num,latent_factor_num])
            self.vars['weight4'] = glorot([size1[0]+size2[0],latent_factor_num])
            

        if self.logging:
            self._log_vars()

    def _call(self,inputs):      
        
        adj = inputs[0]
        feature = inputs[1]
        
        #The first layer
        con_1 = dot(adj,feature)        
        T_1 = dot(con_1,self.vars['weight1'])
        hidden_1 = tf.add(T_1,self.vars['weight2'])
        
        #The second layer
        con_2 = dot(adj,hidden_1)
        T_2 = dot(con_2,self.vars['weight3'])
        hidden_2 = tf.add(T_2,self.vars['weight4'])
        
        #The CRF layer
        hidden_crf = hidden_2
        hidden_new = hidden_2
        for cv in range(0,1):  
           hidden_crf = crf_layer(hidden_crf,hidden_new)  
           hidden_new = hidden_crf
        return self.act(hidden_new)

class Decoder(Layer):
    """Decoder layer."""
    
    def __init__(self,size1,size2,latent_factor_num,act=tf.nn.relu,**kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.size1 = size1
        self.size2 = size2
        
        self.act = act
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weight5'] = glorot([latent_factor_num,latent_factor_num])
            self.vars['weight6'] = glorot([latent_factor_num,latent_factor_num])
    
    def _call(self,hidden):
        num_u = self.size1[0]
        U = hidden[0:num_u,:]
        V = hidden[num_u:,:]
        M = dot(dot(U,self.vars['weight5']),tf.transpose(self.vars['weight6']))
        M = dot(M,tf.transpose(V))
        M = tf.nn.sigmoid(M)
        M = tf.reshape(M,[-1,1])
     
        return M
    
    
    
    
    
    