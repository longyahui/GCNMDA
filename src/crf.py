
import tensorflow as tf
import numpy as np
from utils import preprocess_adj 
from utils import adj_to_bias


def dot(x, y, sparse=False):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def crf_layer(hidden,hidden_new):    
   
   alpha = 50  
   beta = 50
   
   bias_mat = adj_to_bias(sizes=[hidden.shape[0]], nhood=1)
   bias_mat = tf.cast(bias_mat, tf.float32)
   hidden_extend=hidden[np.newaxis]
   
   coefs = attention(hidden_extend,bias_mat,hidden.shape[1])
   coefs = coefs[0]
   
   hidden_neighbor = tf.multiply(dot(coefs,hidden_new),beta) 
   hidden_self = tf.multiply(hidden,alpha) 
   hidden_crf = tf.add(hidden_neighbor,hidden_self)
     
   unit_mat = tf.ones((hidden.shape[0],hidden.shape[1]),dtype=tf.float32)
   coff_sum = tf.multiply(dot(coefs,unit_mat),beta)
   const = tf.add(coff_sum,tf.multiply(unit_mat,alpha))    
    
   hidden_crf = tf.divide(hidden_crf,const)
   
   return hidden_crf

def attention(emb,bias_mat,out_sz=25):
    
    with tf.name_scope('my_attn'):
        
        seq_fts = tf.layers.conv1d(emb, out_sz, 1, use_bias=False)  
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) 
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) 
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)   
            
        return coefs



    
      
      

