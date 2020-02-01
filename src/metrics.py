import tensorflow as tf
import numpy as np


def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)  #负样本的标签值为0，正样本的标签值为1
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32) 
    error *= mask  #*代表点乘 计算误差时需要将负样本的累加
#     return tf.reduce_sum(error)
    return tf.sqrt(tf.reduce_mean(error))

def euclidean_loss(preds, labels):
    euclidean_loss = tf.sqrt(tf.reduce_sum(tf.square(preds-labels),0))
    return euclidean_loss

def dot(x, y, sparse=False):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def prediction(hidden):
    num_u = 1373
    U=hidden[0:num_u,:]
    V=hidden[num_u:,:]
    M1 = dot(U,tf.transpose(V))
        
    U1 = tf.norm(U,axis=1,keep_dims=True, name='normal') #by row
    V1 = tf.norm(V,axis=1,keep_dims=True, name='normal')
    F = dot(U1,tf.transpose(V1))
    Score = tf.divide(M1,F)   #对应
    Score = tf.nn.sigmoid(Score)  #by long   
    Score = tf.reshape(Score,[-1,1])
    return Score

def prediction_np(hidden):
    num_u = 1373
    U=hidden[0:num_u,:]
    V=hidden[num_u:,:]
    M1 = np.dot(U,np.transpose(V))
        
    U1 = np.linalg.norm(U,ord=2,axis=1,keepdims=True) #by row
    V1 = np.linalg.norm(V,ord=2,axis=1,keepdims=True)
    F = np.dot(U1,np.transpose(V1))
    Score = M1/F   #对应
    #Score = tf.nn.sigmoid(Score)  #by long   
    #Score = tf.reshape(Score,[-1,1])
    return Score

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
   
        
    
    
    