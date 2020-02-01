import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import matplotlib as plt
from pylab import *
import random
from inits import *
import pandas as pd
from sklearn.preprocessing import normalize

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(train_arr, test_arr):
    """Load data."""   
    labels = np.loadtxt("../data/adj.txt")  
    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(1373,173)).toarray()
    logits_test = logits_test.reshape([-1,1])  

    logits_train = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(1373,173)).toarray()
    logits_train = logits_train.reshape([-1,1])
      
    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])

    M = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(1373,173)).toarray()      
    
    DS=np.loadtxt("../data/drugsimilarity.txt")  
    MS=np.loadtxt("../data/microbesimilarity.txt")   
    adj = np.vstack((np.hstack((DS,M)),np.hstack((M.transpose(),MS))))   
                
    F1 = np.loadtxt("../data/drugfeatures.txt") 
    F2 = np.loadtxt("../data/microbefeatures.txt") 
    features = np.vstack((np.hstack((F1,np.zeros(shape=(1373,173),dtype=int))), np.hstack((np.zeros(shape=(173,1373),dtype=int), F2))))
    features = normalize_features(features)
    size_u = F1.shape
    size_v = F2.shape         
    
    adj = preprocess_adj(adj)  
    
    return adj, features, size_u, size_v, logits_train,  logits_test, train_mask, test_mask, labels

def generate_mask(labels,N):   
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(1373,173)).toarray()
    mask = np.zeros(A.shape)
    label_neg=np.zeros((5*N,2))  
    while(num<5*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            label_neg[num,0]=a
            label_neg[num,1]=b
            num += 1
    mask = np.reshape(mask,[-1,1])  
    #return mask
    return mask,label_neg

def test_negative_sample(labels,N,negative_mask):  
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(1373,173)).toarray()  
    mask = np.zeros(A.shape)
    test_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if A[a,b] != 1 and mask[a,b] != 1 and negative_mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    return test_neg

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    
    return features

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized


def construct_feed_dict(adj, features, labels, labels_mask, negative_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['adjacency_matrix']: adj})
    feed_dict.update({placeholders['Feature_matrix']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['negative_mask']: negative_mask})
    return feed_dict

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def adj_to_bias(sizes=[1546], nhood=1): 
      
    labels = np.loadtxt("../data/adj.txt")
    reorder = np.arange(labels.shape[0])
    train_arr=reorder.tolist()
    M = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(1373,173)).toarray()       
    adj = np.vstack((np.hstack((np.zeros(shape=(1373,1373),dtype=int),M)),np.hstack((M.transpose(),np.zeros(shape=(173,173),dtype=int)))))
    adj=adj+np.eye(adj.shape[0])
    adj=np.reshape(adj,(1,adj.shape[0],adj.shape[1]))
    nb_graphs = adj.shape[0] 
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)  
    #return adj

