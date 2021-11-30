from utils import div_list
import tensorflow as tf
import numpy as np
from train import Training

if __name__ == "__main__":
  # Initial model
  gcn = Training()
  
  # Set random seed
  seed = 123
  np.random.seed(seed)
  tf.compat.v1.set_random_seed(seed)

  labels = np.loadtxt("data/adj.txt")  
  reorder = np.arange(labels.shape[0])
  np.random.shuffle(reorder)

  cv_num=5

  order = div_list(reorder.tolist(),cv_num)
  for i in range(cv_num):
      print("cross_validation:", '%01d' % (i))
      test_arr = order[i]
      arr = list(set(reorder).difference(set(test_arr)))
      np.random.shuffle(arr)
      train_arr = arr
      scores = gcn.train(train_arr, test_arr)
 
