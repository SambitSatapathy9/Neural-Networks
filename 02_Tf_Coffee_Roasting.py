import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 1. Coffee Roasting Dataset
def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
    X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))
    
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))

X,Y = load_coffee_data()
print(X.shape, Y.shape)
# X - Temperature(Celsius), Duration(minutes)

# 2. Normalize the data in-order to do back propagation quickly
""" 
    create a "Normalization Layer". Note, as applied here, this is not a layer in your model.
    - 'adapt' the data. This learns the mean and variance of the data set and saves the values internally.
    - normalize the data.  
    It is important to apply normalization to any future data that utilizes the learned model. 
"""

print(f"Temperature Max,Min pre normalisation: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration Max,Min pre normalisation: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_1 = tf.keras.layers.Normalization(axis = -1)
norm_1.adapt(X) #Learns mean, variance
Xnorm = norm_1(X)
print(f"Temperature Max,Min post normalisation: {np.max(Xnorm[:,0]):0.2f}, {np.min(Xnorm[:,0]):0.2f}")
print(f"Duration Max,Min post normalisation: {np.max(Xnorm[:,1]):0.2f}, {np.min(Xnorm[:,1]):0.2f}")

# Tile/Copy our data to increase the training set size and reduce the number of training epochs
Xt = np.tile(Xnorm, (1000,1))
Yt = np.tile(Y, (1000,1))
print(Xt.shape, Yt.shape)

""" 
#### Each row of the resulting array "Xt" is an exact copy of the original array "Xnorm", and we repeated the array 
thousand times along the rows, and one time along the column, in order to reduce the number of epochs(iterations), 
during training the dataset. 

Syntax: 
new_arr = np.tile(old_arr, (num_of_times_row_copied, num_of_times_column_copied))
X_t  = np.tile(Xnorm, (1000,1))***
"""


