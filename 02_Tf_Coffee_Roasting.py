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
X_t  = np.tile(Xnorm, (1000,1))
"""

#TENSORFLOW MODEL 
#There are two layers with the sigmoid activations
tf.random.set_seed(1223) #applied to acheive consistent results
model = Sequential(
    [
    tf.keras.Input(shape=(2,)),
    Dense(3, activation = 'sigmoid', name = "Layer1"),
    Dense(1, activation = 'relu', name = 'Layer2' )
    ]
)

model.summary()
"""
- In the summary we see that there are 9 parameters in layer 1 and 4 in layer 2. 
- Let us see explicitely how this works
### Total No. of neurons in a layer = $ \sum (v_i.n_i) + \sum b_i$
- $v_i$ is the number of input features
- $n_i$ = number of units
- $b_i$ = biases

### $w_i$  = $\sum v_i.n_i$

#Total Number of parameters in layer 1
### L1_num_params = 2 * 3 + 3 
- Here 2 - $v_i$ ,  3 - $n_i$ ,  3 - $b_i$
- In layer 1 we have two input features (temp, duration), 3units(neurons) and 3 biases. w_i = v_i.n_i

### L2_num_params = 3 * 1 + 1 
- This line calculates the total number of parameters in the second layer of the model. 
- The second layer has 3 input features (coming from the 3 units of the previous layer) and 1 unit.
So, there are 3 * 1 = 3 weights (W2 parameters) and 1 bias (b2 parameter).
"""
L1_num_params = 2 * 3 + 3
L2_num_params = 3 * 1 + 1

w1, b1 = model.get_layer("Layer1").get_weights()
w2, b2 = model.get_layer("Layer2").get_weights()

print(f"w1: {w1}\nb1: {b1}\nShape of w1: {w1.shape} and b1: {b1.shape}")
print(f"w2: {w2}\nb2: {b2}\nShape of w2: {w2.shape} and b2: {b2.shape}")


