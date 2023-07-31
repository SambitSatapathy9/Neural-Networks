#COFFEE ROASTING 
"""
In this lab, we will build a small neural network using Numpy. 
We will use the example of 'Coffee Roasting' which takes two input features temperature and duration to make the coffee. 
We will get output whether the coffee roast is good or bad. 
The neural network contains an input layer, a hidden layer and an output layer.
"""
#FULL CODE

#Import Packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Prerquesties to be declared before the program
# 1 Define the function to create the data
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
  
#Additional Code to represent colors
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

#2 Visualise the data
def plt_roast(X,Y):
    Y = Y.reshape(-1,)
    fig,ax = plt.subplots(1,1,)
    
    #Plotting the points for good and bad roast
    ax.scatter(X[Y==1,0],X[Y==1,1], marker = 'x', c = 'r', label = 'Good Roast',s=70)
    ax.scatter(X[Y==0,0],X[Y==0,1], marker = 'o', facecolors = 'none' , edgecolors = 'b',s = 70,linewidth = 1, label = "Bad Roast")
    
    #Setting the lines to discriminate good and bad roast
    ax.axhline(y = 12, c = 'orange', linewidth = 1.2)
    ax.axvline(x = 175,c='orange',linewidth = 1.2)
      #Setting the diagonal line
    tr = np.linspace(175,260,50)
    ax.plot(tr, (-3/85)*tr + 21, c = 'orange', linewidth =1.2)    
   
    #Setting the titles
    ax.set_ylabel("Duration (in minutes)")
    ax.set_xlabel("Temperature(in degree C)")
    ax.set_title("Coffee Roasting")
    ax.legend(loc='upper right')

    plt.show()
    
#1. DEFINE SIGMOID FUNCTION
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

#2. Load the dataset
X,Y = load_coffee_data() #Defined elsewhere
print(X.shape,Y.shape)

#2.1 Visualise the dataset for good and bad roast
plt_roast(X,Y) #Defined elsewhere

#2.2Normalise the data
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis = -1)
norm_l.adapt(X) #learns mean and variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

## 3 DEFINE THE FUNCTIONS

#3.0 Define the activation function
g = sigmoid 

#3.1 Define the dense function
def dense(a_in,W,b):
    units = W.shape[1] #Number of columns in the weight parameter defines number of neurons in the layer
    a_out = np.zeros(units)
    for j in range(units):
        # W = W[:,j] #Get weight of the jth neuron. It selects all rows and jth coloumn
        z = np.dot(a_in,W[:,j]) + b[j]  #z represents the neuron's pre-activation value.
        a_out[j] = g(z)  #The output activation a_out[j] is stored in the a_out array for the j-th neuron.
        
    return a_out
    
#3.2 Define the sequential function 
#The sequential function represents a simple feedforward neural network with two dense layers. 
def sequential(x, W1,b1,W2,b2):
    a1 = dense(x,W1,b1)
    a2 = dense(a1,W2,b2)
    return a2

#3.3 Define the predict function
def predict(X,W1,b1,W2,b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = sequential(X[i],W1,b1,W2,b2)
    return p

#4 Trained weights and biases samples

W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

#5. Test the routine on two examples
X_tst = np.array([
    [200,13.9], #Positive example
    [200,17]    #Negative example
])

X_tstn = norm_l(X_tst)
predictions = predict(X_tstn,W1_tmp,b1_tmp,W2_tmp,b2_tmp)

#We can write it in the following way in binary form
yhat = (predictions>=0.5).astype(int)
print(f"Decisions: \n{yhat}")

