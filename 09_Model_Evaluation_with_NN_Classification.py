"""
CLASSIFICATION (Neural Networks)
Continuing our discussion from our previous lab, we will extend our idea to Classification (Neural Networks).
The same model selection process can also be used when choosing between different neural network architectures. 
In this section, we will create the models shown below and apply it to the same regression task above. 
"""
## 1. Imports and Lab setup
#For array computation and loading data
import numpy as np

#For visualising data
import matplotlib.pyplot as plt

#For building linear regression models and preparing data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

#1.1 Load the data set
#Load the dataset from a text file
data = np.loadtxt('data/data_w3_ex2.csv', delimiter = ',')
#Split the data into 
x_bc = data[:,:-1] # selects all rows (:) and all columns except the last one (:-1)
y_bc = data[:,-1]  # selects all rows (:) and only the last column (-1).

#Convert y into 2D , x is already 2D
y_bc = np.expand_dims(y_bc, axis = 1) #selects all rows (:) and only the last column (-1).

print(f"Shape of inputs x is: {x_bc.shape}")
print(f"Shape of target y is: {y_bc.shape}")

#1.2 Visualise the dataset
for i in range(len(y_bc)):
    if y_bc[i]==0:
        plt.scatter(x_bc[i,0], x_bc[i,1],marker = 'o', c='b')
    elif y_bc[i]==1:
        plt.scatter(x_bc[i,0], x_bc[i,1],marker = 'x', c='r', linewidths= 2)

#To have two separate labels (y=0 and y=1) without associating them with individual points, you can create empty scatter plots with the desired labels and marker styles       
plt.scatter([],[],marker = 'o', c='b',label = 'y=0')
plt.scatter([],[],marker = 'x', c='r',label = 'y=1', linewidth = 2)

plt.xlabel("x_1"); plt.ylabel("x_2");
plt.title("x1 vs x2")
plt.legend()
plt.show()

"""
##Prepare the data 
We will use the same training, cross validation and the test sets we generated in the previous lab.
- We know that neural networks can learn non-linear relationships so you can opt to skip adding polynomial features. 
- The default `degree` is set to `1` to indicate that it will just use `x_bc_train`, `x_bc_cv`, and `x_bc_test` as is (i.e. without any additional polynomial features).
"""
#1.3 Split and Prepare the dataset
from sklearn.model_selection import train_test_split
# We have x_bc = data[:,:-1] and y_bc = data[:,-1]

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_bc_train,x_,y_bc_train,y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state = 1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_bc_cv,x_bc_test, y_bc_cv, y_bc_test = train_test_split(x_,y_,test_size = 0.50, random_state=1)

# Delete temporary variables
del x_,y_

print(f"the shape of the training set (input) is: {x_bc_train.shape}")
print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_bc_test.shape}")
print(f"the shape of the test set (target) is: {y_bc_test.shape}")


#1.4 Scale the features
#Initialize the class
scaler_bc = StandardScaler()

X_bc_train_scaled = scaler.fit_transform(x_bc_train)
X_bc_cv_scaled = scaler.transform(x_bc_cv)
X_bc_test_scaled = scaler.transform(x_bc_test)

#1.5 Build the model
def build_models(model):
  tf.random.set_seed(20)
  model_1 = Sequential
  (
    [
    Dense(25, activation = 'relu'),
    Dense(15, activation = 'relu'),
    Dense(1,  activation = 'linear')
    ], name = 'model_1'
  )
   model_2 = Sequential([
        Dense(20, activation = 'relu'),
        Dense(12, activation = 'relu'),
        Dense(12, activation = 'relu'),
        Dense(12, activation = 'relu'),
        Dense(1, activation = 'linear'),
    ], name = 'model_2')
    
    model_3 = Sequential([
        Dense(32, activation = 'relu'),
        Dense(16, activation = 'relu'),
        Dense(8,  activation = 'relu'),
        Dense(4,  activation = 'relu'),
        Dense(12, activation = 'relu'),
        Dense(1,  activation = 'linear'),
        
    ], name = 'model_3')
    
    model_list = [model_1, model_2, model_3]
    
    return model_list

#1.6 Train the model
