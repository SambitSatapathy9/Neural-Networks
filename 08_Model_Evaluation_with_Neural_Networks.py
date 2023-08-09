"""
NEURAL NETWORKS
Continuing our discussion from our previous lab, we will extend our idea to Neural Networks.
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

data = np.loadtxt("/home/sambit/ML_AI_Coursera/Coursera_02_Advanced_Learning/data/data_w3_ex1.csv", delimiter = ',')

#Split the inputs and outputs of the data
x = data[:,0]
y = data[:,1]

#Convert 1D arrays to 2D, required later
x = np.expand_dims(x, axis = 1)
y = np.expand_dims(y, axis = 1)

"""
### 2.1 Split the dataset into training, cross-validation and test sets.

Scikit-learn provides the `train_test_split` function to split the data into the parts mentioned above. 
Here, the entire dataset is divided into 60% training, 20% cross-validation and 20% test sets.
"""
#Splitting the entire dataset as 60% training and rest 40% stored in the temporary varibles x_ and y_
x_train, x_, y_train, y_ = train_test_split(x,y,test_size = 0.40, random_state = 1)

#Splitting the 40% subset into two sets:  one half for cross-validation (20% cv) and other half for test set (20% test).
x_cv,x_test,y_cv,y_test = train_test_split(x_,y_,test_size = 0.50, random_state = 1)

#Delete temporary variables x_ and y_ 
del x_,y_

"""
##Prepare the data 
We will use the same training, cross validation and the test sets we generated in the previous lab.
- We know that neural networks can learn non-linear relationships so you can opt to skip adding polynomial features. 
- The default `degree` is set to `1` to indicate that it will just use `x_train`, `x_cv`, and `x_test` as is (i.e. without any additional polynomial features).
"""
## 1.1 Add Polynomial Features
degree = 1
poly = PolynomialFeatures()
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped    = poly.transform(x_cv)
X_test_mapped  = poly.transform(x_test)

## 1.2 Scale the features
#**Notice that we are using the mean and standard deviation computed from the training set by just using `transform()` in the cross validation and test sets instead of `fit_transform()`.**
scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled    = scaler.transform(X_cv_mapped)
X_test_mapped_scaled  = scaler.transform(X_test_mapped)

##1.3 Create and train the models
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









