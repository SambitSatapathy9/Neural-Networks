"""  
#  Model Evaluation and Selection
Quantifying a learning algorithm's performance and comparing different models are some of the common tasks when applying machine learning to real world applications. 

In this lab, we will- 
- split the datasets into training, cross-validation, and test sets
- evaluate regression and classification models
- add polynomial features to improve the performance of a linear regression model
- compare several neural network architectures
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

#For building and training neural networks
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError

#import custom functions
import utils

#Reduce display precision on numpy arrays
np.set_printoptions(precision=2)

import pandas as pd

## 2. Regression
#Given is the dataset below consisting of 50 examples of input feature `x` and its corresponding target `y`. 
#We will develop a regression problem from the given data
#Load the dataset from the text file
data = np.loadtxt("/home/sambit/ML_AI_Coursera/Coursera_02_Advanced_Learning/data/data_w3_ex1.csv", delimiter = ',')

#Split the inputs and outputs of the data
x = data[:,0]
y = data[:,1]

#Convert 1D arrays to 2D, required later
x = np.expand_dims(x, axis = 1)
y = np.expand_dims(y, axis = 1)

print(f"Shape of the inputs x is {x.shape}, \nShape of the targets y is {y.shape}")

#Visualise the dataset x vs y
plt.scatter(x,y, c = 'r', marker = 'x', linewidths=2)
plt.xlabel("x"); plt.ylabel("y");
plt.title("Inputs vs Targets")
"""
### 2.1 Split the dataset into training, cross-validation and test sets.
- **Training set** - used to train the model
- **Cross-validation set** - used to evaluate the model's performance, and hyperparameter tuning. It lets us decide which features we should use to get the desired model.
- **Test set** - used to test the unseen data. This should not be used to make decisions while you are still developing the models ***(for which we should use the cross-validation set)***


Scikit-learn provides the `train_test_split` function to split the data into the parts mentioned above. 
Here, the entire dataset is divided into 60% training, 20% cross-validation and 20% test sets.
"""
#Splitting the entire dataset as 60% training and rest 40% stored in the temporary varibles x_ and y_
x_train, x_, y_train, y_ = train_test_split(x,y,test_size = 0.40, random_state = 1)

#Splitting the 40% subset into two sets:  one half for cross-validation (20% cv) and other half for test set (20% test).
x_cv,x_test,y_cv,y_test = train_test_split(x_,y_,test_size = 0.50, random_state = 1)

#Delete temporary variables x_ and y_ 
del x_,y_

print(f"Shape of the training set input and target is {x_train.shape} and {y_train.shape}")
print(f"Shape of cross-validation set input and target is {x_cv.shape} and {y_cv.shape}")
print(f"Shape of the test set input and target is {x_test.shape} and {y_test.shape}")

#Visualise the train, cv and test data
X = [x_train,x_cv,x_test]
Y = [y_train,y_cv,y_test]
plt.scatter(x_train,y_train, marker = 'x', c='r', linewidths=2, label='Train')
plt.scatter(x_cv,y_cv, marker = 'o', c='b',label='CV')
plt.scatter(x_test,y_test, marker = '^', c='g',label='Test')
plt.legend()
plt.show()

