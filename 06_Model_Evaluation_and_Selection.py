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

