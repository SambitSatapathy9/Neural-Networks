"""
POLYNOMIAL FEATURES
Continuing our discussion from our previous lab, we will extend our idea to polynomial features. 
From the graphs earlier, we may have noticed that the target `y` rises more sharply at smaller values of `x` compared to
higher ones. A straight line might not be the best choice because the target `y` seems to flatten out as `x` increases. 
Now that we have these values of the training and cross validation MSE from the linear model, we can try adding polynomial 
features to see if we can get a better performance. The code will mostly be the same but with a few extra preprocessing steps. 
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
## 3.1 Create the additional features
We will generate the additional features from the training set. 
- We can use the `PolynomialFeatures` class to demonstrate the same. It will create a new input feature which has 
squared calues of the input `x` (i.e. degree = 2)
"""

#Instantiate the class to make polynomial features
poly = PolynomialFeatures(degree = 2, include_bias = False)

#Compute the number of fetures and transform the training set
X_train_mapped = poly.fit_transform(x_train)

#Preview the first 5 elements of the new training set. Left column is 'x' and right column is 'x^2'
print("     x        x^2   ")
print(X_train_mapped[:5])

## Feature Scaling
scaler_poly = StandardScaler()

#Normalise the dataset
X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped) #It computes mean and standard deviation of the training set then transform it

#Print first 6 elements of the scaled training set
print(X_train_mapped_scaled[:6])

#Instantiate the class
model = LinearRegression()

#Fit the model
model.fit(X_train_mapped_scaled,y_train)

#Compute Prediction for training set
yhat_poly = model.predict(X_train_mapped_scaled)

#Compute MSE for training set
mse_poly_sklearn = mean_squared_error(yhat_poly,y_train)/2
print(f"Training MSE for Polynomial features: {mse_poly_sklearn}")

## 3.3 Cross-Validation Set
#Adding the polynomial feature to the CV set
X_cv_mapped = poly.transform(x_cv)
#Scale the CV set 
X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
#Compute CV MSE
yhat_cv_poly = model.predict(X_cv_mapped_scaled)

mse_cv_poly_sklearn = mean_squared_error(yhat_cv_poly,y_cv)/2
print(f"Cross Validation MSE for Polynomial features: {mse_cv_poly_sklearn}")

"""
- It can be noticed that the MSEs are significantly better for both the training and cross validation set when you added the 2nd order polynomial. 
- We may want to introduce more polynomial terms and see which one gives the best performance. As shown in class, We can have 10 different models.
- We can create a loop that contains all the steps in the previous code cells. Here is one implementation that adds polynomial features up to degree=10. 
  We'll plot it at the end to make it easier to compare the results for each model.
"""
# 4. Generalisation of Polynomial Features




