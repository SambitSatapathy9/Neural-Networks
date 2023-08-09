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
## 1. Initialise lists containing training MSEs, CV MSEs , models and scalars
train_mses = []
cv_mses = []
models = []
scalers = []

## 2. Loop over 10 times. Each adding one more degree of polynomial higher than the previous.
for degree in range (1,11):
    ## 2.1 TRAINING SET
    ### 2.1.1 Adding polynomial feature
    poly = PolynomialFeatures(degree, include_bias = False)
    X_train_mapped = poly.fit_transform(x_train)
     #print(X_train_mapped)
    
    ### 2.1.2 Scale the training set
    scaler = StandardScaler()
    X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
    scalers.append(scaler)
     #print(scalers)
    
    ### 2.1.3 Create and train the model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)
     #print(models)
        
    ### 2.1.4 Compute the training MSEs    
    yhat_train = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat_train) / 2
    train_mses.append(train_mse)
     #print(train_mses)
    
    ## 2.2 CROSS VALIDATION SET    
    ### 2.2.1 Adding polynomial features and scale the CV set 
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
    
    ### 2.2.3 Compute the Cross Validation MSE
    yhat_cv = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat_cv) / 2
    cv_mses.append(cv_mse)

    
## 3. Plot the results    
degrees = range(1,11)
plt.plot(degrees, train_mses, label = 'Train MSEs', marker = 'o',c='r', linewidth = 2)
plt.plot(degrees, cv_mses, label = 'CV MSEs', marker = 'o', c='b', linewidth = 2)
plt.legend()
plt.show()

### Choosing the best model
"""
- While selecting a model we want to choose the one that performs well both on the training set as well as the CV set. 
- It implies that the model is able to learn the patterns from your training set without overfitting. 
- We can observe in the plots the sudden decrease in the cv error from the models with degree=1 to degree=2, followed by a relatively flat line thorugh degree=5.
- However, the cv error is getting worse when the degree increases further from degree=6 ,i.e. as we add more polynomial features.
- Given these, we can decide to use the model with the lowest `cv_mse` as the one best suited for our application
"""
## Get the model with the lowest CV MSE (add 1 because list indices start at 0)
# This also corresponds to the degree of the polynomial added

degree = np.argmin(cv_mses) + 1
print(f"Lowest cross validation error is found in the model with degree = {degree}")

#TEST SET

## Add Polynomial features
poly = PolynomialFeatures(degree, include_bias  = False)
X_test_mapped = poly.fit_transform(x_test)

## Scale the feature
X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

## Compute the MSE
yhat = models[degree-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(yhat,y_test) / 2

print(f"Training MSEs: {train_mses[degree-1]}")
print(f"Cross-Validation MSEs: {cv_mses[degree-1]}")
print(f"Test MSEs: {test_mse}")




