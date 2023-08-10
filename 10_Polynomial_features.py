"""
# Diagnosing Bias and Variance
The training and cross validation errors can tell you what to try next to improve your models. 
Specifically, it will show if you have a high bias (underfitting) or high variance (overfitting) problem.
- To fix a **high bias** problem, you can:

    - try adding polynomial features
    - try getting additional features
    - try decreasing the regularization parameter

- To fix a **high variance** problem, you can:

    - try increasing the regularization parameter
    - try smaller sets of features
    - get more training examples

# Establishing Baseline Level of Performance
Before you can diagnose a model for high bias or high variance, it is usually helpful to first have an idea of what
level of error you can reasonably get to.

We can use any of the following to set a baseline level of performance
- human level performance
- competing algorithm's performance
- guess based on experience

Real-world data can be very noisy and it's often infeasible to get to 0% error.

- **High bias problem**     - if $J_{train}>baseline$
- **High variance problem** - if $J_{cv} > J_{train}$
"""
#1. Imports and Lab Setup
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# for building linear regression models
import sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

# import lab utility functions in utils.py
import utils
"""
## 2. Fixing High Bias
### 2.1 Try adding polynomial features
Here we will use a synthetic dataset for a regression problem with one feature and one target. 
In addition, we will also define an arbitrary baseline performance and include it in the plot.
"""
#2. Prepare the dataset
def prepare_dataset(filename):
    data = np.loadtxt(filename, delimiter = ',')
    
    x = data[:,:-1]
    y = data[:,-1]
    
    # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
    x_train,x_,y_train,y_ = train_test_split(x,y, test_size = 0.40, random_state = 80)
    
    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_cv,x_test,y_cv,y_test = train_test_split(x_,y_,test_size = 0.50, random_state = 80)
    
    del x_, y_
    
    return x_train, y_train, x_cv, y_cv, x_test, y_test

x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset('data/c2w3_lab2_data1.csv')

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

#Preview the first 5 rows
print(f"first 5 rows of the training inputs (1 feature):\n {x_train[:5]}\n")


##Define the function for Polynomial Regression and store train and CV MSEs

def train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree = 10, baseline = None):
    
    #initialise the mses, models and scalars
    models     = []
    scalers    = []
    train_mses = []
    cv_mses    = []
    degrees = range(1, max_degree+1)
    
    #Loop over the model 10 times. Each time adding one more degree of polynomial higher than the previous for our case
    for degree in degrees:
        
        #Initialise the polynomial feature
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_poly = poly.fit_transform(x_train)
        
        #Scale the feature
        scaler =StandardScaler()
        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        scalers.append(scaler)
        
        #Build the model
        model.fit(X_train_poly_scaled, y_train)
        models.append(model)
        
        #Compute the training mses
        yhat_train = model.predict(X_train_poly_scaled)
        mse_train = mean_squared_error(yhat_train, y_train) / 2
        train_mses.append(mse_train)
        
        #Cross Validation Set
        X_cv_poly = poly.fit_transform(x_cv)
        X_cv_poly_scaled = scaler.transform(X_cv_poly)
        
        #COmpute cv mses
        yhat_cv = model.predict(X_cv_poly_scaled)
        mse_cv = mean_squared_error(yhat_cv, y_cv) / 2
        cv_mses.append(mse_cv)
        
    plt.plot(degrees, train_mses, c = 'r', marker = 'o', label = 'train mses', linewidth = 2)
    plt.plot(degrees, cv_mses, c='b', marker = 'o', label = 'cv mses', linewidth = 2)
    #Plot for baseline
    plt.plot(degrees, np.repeat(baseline,len(degrees)), linestyle = '--',label= 'baseline', linewidth = 2)
    plt.xlabel("Degrees"); plt.ylabel('MSEs');
    plt.title("Degree of Polynomial vs MSEs")
    plt.xticks(degrees)
    plt.legend()
    plt.show()   

#instantiate the regression model class
model = LinearRegression()

## Train and plot polynomial regression models. Bias is defined lower.
train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree= 10, baseline = 400)

"""
### Take note and understand what significance the baseline holds!
- As we can see, the more polynomial features we add, the better the model fits the data. In this example it even performed better than the baseline.
- Thus, we can say that the models with degree greater than 4 are **low-bias** as they perform close to or even better than the baseline performance. 

However, if the baseline is defined lower (e.g. you consulted an expert regarding the acceptable error), then the 
models are still considered high bias. You can then try other methods to improve this.
"""

train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree= 10, baseline = 250)

"""
### Try getting additional features
Another thing you can try is to acquire other features. Let's say that after you got the results above, you decided to launch another data collection 
campaign that captures another feature. Your dataset will now have 2 columns for the input features as shown below.
"""
x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset('data/c2w3_lab2_data2.csv')

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

# Preview the first 5 rows
print(f"first 5 rows of the training inputs (2 features):\n {x_train[:5]}\n")

model = LinearRegression()
train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree= 6, baseline = 250)
#In the above plot we can see that there is **low train error, high cv error - thus Overfitting**
