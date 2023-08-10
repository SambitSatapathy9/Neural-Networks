##Prepare the dataset

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
