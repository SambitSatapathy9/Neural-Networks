"""
DECISION TREES, RANDOM FOREST, XGBOOST

## Heart Failure Prediction Dataset - Kaggle
### Context
- Cardiovascular disease (CVDs) is the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of five CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs.
- People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management.  
- This dataset contains 11 features that can be used to predict possible heart disease.
- Let's train a machine learning model to assist with diagnosing this disease.

#### Attribute Information
- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: output class [1: heart disease, 0: Normal]
"""
#1. IMPORT LIBRARIES    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, recall_score,confusion_matrix,precision_score
from sklearn.tree import DecisionTreeClassifier
RANDOM = 55 ## We will pass it to every sklearn call so we ensure reproducibility

#2. Load the dataset
df = pd.read_csv("heart.csv")
df.head()

"""
Data Engineering
#One-hot encoding to the categorical features
- Pandas has a built-in method to one-hot encode variables, it is the function `pd.get_dummies`. There are several arguments to this function, but here we will use only a few. They are:

    - data: DataFrame to be used
    - prefix: A list with prefixes, so we know which value we are dealing with
    - columns: the list of columns that will be one-hot encoded. 'prefix' and 'columns' must have the same length.
"""
# This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
cat_variables = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
df = pd.get_dummies(data=df, 
                    prefix=cat_variables,
                    columns=cat_variables )
df.head()
#The target variable is the heart disease
# Removing the target variable from the features
features = [x for x in df.columns if x not in 'HeartDisease']
len(features)

#Splitting the dataset
X_train,X_val,y_train,y_val = train_test_split(df[features],df.HeartDisease, train_size=0.8, random_state =RANDOM)
print(f"Train samples:      {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Target Proportion:  {sum(y_train)/len(y_train):.4f}")

"""
## 4. Building the Models  ---  Decision Trees  ---  Random Forest  ---  XGBoost
### 4.1 Decision Trees
There are several hyperparameters in the Decision Tree object from Scikit-learn.

The hyperparameters we will use and investigate here are:

 - min_samples_split: The minimum number of samples required to split an internal node. 
   - Choosing a higher min_samples_split can reduce the number of splits and may help to reduce overfitting.
 - max_depth: The maximum depth of the tree. 
   - Choosing a lower max_depth can reduce the number of splits and may help to reduce overfitting.
   
**(i) min_samples_split**
"""
from sklearn.tree import DecisionTreeClassifier

min_samples_split_list = np.array([2,10,30,50,100,200,300,700])
max_depth_list = np.array([1,2,3,4,8,16,32,None]) 
#None refers to there's no depth limit

accuracy_list_train = []
accuracy_list_val   = []

for min_samples_split in min_samples_split_list:
    model = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                  random_state = RANDOM)
    #Fit the model
    model.fit(X_train, y_train)
    
    #Predict the training model and the validation model
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    
    accuracy_train = accuracy_score(pred_train, y_train)
    accuracy_list_train.append(accuracy_train)
    
    accuracy_val   = accuracy_score(pred_val, y_val)
    accuracy_list_val.append(accuracy_val)
    
#Plot the accuracy scores vs min_samples_split
plt.xticks(ticks = range(len(min_samples_split_list)), 
           labels=min_samples_split_list)
plt.plot(accuracy_list_train, label='Train')
plt.plot(accuracy_list_val, label="Validation")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy score")
plt.title('Train x Validation metrics')
plt.legend()

#(ii) max_depth
accuracy_list_train = []
accuracy_list_val   = []

for max_depth in max_depth_list:
    model = DecisionTreeClassifier(max_depth = max_depth, random_state = RANDOM)
    
    #Fit the model
    model.fit(X_train, y_train)
    
    #Predict the train and validation model
    pred_train = model.predict(X_train)
    pred_val   = model.predict(X_val)
    
    #Accuracy scores
    accuracy_train = accuracy_score(pred_train, y_train)
    accuracy_list_train.append(accuracy_train)
    
    accuracy_val   = accuracy_score(pred_val, y_val)
    accuracy_list_val.append(accuracy_val)
    
#Plot the accuracy_score vs max_depth
plt.xticks(ticks = range(len(max_depth_list)), labels = max_depth_list)
plt.plot(accuracy_list_train, label='Train')
plt.plot(accuracy_list_val, label = 'Validation')
plt.legend()
plt.xlabel("max depth")
plt.ylabel("accuracy score")
plt.title("Train x Validation metrics")
plt.show()

"""
**From the above two graphs we can conclude that:**
- The accuracy score is max for validation set in min_samples_split when it is 50
- The accuracy score is max for validation set in max_depth when it is 4

- `max_depth = 4`
- `min_samples_split = 50`

### Build a Decision Tree Model with best metrics

"""
decision_tree_model = DecisionTreeClassifier(min_samples_split=50,
                                            max_depth = 4, 
                                            random_state=RANDOM)
#Fit the model
decision_tree_model.fit(X_train, y_train)

#Predict the train and validation sets
pred_train = decision_tree_model.predict(X_train)
pred_val   = decision_tree_model.predict(X_val)

acc_train = accuracy_score(pred_train, y_train)
acc_val   = accuracy_score(pred_val, y_val)

print(f"Accuracy score for Training set: {acc_train:.4f}")
print(f"Accuracy score for Validation set: {acc_val:.4f}")

#**No sign of overfitting even though the metrics are not that good**
"""
### 4.2 Random Forest
Now let's try the Random Forest algorithm also, using the Scikit-learn implementation. 
- All of the hyperparameters found in the decision tree model will also exist in this algorithm, since a random forest is an ensemble of many Decision Trees.
- One additional hyperparameter for Random Forest is called `n_estimators` which is the number of Decision Trees that make up the Random Forest. 
- if $n$ is the number of features, we will randomly select $\sqrt{n}$ of these features to train each individual tree. 
- Note that you can modify this by setting the `max_features` parameter.
You can also speed up your training jobs with another parameter, `n_jobs`. 
- Since the fitting of each tree is independent of each other, it is possible fit more than one tree in parallel. 
- So setting `n_jobs` higher will increase how many CPU cores it will use.
"""
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  ## If the number is an integer, then it is the actual quantity of samples,
                                             ## If it is a float, then it is the percentage of the dataset
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]

#**(i) min_samples_split**
accuracy_list_train = []
accuracy_list_val = []

for min_samples_split in min_samples_split_list:
    model = RandomForestClassifier(min_samples_split=min_samples_split,
                                  random_state=RANDOM)
    model.fit(X_train,y_train)
    
    pred_train = model.predict(X_train)
    pred_val   = model.predict(X_val)
    
    acc_train = accuracy_score(pred_train, y_train)
    accuracy_list_train.append(acc_train)
    
    acc_val   = accuracy_score(pred_val, y_val)
    accuracy_list_val.append(acc_val)
    
#Plot the accuracy score vs min_samples_split
plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list)), labels = min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])

"""
Notice that, even though the validation accuraty reaches is the same both at `min_samples_split = 2` and `min_samples_split = 10`, in the latter the difference in training and validation set reduces, showing less overfitting.
"""
#**(ii) max_depth**
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = RANDOM).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])


#**(iii)** n_estimators

accuracy_list_train = []
accuracy_list_val = []
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])








