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
