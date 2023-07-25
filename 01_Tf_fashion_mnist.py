#Manually divide the train and test dataset from a given dataset using numpy and scikit-learn
import numpy as np
from sklearn.model_selection import train_test_split

#Load the dataset
data = keras.datasets.fashion_mnist
(train_images,train_labels), (test_images, test_labels) = data.load_data()

#Combine the images and the labels into one array
X = np.concatenate((train_images, test_images), axis = 0)
y = np.concatenate((train_labels, test_labels), axis = 0)

#Shuffle the dataset randomly
indices = np.arange(len(X)) # It generates an array of integers starting from 0 to the number of samples in the dataset.
np.random.shuffle(indices)

X = X[indices] #This line reorders the dataset "X" using the shuffled indices.
y = y[indices] #This line reorders the labels "y" (target classes) based on the same shuffled indices.

#Define the split ratio (80% training, 20% test)
train_ratio = 0.8

#Calculate the number of samples for each set
num_train_samples = int(len(X)*train_ratio)

#Split the dataset into training and test sets
X_train, X_test = X[:num_train_samples], X[num_train_samples:]
y_train, y_test = y[:num_train_samples], y[num_train_samples:]

#NOW WE HAVE SUCCESFULLY SEPARATED THE TRAIN AND TEST DATA MANUALLY

"""X[:num_train_samples]: This is called slicing in Python. 
The colon : indicates that we want to extract a portion of the X array. 
X[:num_train_samples] means we are selecting the first num_train_samples elements of the X array. 
Since the array is shuffled, these will be the first num_train_samples randomly selected images, which will be used for training.

X[num_train_samples:]: This slicing expression selects elements from the X array starting from the num_train_samples 
index and going up to the end of the array. These will be the remaining images after the first num_train_samples that 
will be used for testing."""
