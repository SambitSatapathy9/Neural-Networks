"""
13 - CNN CIFAR 10 Dataset  (17/09/2023)
# CIFAR-10 Dataset
The CIFAR-10 dataset consists of **60000 32x32 colour images** in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. 
The test batch contains exactly 1000 randomly-selected images from each class. 
The training batches contain the remaining images in random order, but some training batches may contain more images 
from one class than another. Between them, the training batches contain exactly 5000 images from each class. 
"""
#1 Load the required dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam,  SGD
from tensorflow.keras.metrics import Accuracy, SparseCategoricalCrossentropy, F1Score, Precision, Recall
from tensorflow.keras.models import Sequential
import sklearn
from sklearn.metrics import classification_report

#2. Load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape,"\n",X_test.shape)
y_train
#This is a 2d array, convert into 1d array
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

#3 Data Preprocessing
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def plot_sample(X,y,index):
#     fig,ax = plt.subplots(figsize=(10,2))
    plt.figure(figsize=(10,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    
# plot_sample(X_train, y_train, 3)
plot_sample(X_train, y_train, 53)

#Normalise the dataset
X_train = X_train/255
X_test  = X_test/255

#4. Define the model with simple ANN (No CNN)
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
#input_shape(dim, v_dim, color channel)) color channel = 1 -> Grayscale, color = 3 -> RGB
model.add(Dense(3000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5)

model.evaluate(X_test,y_test)

y_pred = model.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print(f"Classification Report:\n{classification_report(y_test, y_pred_classes)}")

### The above accuracy score is mere 50%. Now, let us introduce CNN into the picture
cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',
       input_shape = (32,32,3)))
cnn.add(MaxPooling2D(2,2))
cnn.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
       input_shape = (32,32,3)))
cnn.add(MaxPooling2D(2,2))

cnn.add(Flatten(input_shape=(32,32,3)))
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(10, activation='softmax'))
cnn.summary()

cnn.compile(loss='sparse_categorical_crossentropy',
           optimizer='adam',
           metrics = ['accuracy'])
cnn.fit(X_train,y_train,epochs=5)

y_test = y_test.reshape(-1,)
cnn.evaluate(X_test,y_test)
plot_sample(X_test,y_test,15)

y_pred = cnn.predict(X_test)
y_classes = [np.argmax(element) for element in y_pred]

y_classes[4:15]
y_test[4:15]

print(classification_report(y_test,y_classes))

#The accuracy is now 70% which is quite good as the epochs are just 5. The accuracy increased significantly by using CNN
 
