import numpy as np
import tensorflow as tf
import tensorflow.keras.Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

#You can load the model from the MNIST website. The codes for loading the data is not shown here. 
#THe following code is to show the model with an input layer, 2 hidden layers with 25 and 15 neurons respectively and an output layer with 10 outputs, i.e., the digits 0 to 9.

model = Sequential([
    Dense(25, activation = 'relu'),
    Dense(15, activation = 'relu'), 
    Dense(10, activation = 'softmax')
])

#Compile the model
model.compile(loss = SparseCategoricalCrossentropy())

#Fit the model
model.fit(X,y, epochs = 100)

#Prediction
pred = model.predict(X)


#There is another preffered way to write the above model using logits

model_new = Sequential([
    Dense(25, activation = 'relu'),
    Dense(15, activation = 'relu'),
    Dense(10, activation = 'linear')
])

#Compile the model
model_new.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             optimizer = tf.keras.optimizers.Adam(0.001))

#Fit the model
model_new.fit(X,y, epochs = 10)
