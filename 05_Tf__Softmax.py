"""
                                           Softmax Function Implementation
                                           
This file contains the implementation of the softmax function 
 i) without the logits,       usual convention
ii) with the logits,          preferred one 
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import sklearn
from sklearn.datasets import make_blobs 
#The make_blobs function is useful for creating simple datasets for testing and prototyping machine learning models without the need for real-world data.
"""
### sklearn.make_blobs 

- **sklearn.make_blobs** is a utility function used to generate synthetic datasets for clustering and classification tasks. The function creates random data points that are normally distributed around specified cluster centers with a defined standard deviation.

* The **make_blobs** function is useful for creating simple datasets for testing and prototyping machine learning models without the need for real-world data.

**Parameters:**

- n_samples: The total number of data points to generate.
- n_features: The number of features (or dimensions) for each data point.
- centers: The number of cluster centers to generate. It can be an integer or an array-like specifying the center coordinates of the clusters.
- cluster_std: The standard deviation of each cluster. Larger values spread the data points more widely around the cluster centers.
- random_state: Seed for the random number generator. Setting a value for random_state ensures reproducibility of the generated data.

**Returns:**

- X: An array of shape (n_samples, n_features) containing the generated data points (feature vectors).
- y: An array of shape (n_samples,) containing the labels corresponding to the cluster membership of each data point (only available if centers is specified).

"""
# Generate synthetic data with 4 clusters
centers = [[-5,2],[-2,-2],[1,2],[5,-2]] #Specify center for each cluster
std_dvn = [1,0.6,0.9,0.4] #Specify std deviation for each cluster
X_train, y_train = make_blobs(n_samples = 2000, n_features=2, centers = centers, cluster_std = std_dvn, random_state = 99)

#To specify std deviation for all cluster at once, use the following code
#X_train, y_train = make_blobs(n_samples = 2000, n_features=2, centers = centers, cluster_std = 1.0, random_state = 9)

# Plot the data points with different colors for each cluster
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
cluster_names = ['Bus','Car','Cycle','Bike']
#cmap - 'plasma', 'viridis', 'jet', 'coolwarm', 'cividis'

#To get the names of each class in the plot
for i, name in enumerate(cluster_names):
    plt.scatter([],[],c=f'C{i}', label = name)

#Display the legend
plt.legend()

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Synthetic Data with 4 Clusters")
plt.show()
"""
- **cmap** - 'plasma', 'viridis', 'jet', 'coolwarm', 'cividis'
- **c=f'C{i}'**: The c argument specifies the color of the data points. 
- **f'C{i}'** uses an f-string to dynamically set the color based on the index i. 
- The f'C{i}' notation creates a string where the color is set to 'C0' for i=0, 'C1' for i=1, and so on. 
   In matplotlib, 'C0', 'C1', 'C2', etc., represent distinct colors from the default color cycle.
"""





