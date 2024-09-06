#import libraries
import math 
import numpy as np 
from collections import Counter

#define euclidean distance
def euclidean_distance(x1, x2):
        return math.sqrt(np.sum((x1 - x2) ** 2))

#knn class
class KNN:
    
    #initialize k 
    def __init__(self,k=3):
        self.k = k 
    
    #fit the data
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    #compute distance between points
    def _predict(self,x):
        
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        knn_labels = [self.y_train[i] for i in k_idx]  
        
        # return the most common class label
        most_common = Counter(knn_labels).most_common(1)
        return most_common[0][0]

    #predict fn
    def predict(self,X):
        y_pred =  [self._predict(x)for x in X]
        return np.array(y_pred)
    
