import numpy as np
import random
from sklearn import neighbors, datasets


class BaselineModel:
    '''
    Abstract class to show what functions
    each model we implement needs to support
    '''
    def __init__(self):
        '''
        Sets flags for the model to aid in debugging
        '''

    def fit(self, *args):
        '''
        Trains model parameters and saves them as attributes of this class.
        Variable numbers of parameters; depends on the class
        '''
        raise NotImplementedError


    def predict(self, *args):
        '''
        Uses trained model parameters to predict values for unseen data.
        Variable numbers of parameters; depends on the class.
        Raises ValueError if the model has not yet been trained.
        '''
        if not self.trained:
             raise ValueError("This model has not been trained yet")
        raise NotImplementedError


class simpleKNN(BaselineModel):

    def __init__(self, num_classes, weighting = "uniform"):

        super(simpleKNN, self).__init__()
        self.num_classes = num_classes
        self.weighting = weighting


    def fit(self, X_train, y_train):
        '''
        Fits a simple knn Model given training matrix X (num_samples * num_features) and
        class labels y (which is a value in range (0, num_classes-1) )
        '''

        clf = neighbors.KNeighborsClassifier(self.num_classes, self.weighting)
        clf.fit(X_train, y_train)
        
        self.clf = clf
        self.trained = True


    def predict(self, input):

        ''' Predicts class label for given batch of input ((n_query, n_features)
        for simple KNN model.

        Returns predicitions which is a value in range (0, num_classes-1)
        which is of size (n_samples, n_output) or n_samples ?
        '''

        predictions = self.clf.predict(input)
        return predictions
