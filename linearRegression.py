import pickle
import pandas as pd
import numpy


class LinearRegressionModel:
    """
        Initialize the model with the necessary data to train with
    """
    def __init__(self, trainingX, trainingY, testingX, testingY):
        self.trainingX = trainingX
        self.trainingY = trainingY
        self.testingX = testingX
        self.testingY = testingY

        

    """
        Fit a linear regression model given the learning rate
    """
    def fit(self,learningRate):
        pass

    