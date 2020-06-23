import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class LinearRegressionModel:
    """
        Initialize the model with the necessary data to train with
    """
    def __init__(self, trainingX, trainingY, testingX, testingY):
        self.trainingX = trainingX
        self.trainingY = trainingY
        self.testingX = testingX
        self.testingY = testingY
        self.thetaZero = 0
        self.thetaOne = 0



    """ 
        Initialize the necessary parameters
    """
    def initParams(self):
        self.thetaZero = random.random()
        self.thetaOne = random.random()



    """
        Find the values of the derivatives for applying gradient descent
    """
    def findDerivative(self):
        # keeping track of the inner term for derivative accumulation
        sumThetaZero = 0
        sumThetaOne = 0

        m = len(self.trainingX)

        # go through all the examples
        for i in range(m):

            # get the example 
            x = self.trainingX[i]

            # get the real and predicted values
            realY = self.trainingY[i]
            predY = self.thetaZero + (self.thetaOne * x)

            # find the error
            diff = predY - realY

            #accumulate the error
            sumThetaZero = sumThetaZero + diff
            sumThetaOne = sumThetaOne + (diff * x)

        derivativeThetaZero = sumThetaZero/m
        derivativeThetaOne = sumThetaOne/m

        return (derivativeThetaZero, derivativeThetaOne)

    

    """
        Graph the line with values of theta
    """
    def graphLine(self):
        x = np.array(range(-1, 50))
        y = eval(str(self.thetaZero) + "+" + str(self.thetaOne) + "*x")
        plt.plot(x, y)
        plt.show()

        


    """
        Fit a linear regression model given the learning rate
    """
    def fit(self,learningRate):

        # set up parameters necessary for linear regression
        self.initParams()
        go = True

        # repeating until convergence
        while go == True:
            
            # find values of derivatives
            dThetaZero, dThetaOne = self.findDerivative()
            print(dThetaZero, dThetaOne)

            # keep track of the old values of theta
            oldThetaZero = self.thetaZero
            oldThetaOne = self.thetaOne

            # update the values of theta
            self.thetaZero = oldThetaZero - (learningRate * dThetaZero)
            self.thetaOne = oldThetaOne - (learningRate * dThetaOne)

            # see if continuation is needed
            if abs(self.thetaZero - oldThetaZero) < 0.00001 and abs(self.thetaOne - oldThetaOne) < 0.00001:
                go = False
            else:
                oldThetaZero = self.thetaZero
                oldThetaOne = self.thetaOne

        print("FINAL VALUES OF THETA: ", self.thetaZero, self.thetaOne)
        self.graphLine()
        
            



    