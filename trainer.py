import pickle
import pandas as pd
import numpy
import linearRegression
    
"""
    Split the data into training and testing sets.
"""
def splitData():
    df = pd.read_pickle("condenseSales.pkl")
    trainingX = df[:800]["NA_Sales"].to_list()
    trainingY = df[:800]["EU_Sales"].to_list()
    testingX = df[801:]["NA_Sales"].to_list()
    testingY = df[801:]["EU_Sales"].to_list()
    return (trainingX, trainingY, testingX, testingY)


if __name__ == "__main__":
    trainingX, trainingY, testingX, testingY = splitData()
    model = linearRegression.LinearRegressionModel(trainingX, trainingY, testingX, testingY)
    model.fit(0.05)