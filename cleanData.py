import pandas as pd
import numpy
import csv
import matplotlib.pyplot as plt
import pickle

"""
    Open the csv and extract the data. Pickle the dataframe into a file.
"""
def readData():
    with open("sales.csv", newline="") as csvfile:
        filereader = csv.reader(csvfile, delimiter=",")
        allData = list(filereader)
        columns = allData[0]
        data = allData[1:1001]
        df = pd.DataFrame(data, columns = columns)
        df.to_pickle("sales.pkl")



"""
    Extract the necessary information from pickled file
"""
def condenseData():
    df = pd.read_pickle("sales.pkl")
    condenseDf = df[["NA_Sales", "EU_Sales"]]
    condenseDf.apply(pd.to_numeric)
    condenseDf = condenseDf.astype(float)
    condenseDf.to_pickle("condenseSales.pkl")
    print(condenseDf)



"""
    View data points
"""
def plotData():
    df = pd.read_pickle("condenseSales.pkl")
    df.plot(x = "NA_Sales", y= "EU_Sales", kind = 'scatter')	
    plt.title("NA_Sales vs. EU_Sales")
    plt.show()

    

"""
    Complete processing of data
"""
def obtainData():
    readData()
    condenseData()
    plotData()


obtainData()