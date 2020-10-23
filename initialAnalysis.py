import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import mdp
from datetime import datetime, timedelta
from dateutil.parser import parse

# Define all coins
# coins = ["BCH", "BSV", "BTC", "BTG", "DASH", "DOGE", "ETC", "ETH", "LSK", "LTC", "NEO", "OMG", "QTUM", "XLM", "XMR",
# "XRP", "ZEC"]
coins = ["BCH"]


def readData():
    # Define the date on the initial file. Usually 9/23/2020
    startDate = parse('2020-9-23')

    # Define the date on the final file. Usually yesterday's date, which is the most recent complete file (i.e. where complete = not getting updated)
    endDate = datetime.today() - timedelta(days=1)
    # endDay = endDate.day
    # endMonth = endDate.month
    # endYear = endDate.year

    # Folder where all files are stored
    folder = "Data"

    # type of files to read
    fileExtension = ".csv"

    # Data types of columns to read from the folder
    dataTypes = ['f', 'f', 'f', 'f', 'f', 'f', 'U6', 'U36', 'i', 'U36']

    # Read and append all data file files
    empty = []
    DataDictionary = {}
    for coin in coins:
        date = startDate

        DataDictionary[coin] = np.asarray(empty)

        while(date <= endDate):
            day = date.day
            if(day < 10):
                day = '0' + str(day)
            else:
                day = str(day)

            month = date.month
            if(month < 10):
                month = '0' + str(month)
            else:
                month = str(month)

            year = str(date.year)

            pathname = folder + "/" + coin + "/" + coin + "_" + \
                day + "_" + month + "_" + year + fileExtension

            data = np.genfromtxt(pathname, dtype=dataTypes,
                                 delimiter=',', names=True, usecols=np.arange(0, 10))

            if(len(DataDictionary[coin]) == 0):
                DataDictionary[coin] = data
            else:
                DataDictionary[coin] = np.append(DataDictionary[coin], data)
            date = date + timedelta(days=1)

    return DataDictionary


def saveData(DataDictionary):


def main():
    DataDictionary = readData()
    print(DataDictionary)
    saveData(DataDictionary)


if __name__ == "__main__":
    main()
