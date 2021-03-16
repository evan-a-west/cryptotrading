import pandas as pd
import numpy as np
# from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
# import mdp
from datetime import datetime, timedelta
from dateutil.parser import parse
# from fbprophet import Prophet
from os import path
from scipy import signal
import time


def readData(coins) {
    # Define the date on the initial file. Usually 9/23/2020
    startDate = parse('2020-9-23')

    # Define the date on the final file. Usually yesterday's date, which is the most recent complete file (i.e. where complete = not getting updated)
    # endDate = datetime.today() - timedelta(days=1)
    endDate = parse('2020-12-23')

    # Folder where all files are stored
    folder = "Data/Data"

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

            print(pathname)
            if(path.exists(pathname)):
                data = np.genfromtxt(pathname, dtype=dataTypes,
                                     delimiter=',', names=True, usecols=np.arange(0, 10))
            else:
                print("Could not find: " + pathname)

            if(len(DataDictionary[coin]) == 0):
                DataDictionary[coin] = data
            else:
                DataDictionary[coin] = np.append(DataDictionary[coin], data)
            date = date + timedelta(days=1)
    return DataDictionary
}


def durationValues(){
    # Generate duration values
    minInDay = 24*60
    minInHour = 60
    DurationMin = {}

    NumDays = 7
    for i in range(1, NumDays+1):
        DurationMin.update({str(i) + "D": i*minInDay})
    NumHoursInDay = 24
    for i in range(1, NumHoursInDay):
        DurationMin.update({str(i) + "h": i*minInDay})
    return DurationMin
}


def additionalMetrics(rollingData, frameSizes) {
    #######################################
    # Calculate velocity for all durations
    #######################################
    start_total = time.time()
    for duration in frameSizes.keys():
        print("key: " + duration)
        print("duration: " + str(frameSizes.get(duration)))
        start_loop = time.time()

        # Velocity
        rollingData["mark_price_" + duration + "_velocity"] = rollingData["mark_price"].rolling(window=duration, min_periods=100) \
    .apply(lambda x: (x[-1]-x[0])/frameSizes.get(duration))
        # mean
        rollingData["mark_price_" + duration + "_mean"] = rollingData["mark_price"].rolling(
            window=duration, min_periods=100).mean()
        # std
        rollingData["mark_price_" + duration + "_std"] = rollingData["mark_price"].rolling(
            window=duration, min_periods=100).std()

        end_loop = time.time()
        loopTime = end_loop-start_loop
        print("loop time minutes: " + str(loopTime/60))
    end_total = time.time()
    print("total time minutes: " + str((end_total-start_total)/60))
    return rollingData
}


def processDataHelper(DataDictionary, coins) {
    for coin in coins:
        processData(DataDictionary[coin])
}


def processData(coinData) {
    # Create Pandas array
    pd_data = pd.DataFrame(data=coinData, columns=coinData.dtype.names)

    # Update Datatime column to pandas datetime type
    pd_data["datetime"] = pd.to_datetime(pd_data["datetime"])

    # Generate rolling data (i.e. frames from the data)
    rollingData = pd_data.drop(columns=['symbol', 'id']).set_index(
        "datetime").sort_index()

    # Generate frame sizes - the length of time between which data will be used to calculate additional values, such as mean, std, and velocity
    frameSizes = durationVa lues()

    additionalMetrics(rollingData, frameSizes)
}


def main() {
    # Define all coins
    # coins = ["BCH", "BSV", "BTC", "BTG", "DASH", "DOGE", "ETC", "ETH", "LSK", "LTC", "NEO", "OMG", "QTUM", "XLM", "XMR",
    #          "XRP", "ZEC"]
    coins = ["BCH"]

    DataDictionary = readData()
}
