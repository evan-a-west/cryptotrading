from datetime import datetime, timedelta, date
from dateutil.parser import parse
import numpy as np
from os import path
import sys


class BasicPreprocessing:
    # Note: 30 is the default for MIN_WINDOW_SIZE to support windows sizes of 10 minutes
    def __init__(self, folder="Data", fileExtension='.csv',
                 dataTypes=['f', 'f', 'f', 'f', 'f',
                            'f', 'U6', 'U36', 'i', 'U36'],
                 coins=["BCH", "BSV", "BTC", "BTG", "DASH", "DOGE", "ETC", "ETH",
                        "LSK", "LTC", "NEO", "OMG", "QTUM", "XLM", "XMR", "XRP", "ZEC"],
                 MIN_WINDOW_SIZE=30):
        self.absoluteStartDate = parse('2020-9-23')
        self.absoluteEndDate = date.today()
        self.folder = folder
        self.fileExtension = fileExtension
        self.dataTypes = dataTypes
        self.coins = coins
        self.MIN_WINDOW_SIZE = MIN_WINDOW_SIZE

    @staticmethod
    def getListOfAllCoins(self):
        robinhoodCoins = self.getListofRobinhoodCoints()
        otherCoins = self.getOtherCoins()  # Placeholder for other data collections types

        return robinhoodCoins + otherCoins

    @staticmethod
    def getListofRobinhoodCoints():
        return ["BCH", "BSV", "BTC", "BTG", "DASH", "DOGE", "ETC", "ETH",
                "LSK", "LTC", "NEO", "OMG", "QTUM", "XLM", "XMR", "XRP", "ZEC"]

    @staticmethod
    # Placeholder for other forms of datacollection. Replace with service used when implemented.
    def getOtherCoins():
        return []

    def readDataForACoin(self, coin, startDateToFetch, endDateToFetch):
        date = startDateToFetch

        filePaths = {}
        while(date <= endDateToFetch):
            # Generate day string for filenames
            day = date.day
            if(day < 10):
                day = '0' + str(day)
            else:
                day = str(day)

            # Generate month string for filenames
            month = date.month
            if(month < 10):
                month = '0' + str(month)
            else:
                month = str(month)

            # Generate year string for filenames
            year = str(date.year)

            # Generate path to file
            pathname = self.folder + "/" + coin + "/" + coin + "_" + \
                year + "_" + month + "_" + day + self.fileExtension

            self.logInfo("Pathname: " + pathname)
            print("Pathname: " + pathname)
            if(path.exists(pathname)):
                data = np.genfromtxt(pathname, dtype=self.dataTypes,
                                     delimiter=',', names=True, usecols=np.arange(0, 10))

                if(len(DataDictionary[coin]) == 0):
                    DataDictionary[coin] = data
                else:
                    DataDictionary[coin] = np.append(
                        DataDictionary[coin], data)
            else:
                this_function_name = sys._getframe().f_code.co_name
                self.logError("In " + this_function_name +
                              " when reading data, Could not find: " + pathname)
                print("In " + this_function_name +
                      " when reading data, Could not find: " + pathname)

            date = date + timedelta(days=1)

    def readDataSpecifiedCoins(self, coinsToFetch, startDateToFetch, endDateToFetch):
        # Read and append all data file files
        empty = []
        DataDictionary = {}

        for coin in coinsToFetch:
            date = startDateToFetch

            DataDictionary[coin] = np.asarray(empty)

            while(date <= self.endDate):
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

                pathname = self.folder + "/" + coin + "/" + coin + "_" + \
                    year + "_" + month + "_" + day + self.fileExtension

                self.logInfo("Pathname: " + pathname)
                print("Pathname: " + pathname)
                if(path.exists(pathname)):
                    data = np.genfromtxt(pathname, dtype=self.dataTypes,
                                         delimiter=',', names=True, usecols=np.arange(0, 10))

                    if(len(DataDictionary[coin]) == 0):
                        DataDictionary[coin] = data
                    else:
                        DataDictionary[coin] = np.append(
                            DataDictionary[coin], data)
                else:
                    this_function_name = sys._getframe().f_code.co_name
                    self.logError("In " + this_function_name +
                                  " when reading data, Could not find: " + pathname)
                    print("In " + this_function_name +
                          " when reading data, Could not find: " + pathname)

                date = date + timedelta(days=1)
        return DataDictionary
