import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import time
from dateutil.parser import parse
from os import path
from scipy import signal
from concurrent import futures
import functools
import logging
import inspect


class BuildColumns:
    def __init__(self, startDate=parse('2020-9-23'), endDate=date.today(), folder="Data/Data", fileExtension='.csv',
                 dataTypes=['f', 'f', 'f', 'f', 'f',
                            'f', 'U6', 'U36', 'i', 'U36'],
                 coins=["BCH", "BSV", "BTC", "BTG", "DASH", "DOGE", "ETC", "ETH", "LSK", "LTC", "NEO", "OMG", "QTUM", "XLM", "XMR", "XRP", "ZEC"]):
        self.startDate = startDate
        self.endDate = endDate
        self.folder = folder
        self.fileExtension = fileExtension
        self.dataTypes = dataTypes
        self.coins = coins

        logging.basicConfig(level=logging.INFO,
                            filename="log_" + str(date.today()) + ".log")

    def readDataForAllCoins(self):
        # Read and append all data file files
        empty = []
        DataDictionary = {}
        for coin in self.coins:
            date = self.startDate

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
                    day + "_" + month + "_" + year + self.fileExtension

                print(pathname)
                if(path.exists(pathname)):
                    data = np.genfromtxt(pathname, dtype=self.dataTypes,
                                         delimiter=',', names=True, usecols=np.arange(0, 10))
                else:
                    print("Could not find: " + pathname)

                if(len(DataDictionary[coin]) == 0):
                    DataDictionary[coin] = data
                else:
                    DataDictionary[coin] = np.append(
                        DataDictionary[coin], data)
                date = date + timedelta(days=1)
        return DataDictionary

    def buildColumns(self, coinData):
        pd_data = pd.DataFrame(data=coinData, columns=coinData.dtype.names)

        # Update the Datetime column
        pd_data["datetime"] = pd.to_datetime(pd_data["datetime"])

        # Drop uneeded columns. Make the Datetime the index column for use in rolling data frames
        rollingData = pd_data.drop(columns=['symbol', 'id']).set_index(
            "datetime").sort_index()

        windowSizeInMinutes = self.__windowSizes()

        # Multi-processing Run, CPU
        future_velocity_columns = {}
        future_mean_columns = {}
        future_std_columns = {}

        start_total = time.time()

        with futures.ProcessPoolExecutor() as pool:
            future_velocity_columns = pool.submit(self.multiProcessVelocityCalc,
                                                  windowSizeInMinutes, rollingData)
            future_mean_columns = pool.submit(self.multiProcessMeanCalc,
                                              windowSizeInMinutes, rollingData)
            future_std_columns = pool.submit(self.multiProcessSTDCalc,
                                             windowSizeInMinutes, rollingData)

        end_total = time.time()
        totalTime = (end_total-start_total)
        self.logTime(
            "All Columns, Multi-Process Execution", totalTime)

        pd_data = pd.concat(
            [rollingData,
             future_velocity_columns.result(),
             future_mean_columns.result(),
             future_std_columns.result()],
            axis=1, join='outer')

        return pd_data

    @staticmethod
    def __windowSizes():
        # Generate duration values
        # minInDay = 24*60
        minInHour = 60
        windowSizeInMinutes = {}
        pandasRollingMinuteIdentifier = 'T'

        # generate data windows from 0 to <NumHours>
        # NumHours = 24
        NumHours = 3
        # The data windows grow by <NumMinInIncrement> minutes each loop
        NumMinInIncrement = 10
        for i in range(0, NumHours):
            for j in range(NumMinInIncrement, minInHour, NumMinInIncrement):
                numMin = (i*minInHour + j)
                windowSizeInMinutes.update(
                    {str(numMin) + pandasRollingMinuteIdentifier: numMin})

        return windowSizeInMinutes

    def calcVelocity(self, windowSize, rollingData, duration):
        ############################
        # Calc Velocity
        ############################
        startVelocityTime = time.time()
        columnNameVelocity = "mark_price_" + duration + "_velocity"
        newColumn = rollingData["mark_price"].rolling(window=duration, min_periods=100) \
            .apply(lambda x: (x[-1]-x[0])/windowSize.get(duration))
        endVelocityTime = time.time()
        self.logTime(columnNameVelocity, endVelocityTime-startVelocityTime)
        return {columnNameVelocity: newColumn}

    def calcMean(self, windowSize, rollingData, duration):
        ############################
        # Calc Mean
        ############################
        startMeanTime = time.time()
        columnNameMean = "mark_price_" + duration + "_mean"
        newColumn = rollingData["mark_price"].rolling(
            window=duration, min_periods=100).mean()
        endMeanTime = time.time()
        self.logTime(columnNameMean, endMeanTime-startMeanTime)
        return {columnNameMean: newColumn}

    def calcSTD(self, windowSize, rollingData, duration):
        ############################
        # Calc STD
        ############################
        startSTDTime = time.time()
        columnNameSTD = "mark_price_" + duration + "_std"
        newColumn = rollingData["mark_price"].rolling(
            window=duration, min_periods=100).std()
        endSTDTime = time.time()
        self.logTime(columnNameSTD, endSTDTime-startSTDTime)
        return {columnNameSTD: newColumn}

    def multiProcessVelocityCalc(self, windowSizeInMinutes, rollingData):
        #######################################
        # Calculate velocity for all durations
        #######################################
        start_total = time.time()
        newVelocityColumns = pd.DataFrame()
        with futures.ProcessPoolExecutor() as pool:
            for result in pool.map(functools.partial(self.calcVelocity, windowSizeInMinutes, rollingData), windowSizeInMinutes.keys()):
                newColumnName = list(result.keys())[0]
                newColumnData = result[newColumnName]
                newVelocityColumns[newColumnName] = newColumnData
        end_total = time.time()
        totalTime = (end_total-start_total)
        self.logTime(
            "All Velocity Columns, Multi-Process Execution", totalTime)
        return newVelocityColumns

    def multiProcessMeanCalc(self, windowSizeInMinutes, rollingData):
        #######################################
        # Calculate mean for all durations
        #######################################
        start_total = time.time()
        newMeanColumns = pd.DataFrame()
        with futures.ProcessPoolExecutor() as pool:
            for result in pool.map(functools.partial(self.calcMean, windowSizeInMinutes, rollingData), windowSizeInMinutes.keys()):
                newColumnName = list(result.keys())[0]
                newColumnData = result[newColumnName]
                newMeanColumns[newColumnName] = newColumnData
        end_total = time.time()
        totalTime = (end_total-start_total)
        self.logTime(
            "All Mean Columns, Multi-Process Execution", totalTime)
        return newMeanColumns

    def multiProcessSTDCalc(self, windowSizeInMinutes, rollingData):
        #######################################
        # Calculate STD for all durations
        #######################################
        start_total = time.time()
        newSTDColumns = pd.DataFrame()
        with futures.ProcessPoolExecutor() as pool:
            for result in pool.map(functools.partial(self.calcSTD, windowSizeInMinutes, rollingData), windowSizeInMinutes.keys()):
                newColumnName = list(result.keys())[0]
                newColumnData = result[newColumnName]
                newSTDColumns[newColumnName] = newColumnData
        end_total = time.time()
        totalTime = (end_total-start_total)
        self.logTime(
            "All Mean Columns, Multi-Process Execution", totalTime)
        return newSTDColumns

    def logTime(self, identifier, executionTime):
        # Record which function logged
        callingFunctionName = inspect.stack()[1][3]

        # Convert to minutes
        executionTimeMinutes = executionTime/60

        # Log!
        logging.info("Execution time for " + callingFunctionName +
                     " when generating " + identifier + ": " + str(executionTime) + " SECONDS or " + str(executionTimeMinutes) + " MINUTES")

    def readBuildSave(self):
        allCoinData = self.readDataForAllCoins()

        pd_data = allCoinData["BCH"]

        pd_data = self.buildColumns(pd_data)

        pd_data.to_csv("processedData" + str(date.today()) + ".csv")
