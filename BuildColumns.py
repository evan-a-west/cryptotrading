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
from concurrent_log_handler import ConcurrentRotatingFileHandler
from concurrent_log_handler.queue import setup_logging_queues
import os
import sys
from numba import jit
from scipy import signal


class BuildColumns:
    # Note: 30 is the default for MIN_WINDOW_SIZE to support windows sizes of 10 minutes
    def __init__(self, startDate=parse('2020-9-23'), endDate=date.today(), folder="Data", fileExtension='.csv',
                 dataTypes=['f', 'f', 'f', 'f', 'f',
                            'f', 'U6', 'U36', 'i', 'U36'],
                 coins=["BCH", "BSV", "BTC", "BTG", "DASH", "DOGE", "ETC", "ETH",
                        "LSK", "LTC", "NEO", "OMG", "QTUM", "XLM", "XMR", "XRP", "ZEC"],
                 MIN_WINDOW_SIZE=30):
        self.startDate = startDate
        self.endDate = endDate
        self.folder = folder
        self.fileExtension = fileExtension
        self.dataTypes = dataTypes
        self.coins = coins
        self.MIN_WINDOW_SIZE = MIN_WINDOW_SIZE

        ##################################
        # MinMax GLOBALS
        ##################################
        # Flags to indicate min/max columns
        self.MIN_FLAG = 1
        self.MAX_FLAG = 2
        self.NEITHER_FLAG = 0
        # Threshold percent for determining when to keep/delete a min or max value
        # i.e. if .5 is put here, that translates to .5% in the code
        self.THRESHOLD_PERCENTAGE = .25
        self.SIGNAL_MAX_ORDER = 25
        self.SIGNAL_MIN_ORDER = 25

        ##################################
        # Logging GLOBALS
        ##################################
        # logging.basicConfig(level=logging.INFO,
        #                     filename="log_" + str(date.today()) + ".log"

        # Create directories, as needed
        logDirectory = ""
        # if(not os.path.exists(logDirectory)):
        #     os.makedirs(logDirectory)

        logFileSize = 1024*1024

        # Logger for general INFO
        self.info_logger = logging.getLogger(__name__)
        # Use an absolute path to prevent file rotation trouble.
        logfile = os.path.abspath(
            logDirectory + "log_" + str(date.today()) + "_INFO.log")
        # Rotate log after reaching 512K, keep 5 old copies.
        rotateHandler = ConcurrentRotatingFileHandler(
            logfile, "a", logFileSize, 5)
        self.info_logger.addHandler(rotateHandler)
        self.info_logger.setLevel(logging.INFO)

        # Logger for ERRORs
        self.error_logger = logging.getLogger(__name__)
        # Use an absolute path to prevent file rotation trouble.
        logfile = os.path.abspath(
            logDirectory + "log_" + str(date.today()) + "_ERROR.log")
        # Rotate log after reaching 512K, keep 5 old copies.
        rotateHandler = ConcurrentRotatingFileHandler(
            logfile, "a", logFileSize, 5)
        self.error_logger.addHandler(rotateHandler)
        self.error_logger.setLevel(logging.ERROR)

        # convert all configured loggers to use a background thread
        setup_logging_queues()

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
                    year + "_" + month + "_" + day + self.fileExtension

                self.logInfo("Pathname: " + pathname)
                if(path.exists(pathname)):
                    data = np.genfromtxt(pathname, dtype=self.dataTypes,
                                         delimiter=',', names=True, usecols=np.arange(0, 10))
                else:
                    this_function_name = sys._getframe().f_code.co_name
                    self.logError("In " + this_function_name +
                                  " when reading data, Could not find: " + pathname)
                    continue

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

        # Make a separate datetime column that is not the index. Currenlty used in the MinMax analysis. Eventually, that analysis should just use the dataframe datetime index\
        pd_data["datetimeNotTheIndex"] = pd_data["datetime"]

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
        newColumn = rollingData["mark_price"].rolling(window=duration, min_periods=self.MIN_WINDOW_SIZE) \
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
            window=duration, min_periods=self.MIN_WINDOW_SIZE).mean()
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
            window=duration, min_periods=self.MIN_WINDOW_SIZE).std()
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

    def logInfo(self, message):
        self.info_logger.info(message)

    def logError(self, message):
        self.error_logger.error(message)

    def logTime(self, identifier, executionTime):
        # Record which fun tion logged
        callingFunctionName = inspect.stack()[1][3]

        # Convert to minutes
        executionTimeMinutes = executionTime/60

        # Log!
        self.info_logger.info("Execution time for " + callingFunctionName +
                              " when generating " + identifier + ": " + str(executionTime) + " SECONDS or " + str(executionTimeMinutes) + " MINUTES")

    def generateMinmaxColumn(self, pd_data):
        #######################################
        # Label data, for finding local maximum and minima.
        # New columns: Min, Max and Minmax
        #######################################
        pd_data['min'] = pd_data.mark_price[(pd_data.mark_price.shift(
            1) > pd_data.mark_price) & (pd_data.mark_price.shift(-1) > pd_data.mark_price)]
        pd_data['max'] = pd_data.mark_price[(pd_data.mark_price.shift(
            1) < pd_data.mark_price) & (pd_data.mark_price.shift(-1) < pd_data.mark_price)]

        # Min and Max column should not contain NANs
        pd_data['min'] = pd_data['min'].fillna(0)
        pd_data['max'] = pd_data['max'].fillna(0)

        # Minmax column should be 0 for neither min nor max
        pd_data['minmax'] = self.NEITHER_FLAG

        # Minmax column should be 1 for min
        # pd_data.loc[pd_data['min'] != 0, 'minmax'] = self.MIN_FLAG
        minData = signal.argrelmin(
            data=pd_data["mark_price"].values, order=self.SIGNAL_MAX_ORDER, mode='clip')
        pd_data.iloc[minData[0], pd_data.columns.get_loc(
            "minmax")] = self.MIN_FLAG

        # Minmax column should be 2 for max
        # pd_data.loc[pd_data['max'] != 0, 'minmax'] = self.MAX_FLAG
        maxData = signal.argrelmax(
            data=pd_data["mark_price"].values, order=self.SIGNAL_MIN_ORDER, mode='clip')
        pd_data.iloc[maxData[0], pd_data.columns.get_loc(
            "minmax")] = self.MAX_FLAG

        return pd_data

    # @jit(nopython=True, parallel=True)
    def minmaxThresholdCheck(self, pd_data):
        # grab just the rows with non-zero min and max values for easy comparison
        subset = pd_data.loc[(pd_data['minmax'] != self.NEITHER_FLAG)]

        # For each row in subset, check if each subsequent pair of of min/max values pass the given threshold
        subset_row_counter = 0
        subsetSize = subset.shape[0]
        while(subset_row_counter < subsetSize):
            diff_greater_than_threshold = False
            # Because Subset rows are removed in this loop, need to recheck the size here as an ending condition.
            # Note that subset_row_counter+1 is needed because we compare the current subset row against the next subset row
            while (not diff_greater_than_threshold) and (subset_row_counter+1 < subsetSize):
                current_minmax = subset.iloc[subset_row_counter]
                current_minmax_value = float(current_minmax['mark_price'])
                current_minmax_index = pd_data[pd_data['datetimeNotTheIndex']
                                               == current_minmax['datetimeNotTheIndex']].index[0]
                next_minmax = subset.iloc[subset_row_counter+1]
                next_minmax_value = float(next_minmax['mark_price'])
                next_minmax_index = pd_data[pd_data['datetimeNotTheIndex']
                                            == next_minmax['datetimeNotTheIndex']].index[0]

                # If the current va lue and next value in the subset are different (i.e. if one is a MIN and the other is a MAX), then run threshold check
                if(current_minmax['minmax'] != next_minmax['minmax']):
                    percentage_change = 100 * \
                        float(abs(next_minmax_value - current_minmax_value) /
                              current_minmax_value)

                    # If threshold fails (i.e. if the difference between the mark_price of the current and next minmax values is less than the given threshold),
                    # then delete the next minmax
                    if(percentage_change < self.THRESHOLD_PERCENTAGE):
                        # Remove the next min/max value from being a min/max value, since the mark_price difference from the current index is not large enough
                        pd_data.at[next_minmax_index,
                                   "minmax"] = self.NEITHER_FLAG
                        subset = subset.drop(index=next_minmax_index)

                    # Otherwise, exit the loop! Threshold passed
                    else:
                        diff_greater_than_threshold = True

                # If the current value and next value in the subset are both MIN, then keep the lowest MIN value
                elif(current_minmax['minmax'] == 1 and next_minmax['minmax'] == 1):
                    if(current_minmax_value <= next_minmax_value):
                        pd_data.at[next_minmax_index,
                                   "minmax"] = self.NEITHER_FLAG
                        subset = subset.drop(index=next_minmax_index)
                    else:
                        # The current minmax is getting deleted for use the the next_minmax row will replace it
                        current_minmax = next_minmax
                        pd_data.at[current_minmax_index,
                                   "minmax"] = self.NEITHER_FLAG
                        subset = subset.drop(
                            index=current_minmax_index)
                # If the current value and next value in the subset are both MAX, then keep the highest MAX value
                elif(current_minmax['minmax'] == 2 and next_minmax['minmax'] == 2):
                    if(current_minmax_value >= next_minmax_value):
                        pd_data.at[next_minmax_index,
                                   "minmax"] = self.NEITHER_FLAG
                        subset = subset.drop(index=next_minmax_index)
                    else:
                        # The current minmax is getting deleted for use the the next_minmax row will replace it
                        current_minmax = next_minmax
                        pd_data.at[current_minmax_index,
                                   "minmax"] = self.NEITHER_FLAG
                        subset = subset.drop(
                            index=current_minmax_index)
                else:
                    print(
                        "ERROR!!! SHOULD NEVER SEE THIS! Logic failed in minmaxThresholdCheck")

                # reset subSetSize
                subsetSize = subset.shape[0]

            subset_row_counter = subset_row_counter + 1

        return pd_data

    def normalize(self, pd_data):
        normalizedData = (pd_data - pd_data.min()) / \
            (pd_data.max() - pd_data.min())
        normalizedData["datetimeNotTheIndex"] = pd_data["datetimeNotTheIndex"]
        normalizedData["min"] = pd_data["min"]
        normalizedData["max"] = pd_data["max"]
        normalizedData["minmax"] = pd_data["minmax"]
        return normalizedData

    def generateMinmMaxColumn(self, pd_data):
        start_total = time.time()

        pd_data = self.generateMinmaxColumn(pd_data)

        pd_data = self.minmaxThresholdCheck(pd_data)

        end_total = time.time()
        totalTime = (end_total-start_total)
        self.logTime(
            "MinMax columns, Serial Execution", totalTime)

        return pd_data

    def readBuildSave(self):
        allCoinData = self.readDataForAllCoins()

        pd_data = allCoinData["BCH"]

        pd_data = self.buildColumns(pd_data)

        pd_data = self.generateMinmMaxColumn(pd_data)

        pd_data = self.normalize(pd_data)

        pd_data.to_csv("processedData" + str(date.today()) + ".csv")
