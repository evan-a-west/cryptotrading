import dask
from dask.distributed import Client, progress
from dask import delayed
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from scipy import signal
from dask.dataframe.utils import make_meta
import os

##########################
# Constants
##########################
# percentage of MIN and MAX data devoted to training
TRAIN_SIZE_CRITICAL_POINTS = .8
# Size of the non-critical point data set in multiples of the critical-point dataset size (i.e. for undersampling the non-critical point dataset)
NEITHER_SIZE_MULTIPLE = 10
TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE = 120


def trainTestSplitAClass(indexArray, splitIndex, endIndex):
    print(splitIndex, endIndex)
    if(np.isnan(endIndex)):
        Train = indexArray[:splitIndex]
        Test = indexArray[splitIndex:]
    else:
        Train = indexArray[:splitIndex]
        Test = indexArray[splitIndex:endIndex]
    print("Train Size: " + str(len(Train)))
    print("Test Size: " + str(len(Test)))

    return Train, Test


def main():
    client = Client(n_workers=8, threads_per_worker=2, memory_limit='1GB')
    print(client)

    data_resampled_dask = dd.read_csv(
        "DataFromDeepLearningProcessing/DOGE_Deep_2021-04-16.csv")

    data_resampled_dask["datetime"] = dd.to_datetime(
        data_resampled_dask["datetime"])
    data_resampled_dask["datetimeNotTheIndex"] = dd.to_datetime(
        data_resampled_dask["datetimeNotTheIndex"])

    data_resampled_dask = data_resampled_dask.set_index(
        "datetime", sorted=True)

    training_columns = ['mark_price', 'ask_price', 'bid_price', 'spread',
                        'mark_price_10T_velocity', 'mark_price_60T_velocity',
                        'mark_price_1440T_velocity', 'mark_price_10T_mean',
                        'mark_price_60T_mean', 'mark_price_1440T_mean', 'mark_price_10T_std',
                        'mark_price_60T_std', 'mark_price_1440T_std',
                        'mark_price_10T_acceleration_for_10T_velocity',
                        'mark_price_60T_acceleration_for_60T_velocity', "minmax"]

    data_resampled_dask = data_resampled_dask[training_columns]

    # min_indices = np.array(
    #     data_resampled_dask[data_resampled_dask["minmax"] == 1].index.compute().tolist())
    # max_indices = np.array(
    #     data_resampled_dask[data_resampled_dask["minmax"] == 2].index.compute().tolist())
    # neither_indices = np.array(
    #     data_resampled_dask[data_resampled_dask["minmax"] == 0].index.compute().tolist())

    X = data_resampled_dask.drop("minmax", axis=1).to_dask_array().compute()
    Y = data_resampled_dask["minmax"].to_dask_array().compute()

    # The level 1 train/test split should NOT be shuffled! We want to test the final model on unshuffled data
    SpliteIndex = int(len(Y)*TRAIN_SIZE_CRITICAL_POINTS)
    X_train_level_1 = X[:SpliteIndex, :]
    Y_train_level_1 = Y[:SpliteIndex]

    X_test_level_1 = da.from_array(X[SpliteIndex:, :])
    Y_test_level_1 = da.from_array(Y[SpliteIndex:])

    minIndices = np.where(Y_train_level_1 == 1)[0]
    # only take indices above TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE (i.e. min window size). Also remove final index, if present, since cannot build the window off that. Also remove final index, if present, since cannot build the window off that
    minIndices = minIndices[np.where(
        minIndices >= TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE)]

    maxIndices = np.where(Y_train_level_1 == 2)[0]
    # Only take indices above TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE (i.e. min window size). Also remove final index, if present, since cannot build the window off that
    maxIndices = maxIndices[np.where(
        maxIndices >= TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE)]

    neitherIndices = np.where(Y_train_level_1 == 0)[0]
    # Only take indices above TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE (i.e. min window size). Also remove final index, if present, since cannot build the window off that
    neitherIndices = neitherIndices[np.where(
        neitherIndices >= TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE)]

    # Shuffle training indices
    np.random.shuffle(minIndices)
    np.random.shuffle(maxIndices)
    np.random.shuffle(neitherIndices)

    # Split the training indices by class. Then split those classes into train and test sets.
    # The purpose of this is to ensure there is a balance of all 3 classes in both the train and test sets
    train_list = []
    test_list = []
    minSplitIndex = int(len(minIndices)*TRAIN_SIZE_CRITICAL_POINTS)
    maxSplitIndex = int(len(maxIndices)*TRAIN_SIZE_CRITICAL_POINTS)
    neitherEndIndex = int((minSplitIndex+maxSplitIndex)*NEITHER_SIZE_MULTIPLE)
    neitherSplitIndex = int(neitherEndIndex*TRAIN_SIZE_CRITICAL_POINTS)
    for theIndexArray, splitIndex, endIndex in zip(
        [minIndices, maxIndices, neitherIndices],
        [minSplitIndex, maxSplitIndex, neitherSplitIndex],
            [np.nan, np.nan, neitherEndIndex]):
        trainTemp, testTemp = trainTestSplitAClass(
            theIndexArray, splitIndex, endIndex)
        train_list.append(trainTemp)
        test_list.append(testTemp)

    # Recombine the class train indices into 1 array
    TrainIndices = np.append(train_list[0], train_list[1], axis=0)
    TrainIndices = np.append(TrainIndices, train_list[2], axis=0)
    np.random.shuffle(TrainIndices)

    # Recombine the class test indices into 1 array
    TestIndices = np.append(test_list[0], test_list[1], axis=0)
    TestIndices = np.append(TestIndices, test_list[2], axis=0)
    np.random.shuffle(TestIndices)

    # Build Train Windows
    numWindowsTrain = len(TrainIndices)
    numColumns = X_train_level_1.shape[1]
    X_Train = np.zeros(
        [numWindowsTrain, TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE, numColumns])
    Y_Train = np.empty(shape=numWindowsTrain, dtype=np.int32)
    for index, windowIndex in zip(range(0, len(TrainIndices)), TrainIndices):
        # The final row of of the window needs to include windowIndex, so build the start and stop indices accordingly
        windowStartIndex = windowIndex - TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE + 1

        # If the windowIndex happens to the be final index in the X array, then handle that situation
        if(windowIndex == len(X_train_level_1)-1):
            X_Train[index] = X_train_level_1[windowStartIndex:, :]
        # This is normal situation (i.e. windowIndex is not the final index in the array)
        else:
            windowEndIndex = windowIndex + 1
            X_Train[index] = X_train_level_1[windowStartIndex:windowEndIndex, :]
        # Take hte label from The final row in the window
        Y_Train[index] = Y_train_level_1[windowIndex]

        print("Index: " + str(index))
        data = X_Train[index, :, :]
        data_normalized = (data - data.min(axis=0)) / \
            (data.max(axis=0) - data.min(axis=0))
        X_Train[index, :, :] = data_normalized

    X_Train = da.from_array(X_Train)
    Y_Train = da.from_array(Y_Train)

    # Build Test Windows
    numWindowsTest = len(TestIndices)
    numColumns = X_train_level_1.shape[1]
    X_Test = np.zeros(
        [numWindowsTest, TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE, numColumns])
    Y_Test = np.empty(shape=numWindowsTest, dtype=np.int32)
    for index, windowIndex in zip(range(0, len(TestIndices)), TestIndices):
        # The final row of of the window needs to include windowIndex, so build the start and stop indices accordingly
        windowStartIndex = windowIndex - TRAIN_TEST_SPLIT__MIN_WINDOW_SIZE + 1

        # If the windowIndex happens to the be final index in the X array, then handle that situation
        if(windowIndex == len(X)-1):
            X_Test[index] = X_train_level_1[windowStartIndex:, :]
        # This is normal situation (i.e. windowIndex is not the final index in the array)
        else:
            windowEndIndex = windowIndex + 1
            X_Test[index] = X_train_level_1[windowStartIndex:windowEndIndex, :]
        # Take the label from The final row in the window
        Y_Test[index] = Y[windowIndex]

        print("Index: " + str(index))
        data = X_Test[index, :, :]
        data_normalized = (data - data.min(axis=0)) / \
            (data.max(axis=0) - data.min(axis=0))
        X_Test[index, :, :] = data_normalized

    X_Test = da.from_array(X_Test)
    Y_Test = da.from_array(Y_Test)

    current_date = str(date.today())
    dir = "DataFromDeepLearningProcessing/DODGE/" + current_date + "/X_Train"
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    da.to_npy_stack(dir, X_Train, axis=0)

    dir = "DataFromDeepLearningProcessing/DODGE/" + current_date + "/Y_Train"
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    da.to_npy_stack(dir, Y_Train, axis=0)

    dir = "DataFromDeepLearningProcessing/DODGE/" + current_date + "/X_Test"
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    da.to_npy_stack(dir, X_Test, axis=0)

    dir = "DataFromDeepLearningProcessing/DODGE/" + current_date + "/Y_Test"
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    da.to_npy_stack(dir, Y_Test, axis=0)

    dir = "DataFromDeepLearningProcessing/DODGE/" + current_date + "/X_Test_Level_1"
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    da.to_npy_stack(dir, X_test_level_1, axis=0)

    dir = "DataFromDeepLearningProcessing/DODGE/" + current_date + "/Y_Test_Level_1"
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    da.to_npy_stack(dir, Y_test_level_1, axis=0)


if __name__ == '__main__':
    main()
