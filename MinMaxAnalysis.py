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

#################
# GLOBALS
#################
# Flags to indicate min/max columns
MIN_FLAG = 1
MAX_FLAG = 2
NEITHER_FLAG = 0
# Threshold percent for determining when to keep/delete a min or max value
THRESHOLD_PERCENTAGE = .5  # i.e. .5%


def readData(filename):
    #######################################
    # Read data from processedData.csv
    #######################################
    # Data types of columns to read from the folder
    dataTypes = ['U36', 'f', 'f', 'f', 'f', 'f', 'f', 'i',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f']
    data = np.genfromtxt(filename, dtype=dataTypes,
                         delimiter=',', names=True, usecols=np.arange(0, 98))

    pd_data = pd.DataFrame(data=data, columns=data.dtype.names)
    pd_data["datetime"] = pd.to_datetime(pd_data["datetime"])

    return pd_data


def generateMinmaxColumn(pd_data):
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
    pd_data['minmax'] = NEITHER_FLAG

    # Minmax column should be 1 for min
    pd_data.loc[pd_data['min'] != 0, 'minmax'] = MIN_FLAG

    # Minmax column should be 2 for max
    pd_data.loc[pd_data['max'] != 0, 'minmax'] = MAX_FLAG

    return pd_data


def minmaxThresholdCheck(pd_data):
    # grab just the rows with non-zero min and max values for easy comparison
    subset = pd_data.loc[(pd_data['minmax'] != NEITHER_FLAG)]

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
            current_minmax_index = pd_data[pd_data['datetime']
                                           == current_minmax['datetime']].index[0]
            next_minmax = subset.iloc[subset_row_counter+1]
            next_minmax_value = float(next_minmax['mark_price'])
            next_minmax_index = pd_data[pd_data['datetime']
                                        == next_minmax['datetime']].index[0]

            # If the current va lue and next value in the subset are different (i.e. if one is a MIN and the other is a MAX), then run threshold check
            if(current_minmax['minmax'] != next_minmax['minmax']):
                percentage_change = 100 * \
                    float(abs(next_minmax_value - current_minmax_value) /
                          current_minmax_value)

                # If threshold fails (i.e. if the difference between the mark_price of the current and next minmax values is less than the given threshold),
                # then delete the next minmax
                if(percentage_change < THRESHOLD_PERCENTAGE):
                    # Remove the next min/max value from being a min/max value, since the mark_price difference from the current index is not large enough
                    pd_data.at[next_minmax_index, "minmax"] = NEITHER_FLAG
                    subset = subset.drop(index=next_minmax_index)

                # Otherwise, exit the loop! Threshold passed
                else:
                    diff_greater_than_threshold = True

            # If the current value and next value in the subset are both MIN, then keep the lowest MIN value
            elif(current_minmax['minmax'] == 1 and next_minmax['minmax'] == 1):
                if(current_minmax_value <= next_minmax_value):
                    pd_data.at[next_minmax_index, "minmax"] = NEITHER_FLAG
                    subset = subset.drop(index=next_minmax_index)
                else:
                    # The current minmax is getting deleted for use the the next_minmax row will replace it
                    current_minmax = next_minmax
                    pd_data.at[current_minmax_index, "minmax"] = NEITHER_FLAG
                    subset = subset.drop(
                        index=current_minmax_index)
            # If the current value and next value in the subset are both MAX, then keep the highest MAX value
            elif(current_minmax['minmax'] == 2 and next_minmax['minmax'] == 2):
                if(current_minmax_value >= next_minmax_value):
                    pd_data.at[next_minmax_index, "minmax"] = NEITHER_FLAG
                    subset = subset.drop(index=next_minmax_index)
                else:
                    # The current minmax is getting deleted for use the the next_minmax row will replace it
                    current_minmax = next_minmax
                    pd_data.at[current_minmax_index, "minmax"] = NEITHER_FLAG
                    subset = subset.drop(
                        index=current_minmax_index)
            else:
                print(
                    "ERROR!!! SHOULD NEVER SEE THIS! Logic failed in minmaxThresholdCheck")

            # reset subSetSize
            subsetSize = subset.shape[0]

        subset_row_counter = subset_row_counter + 1

    return pd_data


def main():
    filename = "processedData_small.csv"

    pd_data = readData(filename)

    pd_data = generateMinmaxColumn(pd_data)

    pd_data = minmaxThresholdCheck(pd_data)


if __name__ == "__main__":
    main()
