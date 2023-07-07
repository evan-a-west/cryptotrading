import dask
from dask.distributed import Client, progress
from dask import delayed
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, timezone
from scipy import signal
from dask.dataframe.utils import make_meta

##################################
# MinMax GLOBALS
##################################
# Flags to indicate min/max columns
MIN_FLAG = 1
MAX_FLAG = 2
NEITHER_FLAG = 0
# Threshold percent for determining when to keep/delete a min or max value
# i.e. if .5 is put here, that translates to .5% in the code
THRESHOLD_PERCENTAGE = .25
SIGNAL_MAX_ORDER = 25
SIGNAL_MIN_ORDER = 25

# Gapsize for spliting data
GAP_SIZE_MINUTES = 7
RESAMPLE_PERIOD = '30S'
INTERPOLATION_METHOD = 'linear'


def splitDataByGaps(data):
    # Covert from a datetime index back to a 'standard' index for use the for loop, below. This seems to be the easier fix. Datetime index does not work for the iloc
    deltas = data.set_index(np.array(range(0, data.shape[0], 1)))[
        "datetimeNotTheIndex"].diff()[0:]
    gaps = deltas[deltas > timedelta(minutes=GAP_SIZE_MINUTES)]

    data_split = list()
    start_index = 0
    for gap_index in gaps.index:
        data_split.append(data.iloc[start_index:gap_index, :])
        start_index = gap_index
    # Append the final split
    data_split.append(data.iloc[start_index:, :])

    return data_split


def resampleAndInterpolate(data):
    resample_index = pd.date_range(
        start=data.index[0],  end=data.index[-1], freq=RESAMPLE_PERIOD)
    dummy_data = pd.DataFrame(
        np.NAN, index=resample_index, columns=data.columns)
    intermediateResample = data.combine_first(
        dummy_data).interpolate('time')
    finalResample = intermediateResample.resample(
        rule=RESAMPLE_PERIOD, origin=data.index[0]).asfreq()
    # data = data.interpolate(self.INTERPOLATION_METHOD)
    return finalResample


def generateMinmMaxColumn(pd_data):
    pd_data = generateMinmaxColumn(pd_data)
    pd_data = dd.from_pandas(minmaxThresholdCheck(
        pd_data.compute()), npartitions=8)
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
    pd_data['minmax'] = 0

    # Prep intermediate data objects for storing min and max
    tempSeries = pd_data["minmax"]
    tempNumpy = tempSeries.compute().to_numpy()
    tempindex = tempSeries.compute().index

    # Minmax column should be 1 for min
    minData = signal.argrelmin(
        data=pd_data["mark_price"].values.compute(), order=SIGNAL_MAX_ORDER, mode='clip')

    # Minmax column should be 2 for max
    maxData = signal.argrelmax(
        data=pd_data["mark_price"].values.compute(), order=SIGNAL_MIN_ORDER, mode='clip')

    tempNumpy[minData] = MIN_FLAG
    tempNumpy[maxData] = MAX_FLAG
    newTempSeries = pd.Series(tempNumpy)
    newTempSeries.index = tempindex
    pd_data["minmax"] = newTempSeries

    return pd_data

# @jit(nopython=True, parallel=True)


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
            current_minmax_index = pd_data[pd_data['datetimeNotTheIndex']
                                           == current_minmax['datetimeNotTheIndex']].index[0]
            next_minmax = subset.iloc[subset_row_counter+1]
            next_minmax_value = float(next_minmax['mark_price'])
            next_minmax_index = pd_data[pd_data['datetimeNotTheIndex']
                                        == next_minmax['datetimeNotTheIndex']].index[0]

            # If the current value and next value in the subset are different (i.e. if one is a MIN and the other is a MAX), then run threshold check
            if(current_minmax['minmax'] != next_minmax['minmax']):
                percentage_change = 100 * \
                    float(abs(next_minmax_value - current_minmax_value) /
                          current_minmax_value)

                # If threshold fails (i.e. if the difference between the mark_price of the current and next minmax values is less than the given threshold),
                # then delete the next minmax
                if(percentage_change < THRESHOLD_PERCENTAGE):
                    # Remove the next min/max value from being a min/max value, since the mark_price difference from the current index is not large enough
                    pd_data.at[next_minmax_index,
                               "minmax"] = NEITHER_FLAG
                    subset = subset.drop(index=next_minmax_index)

                # Otherwise, exit the loop! Threshold passed
                else:
                    diff_greater_than_threshold = True

            # If the current value and next value in the subset are both MIN, then keep the lowest MIN value
            elif(current_minmax['minmax'] == 1 and next_minmax['minmax'] == 1):
                if(current_minmax_value <= next_minmax_value):
                    pd_data.at[next_minmax_index,
                               "minmax"] = NEITHER_FLAG
                    subset = subset.drop(index=next_minmax_index)
                else:
                    # The current minmax is getting deleted for use the the next_minmax row will replace it
                    current_minmax = next_minmax
                    pd_data.at[current_minmax_index,
                               "minmax"] = NEITHER_FLAG
                    subset = subset.drop(
                        index=current_minmax_index)
            # If the current value and next value in the subset are both MAX, then keep the highest MAX value
            elif(current_minmax['minmax'] == 2 and next_minmax['minmax'] == 2):
                if(current_minmax_value >= next_minmax_value):
                    pd_data.at[next_minmax_index,
                               "minmax"] = NEITHER_FLAG
                    subset = subset.drop(index=next_minmax_index)
                else:
                    # The current minmax is getting deleted for use the the next_minmax row will replace it
                    current_minmax = next_minmax
                    pd_data.at[current_minmax_index,
                               "minmax"] = NEITHER_FLAG
                    subset = subset.drop(
                        index=current_minmax_index)
            else:
                print(
                    "ERROR!!! SHOULD NEVER SEE THIS! Logic failed in minmaxThresholdCheck")

            # reset subSetSize
            subsetSize = subset.shape[0]

        subset_row_counter = subset_row_counter + 1

    return pd_data

    # Upsample data to every 30 seconds (exactly) using linear interpolation and Undersample the entire     on those 30


def main():

    client = Client(n_workers=8, threads_per_worker=2, memory_limit='1GB')
    print(client)

    df = dd.read_csv("ProcessedData/DOGE_2021-04-16.csv")
    df["datetime"] = dd.to_datetime(df["datetime"])
    df["datetimeNotTheIndex"] = dd.to_datetime(
        df["datetimeNotTheIndex"])

    df = df.set_index("datetime", sorted=True)

    training_columns = ['mark_price', 'ask_price', 'bid_price', 'spread',
                        'mark_price_10T_velocity', 'mark_price_60T_velocity',
                        'mark_price_1440T_velocity', 'mark_price_10T_mean',
                        'mark_price_60T_mean', 'mark_price_1440T_mean', 'mark_price_10T_std',
                        'mark_price_60T_std', 'mark_price_1440T_std',
                        'mark_price_10T_acceleration_for_10T_velocity',
                        'mark_price_60T_acceleration_for_60T_velocity', "datetimeNotTheIndex"]

    df_train = df[training_columns]

    data = df_train.compute()
    # Gapsize for spliting data
    GAP_SIZE_MINUTES = 7
    RESAMPLE_PERIOD = '30S'
    # INTERPOLATION_METHOD = 'linear'

    data_split = splitDataByGaps(data)

    data_resampled = pd.DataFrame(columns=training_columns)
    for data in data_split:
        dataReducedCol = data.drop("datetimeNotTheIndex", axis=1)
        # dataReducedCol = data[training_columns]
        dataResamples = resampleAndInterpolate(dataReducedCol)
        data_resampled = pd.concat([data_resampled, dataResamples])

    # Regenerate the "datetimeNotTheIndex" column for use in the minmax calculations
    data_resampled["datetimeNotTheIndex"] = data_resampled.index
    print(type(data_resampled.index))
    # data_resampled.drop(np.nan()) ####Need to drop NAs!
    data_resampled_dask = dd.from_pandas(data_resampled, npartitions=8)
    data_resampled_dask = generateMinmMaxColumn(data_resampled_dask)

    data_resampled_dask.to_csv(
        "ProcessedDataForDeepLearning/DOGE_Deep_2021-04-16.csv")


if __name__ == '__main__':
    main()
