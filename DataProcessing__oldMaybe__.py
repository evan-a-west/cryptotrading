import dask
from dask.distributed import Client, progress
from dask import delayed
import dask.dataframe as dd


# class DataProcessing:

#     def __init__(self, startDate=parse('2020-9-23'), endDate=date.today(), folder="Data", fileExtension='.csv',
#                  dataTypes=['f', 'f', 'f', 'f', 'f',
#                             'f', 'U6', 'U36', 'i', 'U36'],
#                  coins=["BCH", "BSV", "BTC", "BTG", "DASH", "DOGE", "ETC", "ETH",
#                         "LSK", "LTC", "NEO", "OMG", "QTUM", "XLM", "XMR", "XRP", "ZEC"],
#                  MIN_WINDOW_SIZE=30):  # Note: 30 is the default for MIN_WINDOW_SIZE to support windows sizes of 10 minutes
#         self.startDate = startDate
#         self.endDate = endDate
#         self.folder = folder
#         self.fileExtension = fileExtension
#         self.dataTypes = dataTypes
#         self.coins = coins
#         self.MIN_WINDOW_SIZE = MIN_WINDOW_SIZE

# def readDataFromDropBox(directory, startDate, endDateInclusive):
#     # Read and append all data file files
#     empty = []

#     date = startDate

#     DataDictionary[coin] = np.asarray(empty)

#     while(date <= endDateInclusive):
#         day = date.day
#         if(day < 10):
#             day = '0' + str(day)
#         else:
#             day = str(day)

#         month = date.month
#         if(month < 10):
#             month = '0' + str(month)
#         else:
#             month = str(month)

#         year = str(date.year)

#         pathname = self.folder + "/" + coin + "/" + coin + "_" + \
#             year + "_" + month + "_" + day + self.fileExtension

#         self.logInfo("Pathname: " + pathname)
#         print("Pathname: " + pathname)
#         if(path.exists(pathname)):
#             data = np.genfromtxt(pathname, dtype=self.dataTypes,
#                                  delimiter=',', names=True, usecols=np.arange(0, 10))

#             if(len(DataDictionary[coin]) == 0):
#                 DataDictionary[coin] = data
#             else:
#                 DataDictionary[coin] = np.append(
#                     DataDictionary[coin], data)
#         else:
#             this_function_name = sys._getframe().f_code.co_name
#             self.logError("In " + this_function_name +
#                           " when reading data, Could not find: " + pathname)
#             print("In " + this_function_name +
#                   " when reading data, Could not find: " + pathname)

#         date = date + timedelta(days=1)
#     return data


def main():
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='1GB')
    print(client)
    df = dd.read_csv("ProcessedData/DOGE_2021-04-16.csv")
    df = df.set_index("datetime", sorted=True)
    training_columns = ['mark_price', 'ask_price', 'bid_price', 'spread',
                        'mark_price_10T_velocity', 'mark_price_60T_velocity',
                        'mark_price_1440T_velocity', 'mark_price_10T_mean',
                        'mark_price_60T_mean', 'mark_price_1440T_mean', 'mark_price_10T_std',
                        'mark_price_60T_std', 'mark_price_1440T_std',
                        'mark_price_10T_acceleration_for_10T_velocity',
                        'mark_price_60T_acceleration_for_60T_velocity']

    df_train = df[training_columns]

    df_train_resampled = df_train.resample(
        '30S', label='right', closed='right')

    print(df_train_resampled)

    df_train_resampled.


if __name__ == "__main__":
    main()
