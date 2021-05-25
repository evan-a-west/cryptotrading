from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from keras.preprocessing import sequence
# from numba import jit, njit, prange, float64, int32
# from numba.experimental import jitclass
# from numba import cuda, vectorize
# from scikeras.wrappers import KerasClassifier

# Dask stuff
import dask
from dask.distributed import Client, progress
from dask import delayed
import dask.dataframe as dd
import dask.array as da

import numpy as np


def main():
    client = Client(n_workers=8, threads_per_worker=2, memory_limit='1GB')
    print(client.dashboard_link)

    # filename = "processedData2021-03-31.csv"
    Root = "DataFromDeepLearningProcessing/DODGE/2021-05-02"

    dir = "/X_Train"
    path = Root + dir
    X_Train = da.from_npy_stack(path)

    dir = "/Y_Train"
    path = Root + dir
    Y_Train = da.from_npy_stack(path)

    dir = "/X_Test"
    path = Root + dir
    X_Test = da.from_npy_stack(path)

    dir = "/Y_Test"
    path = Root + dir
    Y_Test = da.from_npy_stack(path)

    dir = "/X_Test_Level_1"
    path = Root + dir
    X_Test_Final = da.from_npy_stack(path)

    dir = "/Y_Test_Level_1"
    path = Root + dir
    Y_Test_Final = da.from_npy_stack(path)

    print(X_Train.shape)
    print(Y_Train.shape)
    print(X_Test.shape)
    print(Y_Test.shape)
    print(type(X_Train))
    print(type(X_Train))

    num_features = 5
    batch_size = 100
    epochs = 100
    model = Sequential()
    model.add(LSTM(32, input_dim=num_features))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_Train[:, :, :num_features].compute(), Y_Train.compute(), validation_data=(
        X_Test[:, :, :num_features].compute(), Y_Test.compute()), epochs=epochs, batch_size=batch_size, verbose=2)


if __name__ == '__main__':
    main()
