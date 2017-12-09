import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def fit_LSTM(series, lag, train_percent, n_epochs, n_layers, return_all = False):
    def create_dataset(series, lag = 1):
        X, Y = [], []
        for i in range(len(series)-lag-1):
            a = series[i:(i+lag), 0]
            X.append(a)
            Y.append(series[i + lag, 0])
        return np.array(X), np.array(Y)

    scaler = MinMaxScaler(feature_range=(0, 1))
    series = scaler.fit_transform(series)
    
    train_size = int(len(series) * train_percent)
    test_size = len(series) - train_size
    train, test = series[0:train_size,:], series[train_size:len(series),:]

    trainX, trainY = create_dataset(train, lag)
    testX, testY = create_dataset(test, lag)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    model = Sequential()
    for i in range(n_layers - 1):
        model.add(LSTM(lag + 1, batch_input_shape=(1, lag, 1), stateful = True, return_sequences = True))
    model.add(LSTM(lag + 1, batch_input_shape=(1, lag, 1), stateful = True))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(n_epochs):
        model.fit(trainX, trainY, epochs = 1, batch_size = 1, verbose = 0, shuffle = False)
        model.reset_states()

    testPredict = model.predict(testX, batch_size = 1)

    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    if return_all:
        return testScore, testY, testPredict
    return testScore
