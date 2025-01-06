from sklearn.ensemble import RandomForestRegressor
ALPACA_API_KEY = #
ALPACA_SECRET_KEY = #
BASE_URL = "https://paper-api.alpaca.markets"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from alpaca_trade_api.rest import REST, TimeFrame
import tensorflow as tf


api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)


rf = RandomForestRegressor()
while True:
    bool1 = True
    while bool1:
        try:
            print("Enter A symbol")
            symbol = input()
            barset = api.get_bars(symbol, TimeFrame.Day, "2022-01-01", "2024-07-11").df
            print(barset)
            bool1 = False

        except:
            print("Try again")
    # Fetch SPY historical data

    data = np.array(barset)

    print(data[0][0])
    # Preprocess the data
    #scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data between 0 and 1
    #scaled_data = scaler.fit_transform(barset['close'].values.reshape(-1, 1))


    def create_dataset(data, time_step=365):
        X, Y = [], []
        var1 = 0
        for i in range(len(data)-time_step-7):
            X.append(data[i:(i+time_step)])
            #var = []
            Y.append(data[i+time_step+7][0]) # want close, used to be Y.append(data[i + time_step:i+time_step+7][1][0])
            if var1 == 0:
                print(data[i + time_step:i+time_step+7][1], 'a')
                print(data[i + time_step:i+time_step+7], 'hey')
                var1 += 1
            #Y.append(var)
        return np.array(X), np.array(Y)

    time_step = 60  # Use the last 60 days to predict the next day
    X, Y = create_dataset(data, time_step)

    # make X 2d
    X = np.mean(X,axis=1)

    #X = X.reshape((X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    #Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1]) #new
    #Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1]) #new

    print(data.shape)
    print(X_train.shape, Y.shape)
    #build model
    rf.fit(X_train, Y_train)
    a = rf.predict(X_train)
    print(Y_train[-1], a[-1])
    #b = rf.predict(X_test[-1].reshape(-1,1))
    print(X_test[-1])
    #print(b)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(Y_train,a)
    print(mse)

    #validaiton

    print(mean_squared_error(Y, rf.predict(X)))

    params = {
        'max_depth' : [15,25,100,125,200], 
        'min_samples_split' : [2,4,8],
        'bootstrap' : [True, False],
        'n_estimators': [100,150,200,400]

    }

    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X, Y)

    # Best hyperparameters found
    print("Best Hyperparameters:", grid_search.best_params_)

    # Best estimator
    best_rf = grid_search.best_estimator_

    print(mean_absolute_error(Y, best_rf.predict(X)))
    print(mean_absolute_error(Y_test, best_rf.predict(X_test)))


    barset = api.get_bars(symbol, TimeFrame.Day, "2024-07-01", "2024-07-22").df
    print(barset)

    print(X[-1:], X.shape)
    recent_pred = X[-1:]
    recent_pred = best_rf.predict(recent_pred)
    print(recent_pred, barset)

    data = np.array(barset)
