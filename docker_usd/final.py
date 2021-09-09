#!/usr/bin/env python
# coding: utf-8


from datetime import datetime

import requests
import numpy as np
import pandas as pd
from loguru import logger
from keras.layers import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


LAYERS_COUNT = 2


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/92.0.4515.159 Safari/537.36"
}


# transform date to datetime type
def to_datetime(df):
    return datetime.strptime(df, "%d.%m.%Y").strftime("%Y-%m-%d")


# api request and creating a table with date and exchange rate
def get_data():
    tables_list = []
    for month in range(1, 9, 1):
        dollar_rub = f"https://www.calc.ru/kotirovka-dollar-ssha.html?date=2020-0{month}"
        full_page = requests.get(dollar_rub, headers=HEADERS)
        tables = pd.read_html(full_page.text, match="Дата")[1]
        tables_list.append(tables)
    tables_concat = pd.concat(tables_list)
    tables_concat["Дата"] = tables_concat["Дата"].apply(lambda x: to_datetime(x))
    tables_concat = tables_concat.sort_values("Дата").reset_index(drop=True)

    return tables_concat


def create_model(shape):
    model = Sequential()
    # Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
    # Units - dimensionality of the output space

    model.add(LSTM(units=20, return_sequences=True, input_shape=(shape, 1)))
    model.add(Dropout(0.2))

    for _ in range(LAYERS_COUNT):
        model.add(LSTM(units=20, return_sequences=True))
        model.add(Dropout(0.2))

    model.add(LSTM(units=20))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units=1))
    model.summary()

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def visualize_data(full_data):
    plt.figure(
        figsize=(
            20,
            7,
        )
    )
    plt.plot(
        full_data["Дата"],
        full_data["Курс"].values,
        label="Exchange Rate (Rub/USD)",
        color="red",
    )
    # plt.xticks(np.arange(100,full_data.shape[0],300))
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate (Rub)")
    plt.legend()
    plt.savefig('Data.png')


def visualize_results(full_data, df_volume, predict):
    plt.figure(figsize=(20, 7))
    plt.plot(
        full_data["Дата"].values[200:],
        df_volume[200:],
        color="red",
        label="Real Exchange Rate",
    )
    plt.plot(
        full_data["Дата"][-predict.shape[0]:].values,
        predict,
        color="blue",
        label="Predicted Exchange Rate",
    )
    plt.xticks(np.arange(100, full_data[200:].shape[0], 200))
    plt.title("Exchange prediction")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.savefig('Results.png')


def main():
    # try_except block for logger writting
    try:
        full_data = get_data()
        logger.info("Exchange data has been successfully loaded")
    except Exception as exc:
        logger.error(f"Can't load data, because: {str( exc )}")
        logger.exception(exc)
        raise exc

    full_data["Дата"] = full_data["Дата"].astype("datetime64")

    # make a plot with exchange rate by date
    visualize_data(full_data)

    num_shape = 200

    # creating train/test data
    train = full_data.iloc[:num_shape, 1:2].values
    test = full_data.iloc[num_shape:, 1:2].values

    # try_except block for writing scalar_transform into logger
    try:
        sc = MinMaxScaler(feature_range=(0, 1))
        train_scaled = sc.fit_transform(train)
        logger.info("Exchange data has been successfully transformed")
    except TypeError as exc:
        logger.error(f"Can't transform data, because: {str( exc )}")
        logger.exception(exc)
        raise exc

    X_train = []

    # Price on next day
    y_train = []

    window = 10

    for i in range(window, num_shape):
        X_train_ = np.reshape(train_scaled[i - window: i, 0], (window, 1))
        X_train.append(X_train_)
        y_train.append(train_scaled[i, 0])
    X_train = np.stack(X_train)
    y_train = np.stack(y_train)

    # Initializing the Recurrent Neural Network
    model = create_model(X_train.shape[1])

    model.fit(X_train, y_train, epochs=1000, batch_size=32)

    df_volume = np.vstack((train, test))

    inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    num_2 = df_volume.shape[0] - num_shape + window

    X_test = []

    for i in range(window, num_2):
        X_test_ = np.reshape(inputs[i - window: i, 0], (window, 1))
        X_test.append(X_test_)

    X_test = np.stack(X_test)

    predict = model.predict(X_test)

    predict = sc.inverse_transform(predict)
    logger.info(f"Predicted value: {predict}")

    diff = predict - test

    logger.info(f"MSE: {np.mean(diff ** 2)}")
    logger.info(f"MAE: {np.mean(abs(diff))}")
    logger.info(f"RMSE: {np.sqrt(np.mean(diff ** 2))}")

    visualize_results(full_data, df_volume, predict)


if __name__ == "__main__":
    main()
