from txus_library import txuslib
from data_loading import data_loading as dload
from datetime import date
import config_data.constants as conf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from tensorflow.keras.models import Sequential  # pip install tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)
from datetime import datetime
import time


class DataLoadingError(Exception):
    pass


class ModelLoadingError(Exception):
    pass


class ScalerLoadingError(Exception):
    pass


class DataPrediction():
    def __init__(self, insts: list, meas: str, comp: str, field: str):
        self.ne = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['Training']['epochs']
        self.ma_type = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['Training']['ma_type']
        self.insts = insts
        self.meas = meas
        self.comp = comp
        self.field = field

    # TODO: decorator with different filters (least squares, fourier, etc.)
    # Long Short Term Memory deep learning network. Training method.
    def lstm_trainer(self, train_df: pd.DataFrame) -> str:
        print('[INFO] running: ', self.lstm_trainer)
        txuslib.delete_dir(conf.MODEL_DIR)
        training_set = np.array(train_df)
        training_set = training_set.reshape(-1, 1)
        sc = MinMaxScaler(feature_range=(0, 1))
        txuslib.create_dir(conf.MODEL_DIR)
        training_set_scaled = sc.fit_transform(training_set)
        dump(sc, conf.SCALER_PATH)

        x_train = []
        y_train = []
        for i in range(60, len(training_set_scaled)):
            x_train.append(training_set_scaled[i-60:i])  # i not included in range => i-60:i => from 0 to 59 for i=0
            y_train.append(training_set_scaled[i, 0])  # prediction of value 60

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape data structure

        regressor = Sequential()  # Initialize RNN

        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50))  # return sequences = False (because it is the last layer)
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=1))

        regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        # alternative --> RMSprop

        regressor.fit(x_train, y_train, batch_size=32, epochs=self.ne)

        score_train = regressor.evaluate(x_train, y_train, verbose=0)

        regressor.save(conf.MODEL_PATH)
        print("[INFO] Model correctly saved to disk")
        print("[INFO] %s during training: %.2f%%" % (regressor.metrics_names[1], score_train[1]*100))

        return score_train[1]*100

    # TODO
    # Long Short Term Memory deep learning network. Testing method
    def lstm_tester(self, test_df: pd. DataFrame) -> dict:

        loaded_model = load_model(conf.MODEL_PATH)
        loaded_model.summary()

        inputs = np.array(test_df)
        inputs = inputs.reshape(-1, 1)
        sc = load(conf.SCALER_PATH)

        inputs = sc.transform(inputs)

        x_test = []
        for i in range(60, len(inputs)):
            x_test.append(inputs[i-60:i])
        x_test = np.array(x_test)
        print('x_test.shape[0]', x_test.shape[0])
        print('x_test.shape[1]', x_test.shape[1])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape data structure
        # shape[0] = dimension of rows
        # shape[1] = dimension of columns
        # 1 is a new unit dimension axis

        print()

        predicted_data_set = loaded_model.predict(x_test)
        predicted_data_set = sc.inverse_transform(predicted_data_set)

        loaded_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        score_test = loaded_model.evaluate(x_test, predicted_data_set, verbose=0)
        print("[INFO] %s during testing: %.2f%%" % (loaded_model.metrics_names[1], score_test[1]*100))

        # plt.plot(test_df, color='blue', label='raw data')
        # plt.plot(predicted_data_set, color='red', label='predicted price data stream')
        # plt.show()
        return {'test_score': score_test[1]*100,
                'predicted_data': predicted_data_set}

    # predicts just one value based on a series of values inside a pandas dataframe
    # this method is thought to be called recurrently to predict a changing series in progress using "time" library
    def dynamic_predictor(self, pred_df: pd.DataFrame, model_path: str, scaler_path: str) -> float:
        loaded_model = self.lstm_load_model(model_path)
        loaded_scaler = self.lstm_load_scaler(scaler_path)
        inputs = pred_df.to_numpy()
        inputs = inputs.reshape(-1, 1)
        inputs = loaded_scaler.transform(inputs)
        #x_test = [inputs[-60:]]
        x_test = np.array(inputs)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape data structure
        pred_data_set = loaded_model.predict(x_test)
        pred_data_set = loaded_scaler.inverse_transform(pred_data_set)
        return pred_data_set[0][0]  # TODO: return a dictionary including the error incurred

    def lstm_load_model(self, h5_file_path: str) -> load_model:
        try:
            loaded_model = load_model(h5_file_path)
            loaded_model.summary()
            return loaded_model
        except ModelLoadingError:
            print('[ERROR] Model .h5 could not be loaded')

    def lstm_load_scaler(self, joblib_file_path):
        try:
            loaded_scaler = load(joblib_file_path)
            return loaded_scaler
        except ScalerLoadingError:
            print('[ERROR] Scaler .joblib could not be loaded')

    # splits the original dataframe into training and testing one
    # assuming that the lowest volume of data is Friday with 11500 values ==> at least 1500 values for testing
    @staticmethod
    def data_frame_splitter(data_frame: pd.DataFrame) -> dict:
        train_data = data_frame.iloc[:-1500, :]
        test_data = data_frame.iloc[-1500:, :]
        return {'train_data': train_data,
                'test_data': test_data}  # two data frames with train and test price data

    # TODO Unit testing
    # recovers the price data from the previous day to feed the training
    def get_training_data(self) -> dict:
        try:
            # Sunday's the query is different as it recovers data from Friday
            if datetime.now().strftime("%A") == 'Sunday':
                start_query = '-64h'
                stop_query = '-48h'
                dl = dload.DataLoading(self.insts, self.meas, self.comp, self.field,
                                       start_query=start_query, stop_query=stop_query)
                train_df_dict = dl.get_training_df()
            else:  # default values for start and stop query:
                dl = dload.DataLoading(self.insts, self.meas, self.comp, self.field)
                train_df_dict = dl.get_training_df()
            return train_df_dict  # dictionary of data frames corresponding to each instrument
        except DataLoadingError:
            print('[ERROR] Data could not be loaded from InfluxDB')

    # uses the data from previous day to train and test the prediction algorithm
    def daily_train_test(self):
        train_df_dict = self.get_training_data()
        split_df_dict = {}
        start_time = time.time()
        for key in train_df_dict.keys():  # each key is a different instrument, the values are price data frames
            split_df_dict[key] = self.data_frame_splitter(train_df_dict[key])
            train_error = self.lstm_trainer(split_df_dict[key]['train_data'])
            if float(train_error) > 1:
                print('[WARNING] Training error > 1%')
            test_error = self.lstm_tester(split_df_dict[key]['test_data'])['test_score']
            if float(test_error) > 1:
                print('[WARNING] Testing error > 1%')
        train_test_duration = time.time() - start_time
        gmt = time.gmtime(train_test_duration)
        form_gmt = time.strftime("%H hours, %M minutes, %SS seconds", gmt)
        print('[INFO] Train and test duration: ', form_gmt)



