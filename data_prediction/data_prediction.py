import time
import os
from datetime import datetime
from txus_library import txuslib
from data_loading import data_loading as dload
import config_data.constants as conf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from tensorflow.keras.models import Sequential  # pip install tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)


class DataLoadingError(Exception):
    pass


class ModelLoadingError(Exception):
    pass


class ScalerLoadingError(Exception):
    pass


class DataPrediction:
    def __init__(self, insts: list, meas: str, comp: str, field: str):
        self.ne = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['Training']['epochs']
        self.ma_type = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['Training']['ma_type']
        self.insts = insts
        self.meas = meas
        self.comp = comp
        self.field = field

    # NOTE ON KERAS TENSORFLOW #
    # Input shape --> [samples, time steps, features] where:
    # samples: number of input samples in the training data set
    # time steps: number of price values of each sample
    # features: number of currencies

    '''
    ####################################################################################################################
    UNISTEP VANILLA LSTM PREDICTOR
    ####################################################################################################################
    '''

    # TODO: decorator with different filters (least squares, fourier, etc.)
    # Long Short Term Memory deep learning network. Training method.
    def lstm_trainer(self, train_df: pd.DataFrame, inst: str):
        print('[INFO] Running: %s at %s' % (self.lstm_trainer, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        training_set = np.array(train_df)
        training_set = training_set.reshape(-1, 1)
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)
        try:
            os.makedirs(conf.MODEL_DIR)
        except OSError as e:
            print("[ERROR] lstm_trainer. Creation of the directory %s failed : %s " % (conf.MODEL_DIR, e.strerror))
        dump(sc, conf.SCALER_PATH % inst)

        x_train = []
        y_train = []
        print('len(training_set_scaled): ', len(training_set_scaled))
        for i in range(60, len(training_set_scaled)):
            x_train.append(training_set_scaled[i-60:i])  # i not included in range => i-60:i => from 0 to 59 for i=0
            y_train.append(training_set_scaled[i, 0])  # prediction of value 60

        x_train, y_train = np.array(x_train), np.array(y_train)
        print('len(x_train): ',len(x_train))
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape data structure

        print('x_train.shape:', x_train.shape)
        print('x_train.shape[0]:', x_train.shape[0])
        print('x_train.shape[1]:', x_train.shape[1])
        """regressor = Sequential()  # Initialize RNN
        
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

        regressor.save(conf.MODEL_PATH % inst)
        print("[INFO] Model correctly saved to disk")
        print("[INFO] %s during training: %.2f%%" % (regressor.metrics_names[1], score_train[1]*100))

        return score_train[1]*100"""

    # TODO
    # Long Short Term Memory deep learning network. Testing method
    def lstm_tester(self, test_df: pd. DataFrame, inst: str) -> dict:
        print('[INFO] Running: %s at %s' % (self.lstm_tester, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        loaded_model = load_model(conf.MODEL_PATH % inst)
        loaded_model.summary()

        inputs = np.array(test_df)
        inputs = inputs.reshape(-1, 1)
        sc = load(conf.SCALER_PATH % inst)

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

    '''
    ####################################################################################################################
    MULTISTEP VANILLA LSTM PREDICTOR
    ####################################################################################################################
    '''
    # takes last training 60 values and predicts the next 60 values
    def lstm_multistep_trainer(self, train_df: pd.DataFrame, inst: str):
        print('[INFO] Running: %s at %s' % (self.lstm_multistep_trainer, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        training_set_scaled = self.scaler(train_df=train_df, inst=inst)

        # convert the training array into a set of multi step data
        x_train, y_train = self.restructure_multistep_data(t_array=training_set_scaled, steps_in=60, steps_out=60)

        # reshape input to be [samples, time steps, features]. For lstm_multistep_trainer features = 1
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        verbose, epochs, batch_size = 0, 50, 32

        regressor = Sequential()  # Initialize RNN

        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
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

        regressor.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

        score_train = regressor.evaluate(x_train, y_train, verbose=verbose)

        regressor.save(conf.MODEL_PATH % inst)
        print("[INFO] Model correctly saved to disk")
        print("[INFO] %s during training: %.2f%%" % (regressor.metrics_names[1], score_train[1] * 100))

        return score_train[1] * 100

    def lstm_multistep_tester(self, test_df: pd. DataFrame, inst: str) -> dict:
        print('[INFO] Running: %s at %s' % (self.lstm_multistep_tester, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        loaded_model = load_model(conf.MODEL_PATH % inst)
        loaded_model.summary()

        inputs = np.array(test_df)
        inputs = inputs.reshape(-1, 1)
        sc = load(conf.SCALER_PATH % inst)

        inputs = sc.transform(inputs)
        x_test, not_used = self.restructure_multistep_data(t_array=inputs, steps_in=60, steps_out=60)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_data_set = loaded_model.predict(x_test)
        predicted_data_set = sc.inverse_transform(predicted_data_set)

        loaded_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        score_test = loaded_model.evaluate(x_test, predicted_data_set, verbose=0)
        print("[INFO] %s during testing: %.2f%%" % (loaded_model.metrics_names[1], score_test[1] * 100))

        # plt.plot(test_df, color='blue', label='raw data')
        # plt.plot(predicted_data_set, color='red', label='predicted price data stream')
        # plt.show()
        return {'test_score': score_test[1] * 100,
                'predicted_data': predicted_data_set}

    # converts an array (train_array) of length N into 2 input and output arrays,
    # build at the same time of N/(steps_in + steps_out) arrays of length steps_in and steps_out respectively.
    # This serves to feed a Neural Net model with an input array of arrays, and train it with an output array of arrays
    # This method is used by multistep trainers as lstm_multistep_trainer
    def restructure_multistep_data(self, t_array: np.array, steps_in: int = 60, steps_out: int = 60) -> np.array:
        x, y = list(), list()
        for start_x in range(len(t_array)):
            end_x = steps_in + start_x
            start_y = steps_in + start_x
            end_y = steps_in + steps_out + start_x
            if end_y <= len(t_array):
                x_input = t_array[start_x:end_x]
                y_input = t_array[start_y:end_y]
                x.append(x_input)
                y.append(y_input)
                start_x += 1
            else:
                break
        return np.array(x), np.array(y)

    '''
    ####################################################################################################################
    MULTISTEP ENCODER-DECODER LSTM PREDICTOR
    ####################################################################################################################
    '''
    # TODO
    def enc_dec_lstm_multistep_trainer(self):
        pass

    '''
    ####################################################################################################################
    MULTISTEP CONVOLUTIONAL NEURAL NETWORK LSTM PREDICTOR
    ####################################################################################################################
    '''

    # TODO
    def cnn_lstm_multistep_trainer(self):
        pass

    '''
    ####################################################################################################################
    Multistep conv LSTM predictor
    ####################################################################################################################
    '''

    # TODO
    def conv_lstm_multistep_trainer(self):
        pass


    '''
    ####################################################################################################################
    Common methods
    ####################################################################################################################
    '''

    # uses the data from previous day to train and test the prediction algorithm
    def daily_train_test(self, train_df_dict=None):
        print('[INFO] Running: %s at %s' % (self.daily_train_test, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        if train_df_dict is None:
            train_df_dict = self.get_training_data()
        split_df_dict = {}
        start_time = time.time()
        txuslib.delete_dir(conf.MODEL_DIR)  # delete data from previous day
        txuslib.create_dir(conf.MODEL_DIR)  # create a new directory
        for inst in train_df_dict.keys():  # each key is a different instrument, the values are price data frames
            split_df_dict[inst] = self.data_frame_splitter(train_df_dict[inst].loc[:, ['_value']])
            train_error = self.lstm_trainer(train_df=split_df_dict[inst]['train_data'], inst=inst)
            if float(train_error) > 1:
                print('[WARNING] Training error in instrument %s > 1%%' % inst)
            test_error = self.lstm_tester(test_df=split_df_dict[inst]['test_data'], inst=inst)['test_score']
            if float(test_error) > 1:
                print('[WARNING] Testing error in instrument %s > 1%%' % inst)
        train_test_duration = time.time() - start_time
        gmt = time.gmtime(train_test_duration)
        form_gmt = time.strftime("%H hours, %M minutes, %S seconds", gmt)
        print('[INFO] Train and test duration: ', form_gmt)

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

    # recovers the price data from the previous day to feed the training
    def get_training_data(self) -> dict:
        print('[INFO] Running: %s at %s' % (self.get_training_data, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
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

    def scaler(self, train_df: pd.DataFrame, inst: str) -> np.array:
        training_set = np.array(train_df)
        training_set = training_set.reshape(-1, 1)  # convert a column vector to nx1 matrix
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)
        try:
            os.makedirs(conf.MODEL_DIR)
        except OSError as e:
            print("[ERROR] Creation of the directory %s failed : %s " % (conf.MODEL_DIR, e.strerror))
        dump(sc, conf.SCALER_PATH % inst)
        return np.array(training_set_scaled)
