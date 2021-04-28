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


'''
####################################################################################################################
UNISTEP VANILLA LSTM PREDICTOR
####################################################################################################################

Takes steps_x values and predicts the next immediate value
'''


class LSTMPredictor:
    def __init__(self, data_frame: pd.DataFrame, steps_x: int, steps_y: int, nn_arch: list, inst: str, epochs: int,
                 batch_size: int, forecast_type: str, dropout=0, verbose: int = 0, optimizer='adam', loss='mse',
                 activation='relu'):
        self.data_frame = data_frame  # data frame including train and test
        self.steps_x = steps_x  # nb of steps in the x array (input)
        self.steps_y = steps_y  # nb of steps in the y array (output)
        self.nn_arch = nn_arch  # list length: nb of layers, element value: depth of network (nb of neurons)
        self.dropout = dropout  # dropout forgets a % of training to prevent overfitting. Dropout: ]0, 1[
        self.epochs = epochs  # nb of training loops
        self.batch_size = batch_size
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.inst = inst
        self.activation = activation
        self.forecast_type = forecast_type

    # interface that instantiates the train and tet activities and returns the predicted results in a dict
    def launcher(self) -> dict:
        if self.steps_y != self.nn_arch[-1] and self.forecast_type != 'recursive_forecast':
            raise ValueError('[ERROR] The number of steps of the prediction output vector should coincide with the'
                             'number of neurons in the output layer.'
                             'Expected %s neurons, found %s' % (self.steps_y, self.nn_arch[-1]))
        train_test_dict = self.data_frame_splitter(self.data_frame)
        model = self.trainer(train_test_dict['train_data'])
        test_dict = self.tester((train_test_dict['test_data']))
        return test_dict

    # Long Short Term Memory deep learning network. Training method.
    def trainer(self, train_df: pd.DataFrame):
        print('[INFO] Running: %s at %s' % (self.trainer, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        training_set_scaled = self.scaler(train_df=train_df, inst=self.inst)
        x_train, y_train = self.to_supervised(training_set_scaled)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape data structure
        model = self.build_nn_arch(x=x_train, nn_arch=self.nn_arch, dropout=self.dropout)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mean_squared_error'])
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        score_train = model.evaluate(x_train, y_train, verbose=self.verbose)
        model.save(conf.MODEL_PATH % self.inst)
        print("[INFO] %s during training: %.2f%%" % (model.metrics_names[1], score_train[1]*100))
        return model

    # converts an array (train_array) of length N into 2 input and output arrays,
    # build at the same time of N/(steps_in + steps_out) arrays of length steps_in and steps_out respectively.
    # This serves to feed a Neural Net model with an input array of arrays, and train it with an output array of arrays
    # This method is used by multistep trainers as lstm_multistep_trainer
    def to_supervised(self, t_array: np.array) -> np.array:
        x, y = list(), list()
        for step in range(len(t_array) - self.steps_x - self.steps_y + 1):
            x.append(t_array[step:step + self.steps_x])
            y.append(t_array[step + self.steps_x:step + self.steps_x + self.steps_y])
        return np.array(x), np.array(y)

    def test_structure_generator(self, test_array):
        x_test = []
        for i in range(self.steps_x, len(test_array)):
            x_test.append(test_array[i-self.steps_x:i])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape data structure
        return x_test

    # based on a history array, predict the nex value
    def direct_forecast(self, model, history: np.array):
        if len(history) != self.steps_x:
            raise Exception('[ERROR] Bad input vector definition. '
                            'Expected length %s. found %s' % (self.steps_x, len(history)))
        try:
            input_x = np.reshape(history, (history.shape[0], history.shape[1], 1))
            yhat = model.predict(input_x, verbose=self.verbose)
            return yhat
        except Exception('[ERROR] Prediction error'):
            print('[ERROR] Next value could not be predicted')

    def recursive_forecast(self, model, history: np.array) -> np.array:
        if len(history) != self.steps_x:
            raise Exception('[ERROR] Bad input vector definition. '
                            'Expected length %s. found %s' % (self.steps_x, len(history)))
        try:
            yhat = list()
            for current_step in range(self.steps_y):
                input_x = np.reshape(history, (1, len(history), 1))
                new_step = model.predict(input_x, verbose=self.verbose)
                input_x = np.reshape(input_x, (len(history)))
                yhat = np.append(yhat, new_step)
                input_x = self.add_last_drop_first(input_x, new_step)
            return yhat
        except Exception('[ERROR] Prediction error'):
            print('[ERROR] Next value could not be predicted')

    # adds a value at the end of a pandas data frame and deletes the first value, so the df remains of same length
    @staticmethod
    def add_last_drop_first(old_array: np.array, new_value) -> np.array:
        new_array = np.append(old_array, new_value)
        new_array = np.delete(new_array, [0])
        return new_array

    # Long Short Term Memory deep learning network. Testing method
    # the last layer should meet nb_neurons = steps_y = 1
    def direct_unistep_tester(self, test_df: pd.DataFrame) -> dict:
        if self.steps_y != 1 or self.nn_arch[-1] != 1:
            print('[ERROR] For direct unistep models, the nb of neurons of the last layer '
                             'and the length of the predicted array = 1')
            raise ValueError('[ERROR] Expected one single neuron in the last layer, found %s'
                             'Expected length of the prediction array equal to 1, found %s'
                             % (self.nn_arch[-1], self.steps_y))

        print('[INFO] Running: %s at %s' % (self.direct_unistep_tester, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        loaded_model = self.load_model()
        sc = self.load_scaler()
        test_data = np.array(test_df)
        test_data = sc.transform(test_data.reshape(-1, 1))
        test_data = self.test_structure_generator(test_data)
        yhat = self.direct_forecast(loaded_model, test_data)
        trans_yhat = sc.inverse_transform(yhat)
        loaded_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mean_squared_error'])


        if self.forecast_type == 'direct_forecast':

        if self. forecast_type == 'recursive_forecast':  # nb_neurons = 1, steps_y >= 1
            yhat = self.recursive_forecast(loaded_model, trans_test_array)


        score_test = loaded_model.evaluate(trans_test_array, trans_yhat, verbose=self.verbose)
        print("[INFO] %s during testing: %.2f%%" % (loaded_model.metrics_names[1], score_test[1]*100))
        return {'test_score': score_test[1]*100,
                'input_test_data': test_array,
                'predicted_data': trans_yhat}

    def recursive_tester(self, test_df: pd.DataFrame) -> dict:

    def build_nn_arch(self, x: np.array, nn_arch: list, dropout: float):
        input_return = False
        if len(nn_arch) > 2:
            input_return = True
        regressor = Sequential()
        print('regressor = Sequential()')  # initializeNN and first layer:
        regressor.add(LSTM(units=nn_arch[0],
                           activation=self.activation, return_sequences=input_return, input_shape=(x.shape[1], 1)))
        print('regressor.add(LSTM(units=%s, activation=\'relu\', return_sequences=%s, input_shape=(x.shape[1], 1)))' %
              (nn_arch[0], input_return))
        if dropout > 0:
            print('regressor.add(Dropout(%s))' % dropout)
            regressor.add(Dropout(dropout))
        for layer in range(len(nn_arch) - 2):
            if layer == len(nn_arch) - 3:
                regressor.add(LSTM(units=nn_arch[layer + 1]), return_sequences=False)
                print('regressor.add(LSTM(units=%s, return_sequences=False))' % nn_arch[layer + 1])
            else:
                regressor.add(LSTM(units=nn_arch[layer + 1]), return_sequences=True)
                print('regressor.add(LSTM(units=%s, return_sequences=True))' % nn_arch[layer + 1])
            if dropout > 0:
                regressor.add(Dropout(dropout))
                print('regressor.add(Dropout(%s))' % dropout)
        regressor.add(Dense(units=nn_arch[-1]))
        print('regressor.add(Dense(units=%s))' % nn_arch[-1])
        return regressor

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

    def descaler(self):
        pass

    # splits the original dataframe into training and testing one
    # assuming that the lowest volume of data is Friday with 11500 values ==> at least 1500 values for testing
    @staticmethod
    def data_frame_splitter(data_frame: pd.DataFrame) -> dict:
        train_data = data_frame.iloc[:-1500]
        test_data = data_frame.iloc[-1500:]
        return {'train_data': train_data,
                'test_data': test_data}  # two data frames with train and test price data

    def load_scaler(self):
        try:
            loaded_scaler = load(conf.SCALER_PATH % self.inst)
            return loaded_scaler
        except ScalerLoadingError:
            print('[ERROR] Scaler .joblib could not be loaded')

    def load_model(self) -> load_model:
        try:
            loaded_model = load_model(conf.MODEL_PATH % self.inst)
            loaded_model.summary()
            return loaded_model
        except ModelLoadingError:
            print('[ERROR] Model .h5 could not be loaded')

'''
####################################################################################################################
RECURRENT UNISTEP VANILLA LSTM PREDICTOR
####################################################################################################################

Takes the steps_x values and predicts the steps_y values, 
where steps_y values is an array made of last unistep predictions
'''



class DataPrediction:
    def __init__(self, insts: list, meas: str, comp: str, field: str):
        self.ne = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['Training']['epochs']
        self.ma_type = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['Training']['ma_type']
        self.insts = insts
        self.meas = meas
        self.comp = comp
        self.field = field
        self.aux_methods = AuxPredictorMethods()


    # NOTE ON KERAS TENSORFLOW #
    # Input shape --> [samples, time steps, features] where:
    # samples: number of input samples in the training data set
    # time steps: number of price values of each sample
    # features: number of currencies

    '''
    ####################################################################################################################
    MULTISTEP VANILLA LSTM PREDICTOR
    ####################################################################################################################
    '''
    # takes last training 60 values and predicts the next 60 values
    def lstm_multistep_trainer(self, train_df: pd.DataFrame, inst='EUR_USD', steps_in=60, steps_out=60) -> dict:
        print('[INFO] Running: %s at %s' % (self.lstm_multistep_trainer, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        training_set_scaled = self.aux_methods.scaler(train_df=train_df, inst=inst)
        training_set = np.array(training_set_scaled)
        #training_set = training_set.reshape(-1, 1)
        # convert the training array into a set of multi step data
        x_train, y_train = self.to_supervised(
            t_array=training_set,
            steps_in=steps_in,
            steps_out=steps_out)
        # x_train shape : (n_samples, n_steps, n_features)
        # y_train shape : (n_samples, n_steps)
        n_samples, n_steps, n_features = x_train.shape[0], x_train.shape[1], x_train.shape[2]
        verbose, epochs, batch_size = 0, 50, 1

        regressor = Sequential()  # Initialize RNN
        regressor.add(LSTM(units=60, activation='relu', input_shape=(n_steps, n_features)))
        #regressor.add(Dropout(0.2))
        #regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
        #regressor.add(Dropout(0.2))
        #regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
        #regressor.add(Dropout(0.2))
        #regressor.add(LSTM(units=60, activation='relu'))  # return sequences = False (because it is the last layer)
        #regressor.add(Dropout(0.2))
        #regressor.add(Dense(units=100, activation='relu'))
        regressor.add(Dense(units=n_steps))
        regressor.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
        regressor.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        score_train = regressor.evaluate(x_train, y_train, verbose=verbose)
        regressor.save(conf.MODEL_PATH % inst)
        print("[INFO] Model correctly saved to disk")
        print("[INFO] %s during training: %.2f%%" % (regressor.metrics_names[1], score_train[1] * 100))

        return score_train[1] * 100

    def lstm_multistep_tester(self, test_df: pd. DataFrame, inst='EUR_USD') -> dict:
        print('[INFO] Running: %s at %s' % (self.lstm_multistep_tester, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        loaded_model = load_model(conf.MODEL_PATH % inst)
        loaded_model.summary()

        test_array = np.array(test_df)
        test_array = test_array[180:240]
        test_array = test_array.reshape(-1, 1)

        sc = load(conf.SCALER_PATH % inst)
        x_test = sc.transform(test_array)

        x_test = np.reshape(x_test, (1, len(x_test), 1))

        predicted_data_set = loaded_model.predict(x_test, verbose=0)
        predicted_data_set = sc.inverse_transform(predicted_data_set)

        loaded_model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
        score_test = loaded_model.evaluate(x_test, predicted_data_set, verbose=0)
        print("[INFO] %s during testing: %.2f%%" % (loaded_model.metrics_names[1], score_test[1] * 100))

        return {'test_score': score_test[1] * 100,
                'predicted_data': predicted_data_set}



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

class AuxPredictorMethods:
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



    def deseasonalize(self):
        pass




