from data_prediction import data_prediction as dpred
from data_extraction import data_extraction as dext
from txus_library import txuslib
import pandas as pd
import config_data.constants as conf
import matplotlib.pyplot as plt
import numpy as np


class AITester:
    def data_volume_error(self):
        raise ValueError('[ERROR] Not enough data for training and testing. Choose a broader data range')

    # instanciates the trainer
    def vanilla_multistep_lstm_trainer(self, train_data, steps_in, steps_out):
        print('[INFO] running', self.vanilla_multistep_lstm_trainer)
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close')
        trainer_score = dp.lstm_multistep_trainer(train_df=train_data, inst='EUR_USD', steps_in=steps_in, steps_out=steps_out)

    # instanciates the tester
    def vanilla_multistep_lstm_tester(self, test_data):
        print('[INFO] running', self.vanilla_multistep_lstm_tester)
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close')
        tester_data = dp.lstm_multistep_tester(test_df=test_data, inst='EUR_USD')
        return tester_data

    # plots real vs predicted data. Be sure to provide data starting from the same timestamp
    def plot(self, real_data, predicted_data):
        plt.plot(predicted_data, color='red', label='predicted data')
        plt.plot(real_data, color='blue', label='raw data')
        plt.show()

    # adds nb_zeros number of zeroes at the beginning of the original_array
    def add_nan(self, original_array, nb_nan):
        nan_array = np.empty(nb_nan)
        nan_array[:] = np.nan
        original_nan = np.append(nan_array, original_array)
        return original_nan

    # gets the data coming from two csv files, testing and train
    def get_train_test_from_csv(self):
        train_data = pd.read_csv(conf.TESTING_TRAIN_CSV)
        test_data = pd.read_csv(conf.TESTING_TEST_CSV)
        return train_data['c'], test_data['c']

    # joins data coming from two train and test csv for testing purposes
    def join_train_test_data(self):
        train_data, test_data = self.get_train_test_from_csv()
        overall_data = np.append(train_data, test_data)
        return overall_data

    # extracts massive data from database and saves it in a CSV file
    def get_bulk_price_data_to_csv(self, start_date: str, end_date: str):
        de = dext.DataExtraction(insts=['EUR_USD'], gran='S5', comp='M')
        bulk_json = de.get_bulk_price_data_set(
            inst='EUR_USD',
            start_date=start_date,
            end_date=end_date)
        txuslib.write_json_candles_to_csv(json_file=bulk_json, file_path=conf.TESTING_BULK_CSV)


    # splits a json made of candles into train and test json candles
    def split_train_test_data(self, json_file):
        if len(json_file['candles']) < 2000:  # choose not less than 1500 values for training and 500 for testing
            raise self.data_volume_error()
        test_json, train_json = {}, {}
        test_json['candles'] = json_file['candles'][-500:]
        test_json['granularity'] = json_file['granularity']
        test_json['instrument'] = json_file['instrument']
        train_json['candles'] = json_file['candles'][:-500]
        train_json['granularity'] = json_file['granularity']
        train_json['instrument'] = json_file['instrument']
        return test_json, train_json

    def smoothen_data(self, win_length: int):
        train_df, test_df = self.get_train_test_from_csv()
        train_ema = txuslib.moving_average(train_df, win_length=win_length, del_nan=True, ma_type='ema')
        test_ema = txuslib.moving_average(test_df, win_length=win_length, del_nan=True, ma_type='ema')
        return train_ema, test_ema

    def vanilla_unistep(self, data_frame):
        lstm = dpred.LSTMPredictor(data_frame=data_frame, steps_x=60, steps_y=60, nn_arch=[50, 1], inst='EUR_USD',
                                   batch_size=16, epochs=50)
        test_dict = lstm.launcher()
        return test_dict


def main():
    steps_in = 60
    steps_out = 60
    win_length = 60
    start_date = '2021-04-06T00:00:00.000000000Z'
    end_date = '2021-04-08T00:00:00.000000000Z'
    get_new_data = False

    ait = AITester()
    if get_new_data:  # refreshes train and test data (deletes current exisitng train and test csv)
        ait.get_bulk_price_data_to_csv(start_date=start_date, end_date=end_date)
    '''train_data, test_data = ait.smoothen_data(win_length=win_length)
    ait.vanilla_multistep_lstm_trainer(train_data=train_data, steps_in=steps_in, steps_out=steps_out)
    tester_data = ait.vanilla_multistep_lstm_tester(test_data=train_data)  # dict result of test
    np.savetxt(conf.VANILLA_MULTISTEP_OUTPUT, tester_data['predicted_data'][0], delimiter=",")
    plt.plot(pd.read_csv(conf.TESTING_TEST_CSV)['c'], color='blue')
    plt.plot(ait.add_nan(test_data, win_length), color='orange')
    plt.plot(ait.add_nan(pd.read_csv(conf.VANILLA_MULTISTEP_OUTPUT), steps_in), color='red')
    plt.show()'''
    data_frame = pd.read_csv(conf.TESTING_BULK_CSV)['c']
    lstm = dpred.LSTMPredictor(data_frame=data_frame, forecast_type='recursive_forecast', steps_x=60, steps_y=6,
                               nn_arch=[50, 1], inst='EUR_USD', batch_size=16, epochs=50)
    test_dict = lstm.launcher()
    print(test_dict)
    plt.plot(data_frame[-1500:], color='blue')
    plt.plot(test_dict['input_test_data'][0], color='orange')
    plt.plot(ait.add_nan(test_dict['predicted_data'], 60), color='red')
    plt.show()





if __name__ == '__main__':
    main()

