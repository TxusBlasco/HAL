import csv
import shutil

import pandas as pd
import os
import json
import math
import datetime
import config_data.constants as conf
import numpy as np
from sklearn.metrics import mean_squared_error
import yaml
import matplotlib.pyplot as plt

'''
#######################################################################################################################
auxiliary methods for file operations
#######################################################################################################################
'''


class Pandas:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    # read a csv file with comma separated values
    def read_csv(self):
        df = pd.read_csv(self.csv_file, sep=',', header=None)
        return df


# create_dir creates a folder in the directory path
# path should include folder name
def create_dir(path: str):
    try:
        os.makedirs(path)
    except FileExistsError:
        print('[INFO] File {} already exists'.format(path))


# TODO unit test
def delete_dir(path: str):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("[ERROR] Deletion of directory %s failed: %s" % (path, e.strerror))


# takes a json file with the following format:
#     json = {
#         "candles": [X],
#         "granularity": gran,
#         "instrument": inst
#     }
# where X shows the candles info
def write_json_candles_to_csv(json_file: json, file_path: str):
    ordered_data = {
        'volume': [json_file['candles'][x]['volume'] for x in range(len(json_file['candles']))],
        'time': [json_file['candles'][x]['time'] for x in range(len(json_file['candles']))],
        'o': [json_file['candles'][x]['mid']['o'] for x in range(len(json_file['candles']))],
        'h': [json_file['candles'][x]['mid']['h'] for x in range(len(json_file['candles']))],
        'l': [json_file['candles'][x]['mid']['l'] for x in range(len(json_file['candles']))],
        'c': [json_file['candles'][x]['mid']['c'] for x in range(len(json_file['candles']))],
    }
    df = pd.DataFrame(data=ordered_data)
    try:
        df.to_csv(file_path)
    except:
        print('[ERROR] Could not write to CSV file')


def get_data_from_yaml(yaml_path):
    try:
        with open(yaml_path, 'r') as opened_yaml:
            parsed_yaml = yaml.load(opened_yaml, Loader=yaml.FullLoader)
        return parsed_yaml
    except yaml.YAMLError as e:
        print('[ERROR] No data from yaml file could be recovered', e)


def append_data_to_file(file_path, data):
    try:
        f = open(file_path, "a")
        f.write(data)
        f.close()
    except:
        print('[ERROR] No data could be appended to file {}'.format(file_path))


def overwrite_data_in_file(file_path, data):
    try:
        f = open(file_path, "w")
        f.write(data)
        f.close()
    except:
        print('[ERROR] No data could be overwritten in file {}'.format(file_path))


''' 
#######################################################################################################################
auxiliary methods for operations over the Oanda API
#######################################################################################################################
'''


# retrieves the timestamp from the last candle with "complete" status equal to true
def get_timestamp_from_last_candle(js):
    if str(js['candles'][-1]['complete']) == 'True':
        last_true_time = js['candles'][-1]['time']
        return last_true_time
    elif str(js['candles'][-2]['complete']) == 'True':
        last_true_time = js['candles'][-2]['time']
        return last_true_time
    else:
        print('[ERROR] Wrong value executing method get_timestamp_from_last_candle')


# receives a json file from a get candles API request
# retrieves the price value from the json corresponding to the close price for the last candle that is complete
# returns that price as a float
def get_open_price_from_candle(js):
    if js["candles"][-1]['complete'] == 'true':
        open_price = float(js["candles"][-1]["mid"]["o"])
        return open_price
    elif js["candles"][-2]['complete'] == 'true':
        open_price = float(js["candles"][-2]["mid"]["o"])
        return open_price
    else:
        print('[ERROR] Wrong value executing method get_open_price_from_candle')

def get_higher_price_from_candle(js):
    if js["candles"][-1]['complete'] == 'true':
        higher_price = float(js["candles"][-1]["mid"]["h"])
        return higher_price
    elif js["candles"][-2]['complete'] == 'true':
        higher_price = float(js["candles"][-2]["mid"]["h"])
        return higher_price
    else:
        print('[ERROR] Wrong value executing method get_higher_price_from_candle')


def get_lower_price_from_candle(js):
    if js["candles"][-1]['complete'] == 'true':
        lower_price = float(js["candles"][-1]["mid"]["l"])
        return lower_price
    elif js["candles"][-2]['complete'] == 'true':
        lower_price = float(js["candles"][-2]["mid"]["l"])
        return lower_price
    else:
        print('[ERROR] Wrong value executing method get_lower_price_from_candle')


def get_close_price_from_candle(js):
    if js['candles'][-1]['complete'] == 'true':
        close_price = float(js['candles'][-1]['mid']['c'])
        return close_price
    elif js['candles'][-2]['complete'] == 'true':
        close_price = float(js['candles'][-2]['mid']['c'])
        return close_price
    else:
        print('[ERROR] Wrong value executing method get_close_price_from_candle')


# receives a json file from an order request
# retrieves the price at which that order has been made
# returns that price as a float
def get_price_from_order(js):
    price = float(js["orderFillTransaction"]["price"])
    return price


# TODO Unit test pending
# receives a json file from an order request
# retrieves the trade id for the order that has been made
# returns that trade id as a string
def get_trade_id_from_order(js):
    trade_id = str(js["orderFillTransaction"]["tradeOpened"]["tradeID"])
    return trade_id


# gets two dates in OANDA format and returns the difference in granularity_to_seconds
# oanda json format: 2016-01-01T00:00:00.000000000Z
def get_nb_seconds(start_date, end_date):
    start_date_dict = oanda_date_format_break_down(start_date)
    end_date_dict = oanda_date_format_break_down(end_date)
    start = datetime.datetime(start_date_dict['year'],
                              start_date_dict['month'],
                              start_date_dict['day'],
                              start_date_dict['hour'],
                              start_date_dict['minute'],
                              start_date_dict['second']
                              )
    end = datetime.datetime(end_date_dict['year'],
                            end_date_dict['month'],
                            end_date_dict['day'],
                            end_date_dict['hour'],
                            end_date_dict['minute'],
                            end_date_dict['second']
                            )
    diff = end - start
    sec = diff.total_seconds()
    return sec


# gets a date and returns the number of_seconds elapsed from 00:00 of the 1st of January 1900 till  that date
# format: 2016-01-01T00:00:00.000000000Z
def get_nb_seconds_1900(date):
    sec = get_nb_seconds('1900-01-01T00:00:00:000000000Z', date)
    return sec


# Gets date in OANDA format and provides a dictionary with its values
# OANDA format: 2016-01-01T00:00:00.000000000Z
def oanda_date_format_break_down(date):
    date_dict = {
        'year': int(date[0:4]),
        'month': int(date[5:7]),
        'day': int(date[8:10]),
        'hour': int(date[11:13]),
        'minute': int(date[14:16]),
        'second': int(date[17:19]),
        'microsecond': int(date[20:-1])
    }
    return date_dict


# converts UNIX date format to the format used in Oanda's Json file responses
# Unix format: '2016-01-01 00:00:00.000000000'
# OANDA Format: '2016-01-01T00:00:00.000000000Z'
def conv_unix_to_oanda_json(date):
    frmtd_date = date.replace(" ", "T")
    frmtd_date += "Z"
    return frmtd_date


# converts a date in unix format  to url encoded format
# unix: '2016-01-01 00:00:00.000000000'
# url encoded: '2016-01-01T00%3A00%3A00.000000000Z'
def conv_unix_to_x_www_form_url_encoded(date):
    frmtd_date = date.replace(" ", "T")
    frmtd_date = frmtd_date.replace(":", "%3A")
    frmtd_date += "Z"
    return frmtd_date


# converts a date in oanda json format  to url encoded format
# oanda json: '2016-01-01T00:00:00.000000000Z'
# url encoded: '2016-01-01T00%3A00%3A00.000000000Z'
def conv_oanda_json_to_x_www_form_url_encoded(date):
    frmtd_date = date.replace(":", "%3A")
    return frmtd_date


# gets the number of candles between two dates
# weekends are discounted from the sum
# date in oanda json format: 2016-01-01T00:00:00.000000000Z
# NOTE: the number of candles provided is estimated, because there are gaps in the market that can not be accounted
# The real number of candles is always equal or less to the estimate
def get_nb_candles_two_dates(start_date, end_date, granularity):
    nb_sec = get_nb_seconds(start_date, end_date)
    gran_sec = conf.granularity_to_seconds[granularity]
    week_seconds = 604800
    weekend_seconds = 172800
    nb_weekends = round_down(nb_sec/week_seconds)
    nb_trade_sec = nb_sec - nb_weekends * weekend_seconds
    nb_candles = round_down(nb_trade_sec/gran_sec)
    return nb_candles


# takes 2 json files from candle request to OANDA API (restricted to max. 5000 candles due to Oanda)
# and joins them in a single json file
# the resulting json file has the same format as Oanda json http responses
def join_js_candles(orgnl_json, new_json):
    print('[INFO] executing method: join_js_candles(orgnl_json, new_json)')
    orgnl_json['candles'].extend(new_json['candles'])
    return orgnl_json


# compares two dates in OANDA Api format,
# asserts True if the first one is an earlier date or False if the first one is a later date
def is_earlier_date(earlier_date, later_date):
    if get_nb_seconds_1900(earlier_date) < get_nb_seconds_1900(later_date):
        return True
    else:
        return False


''' 
#######################################################################################################################
auxiliary methods for math operations
#######################################################################################################################
'''


# rounds up a float number to the number of decimals wanted
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


# rounds down a float number to the number of decimals wanted
def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


# Calculates the mean of the slopes in a series data array
def slope_sum(data_array: np.ndarray):
    return sum([(data_array[x + 2] - data_array[x]) / 2 for x in range(len(data_array) - 2)]) / (len(data_array) - 2)


def weighted_slope_sum(data_array: np.ndarray):
    return sum([((data_array[x + 2] - data_array[x]) / 2) *
                (x + 1) for x in reversed(range(len(data_array) - 2))]) \
           / (sum(range(len(data_array) - 2)))


# gets an array with a window of prices and returns the mean
def mean(data_array):
    return sum(data_array)/len(data_array)


# moving_average receives a pandas data frame and applies a sma, wma or ema moving average filter
# win_length: window for which we apply the operation. Example: 10-day SMA has win_length = 10
# del_nan: if set to 'True', deletes all NaN values without replacing them and resets the indexes
# ma_type:
#   sma = single moving average
#   wma = linearly weighted moving average
#   ema = exponentially weighted moving average
def moving_average(data_array: pd.DataFrame, win_length=1, del_nan=False, ma_type='sma'):
    if ma_type == 'sma':
        sma = data_array.rolling(win_length).mean()
        if del_nan:
            sma = sma.dropna()
            sma = sma.reset_index()
            sma = sma.drop('index', axis=1)
        return sma
    elif ma_type == 'wma':
        weights = np.arange(1, win_length + 1)
        wma = data_array.rolling(win_length).apply(lambda prices: np.dot(prices, weights/weights.sum()))
        if del_nan:
            wma = wma.dropna()
            wma = wma.reset_index()
            wma = wma.drop('index', axis=1)
        return wma
    elif ma_type == 'ema':
        ema = data_array.ewm(span=win_length, min_periods=win_length, adjust=True).mean()
        if del_nan:
            ema = ema.dropna()
            ema = ema.reset_index()
            ema = ema.drop('index', axis=1)
        return ema
    else:
        print('[ERROR] ma_type variable should be one of the element in the list (sma, wma, ema)')
        return None


# Linearly weighted moving average
# gets an array with prices and returns an array with the wma
def linear_wma(data_array: pd.DataFrame, shift=0, win_len=1):
    weights = np.arange(1, win_len + 1)
    wma = data_array.rolling(win_len).apply(lambda prices: np.dot(prices, weights/weights.sum()))
    wma = wma.shift(periods=shift)
    return wma

# Linearly weighted moving average
# gets an array with prices and returns an array with the wma
def inv_linear_wma(data_array: pd.DataFrame, shift, window_length=1):
    weights = np.arange(1, window_length + 1)
    wma = data_array.rolling(window_length).apply(lambda prices: np.dot(prices, weights/weights.sum()))
    wma = wma.shift(periods=shift)

    return wma


def mean_sq_error(real_stock_price, predicted_stock_price):
    rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    return rmse


# gets the last 3 values of a time series and throws a string 'MAX' if the series has reached a maximum
# and 'MIN if the series has reached a minimum
def is_max_min(vals):
    if vals[0] < vals[1] and vals[2] < vals[1]:
        return 'MAX'
    elif vals[0] > vals[1] and vals[2] > vals[1]:
        return 'MIN'
    else:
        return None


''' 
#######################################################################################################################
auxiliary methods for plot operations
#######################################################################################################################
'''

# gets price and time and plots them in an dynamically animated graphic
def plot_real_time_price_lstm(time_array, price_array, prediction):
    plt.plot(time_array, price_array)
    plt.plot(time_array, prediction)
    #plt.ylim([1.083,1.085])
    plt.draw()
    plt.pause(0.0001)
    plt.clf()


''' 
#######################################################################################################################
auxiliary methods for data operations
#######################################################################################################################
'''


def df_from_inst_dict(inst_dict:dict, field:str) -> pd.DataFrame:
    aux_dict = {}
    for inst in inst_dict.keys():
        aux_dict[inst] = inst_dict[inst][field]
    df = pd.DataFrame(aux_dict, index=[0])
    return df
