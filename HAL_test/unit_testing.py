import unittest
from data_transformation import data_transformation as dtrans
from data_prediction import data_prediction as dpred
from data_extraction import data_extraction as dext
from data_loading import data_loading as dload
from txus_library import txuslib as txuslib
from task_manager import task_manager as tman
import config_data.constants as conf
import pandas as pd
import numpy as np
import os
from datetime import date
import shutil


class TestDataExtraction(unittest.TestCase):
    def test_inst_dict_constructor(self):
        insts=['EUR_USD', 'EUR_JPY', 'EUR_AUD']
        keys = ['status_code', 'timestamp', 'open_price', 'higher_price', 'lower_price', 'close_price']
        de = dext.DataExtraction(insts=insts, comp='M', gran='S5')
        insts_dict = de.inst_dict_constructor()
        for inst in insts:
            self.assertIn(inst, insts_dict.keys())
            for key in keys:
                self.assertIn(key, insts_dict[inst].keys())

class TestDataPrediction(unittest.TestCase):

    def test_lstm_trainer(self):
        # the test creates h5 and joblib files in the \test directory
        _dataset_train = pd.read_csv(conf.TESTING_TRAIN_CSV)
        _df_filt = _dataset_train[['c']]
        _dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close_price')
        score = _dp.lstm_trainer(_df_filt)
        self.assertTrue(score < 0.3, msg='[TEST FAIL] Calculation error is higher than 0.3')
        self.assertTrue(os.path.exists(conf.SCALER_PATH), msg='[TEST FAIL] %s file does not exist' % conf.SCALER_PATH)
        self.assertTrue(os.path.exists(conf.MODEL_PATH), msg='[TEST FAIL] %s file does not exist' % conf.MODEL_PATH)
        try:
            shutil.rmtree(conf.MODEL_DIR)  # removes the folder after the test
        except OSError as e:
            print("[ERROR] Deletion of the directory %s failed : %s" % (conf.MODEL_DIR, e.strerror))

    def test_lstm_tester(self):
        train_path = r'models_data\%s_training' % date.today().strftime('%y%m%d')
        try:
            os.makedirs(conf.MODEL_DIR)
        except OSError as e:
            print("[ERROR] Creation of the directory %s failed : %s " % (conf.MODEL_DIR, e.strerror))

        shutil.copyfile(conf.TESTING_MODEL_PATH, conf.MODEL_PATH)

        shutil.copyfile(conf.TESTING_SCALER_PATH, conf.SCALER_PATH)

        _dataset_test = pd.read_csv(conf.TESTING_TEST_CSV)
        _df_filt = _dataset_test[['c']]
        _dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close_price')
        ret = _dp.lstm_tester(_df_filt)
        self.assertTrue(ret[0] < 1, msg='[TEST FAIL] Calculation error is higher than 1%')
        self.assertTrue(type(ret[1]) == list)
        try:
            shutil.rmtree(conf.MODEL_DIR)  # removes the folder after the test
        except OSError as e:
            print("[ERROR] Deletion of the directory %s failed : %s" % (conf.MODEL_DIR, e.strerror))

    def test_dynamic_predictor(self):
        df = pd.read_csv(conf.TESTING_TRAIN_CSV, sep=',')
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close_price')
        pred_price = dp.dynamic_predictor(df['c'], conf.MODEL_PATH, conf.SCALER_PATH)
        self.assertEqual(str(pred_price), '1.1882545')

    def test_lstm_load_model(self):
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close_price')
        loaded_model = dp.lstm_load_model(conf.TESTING_MODEL_PATH)
        print(type(loaded_model))
        self.assertEqual(str(type(loaded_model)), '<class \'keras.engine.sequential.Sequential\'>')

    def test_lstm_load_scaler(self):
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close_price')
        loaded_scaler = dp.lstm_load_scaler(conf.TESTING_SCALER_PATH)
        print(type(loaded_scaler))
        self.assertEqual(str(type(loaded_scaler)), '<class \'sklearn.preprocessing._data.MinMaxScaler\'>')

    def test_dataframe_splitter(self):
        df = pd.DataFrame(np.random.random(15000), columns=['random'])
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close_price')
        dp_train = dp.data_frame_splitter(df)['train_data']
        dp_test = dp.data_frame_splitter(df)['test_data']
        self.assertEqual((dp_train.shape, dp_test.shape), ((13500, 1), (1500, 1)))


class TestDataTransformation(unittest.TestCase):
    # this test case is testing the business logic part of the method, not the API connection (tested in integration t.)
    def test_set_price_data_db(self):
        dt = dtrans.DataTransformation(price_dict={'timestamp': 'Hello, world',
                                                   'open_price': 1983,
                                                   'higher_price': 1983,
                                                   'lower_price': 1983,
                                                   'close_price': 1983},
                                       insts=['EUR_USD', 'EUR_CHF', 'EUR_JPY'],
                                       meas='raw_price',
                                       comp='M',
                                       bucket='test_env_bucket')
        actual_query = dt.set_price_data_db()
        expected_query = ['raw_price,inst=EUR_USD,price_comp=M open_price=1983',
                          'raw_price,inst=EUR_USD,price_comp=M higher_price=1983',
                          'raw_price,inst=EUR_USD,price_comp=M lower_price=1983',
                          'raw_price,inst=EUR_USD,price_comp=M close_price=1983',
                          'raw_price,inst=EUR_CHF,price_comp=M open_price=1983',
                          'raw_price,inst=EUR_CHF,price_comp=M higher_price=1983',
                          'raw_price,inst=EUR_CHF,price_comp=M lower_price=1983',
                          'raw_price,inst=EUR_CHF,price_comp=M close_price=1983',
                          'raw_price,inst=EUR_JPY,price_comp=M open_price=1983',
                          'raw_price,inst=EUR_JPY,price_comp=M higher_price=1983',
                          'raw_price,inst=EUR_JPY,price_comp=M lower_price=1983',
                          'raw_price,inst=EUR_JPY,price_comp=M close_price=1983',
                          'raw_price,inst=CHF_JPY,price_comp=M open_price=1983',
                          'raw_price,inst=CHF_JPY,price_comp=M higher_price=1983',
                          'raw_price,inst=CHF_JPY,price_comp=M lower_price=1983',
                          'raw_price,inst=CHF_JPY,price_comp=M close_price=1983']
        self.assertListEqual(actual_query, expected_query)


class TestTaskmanager(unittest.TestCase):
    # not in testing plan
    def test_data_extract(self):
        pass

    # not in testing plan
    def test_data_transform(self):
        pass

    # not in testing plan
    def test_data_predict(self):
        pass

    # not in testing plan
    def test_is_training_time(self):
        pass

    # not in testing plan
    def test_is_end_of_trading_time(self):
        pass

    def test_etp_job(self):
        tm = tman.TaskManager(insts=['EUR_USD'], gran='S5', comp='S5', meas='raw_price', field='close_price')
        etp_dict = tm.etp_job()




class TestTxuslib(unittest.TestCase):
    def test_create_directory(self):
        self.path = r'C:\Users\Jesus Garcia\Downloads\carpeta_test'

        # Positive Test Case: folder does not exist
        txuslib.create_dir(self.path)
        os.chdir(r'C:\Users\Jesus Garcia\Downloads\carpeta_test')
        self.assertEqual(r'C:\Users\Jesus Garcia\Downloads\carpeta_test', os.getcwd())

    def test_get_close_time_from_candle(self):
        self.json = {
            "instrument": "EUR_USD",
            "granularity": "S5",
            "candles": [
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09946",
                        "h": "1.09949",
                        "l": "1.09946",
                        "o": "1.09949"
                    },
                    "time": "2016-10-17T15:16:40.000000000Z",
                    "volume": 2
                },
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09947",
                        "h": "1.09947",
                        "l": "1.09947",
                        "o": "1.09946"
                    },
                    "time": "2016-10-17T15:16:45.000000000Z",
                    "volume": 1
                },
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09947",
                        "h": "1.09950",
                        "l": "1.09947",
                        "o": "1.09947"
                    },
                    "time": "2016-10-17T15:17:00.000000000Z",
                    "volume": 2
                },
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09951",
                        "h": "1.09951",
                        "l": "1.09951",
                        "o": "1.09947"
                    },
                    "time": "2016-10-17T15:17:05.000000000Z",
                    "volume": 1
                },
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09954",
                        "h": "1.09954",
                        "l": "1.09954",
                        "o": "1.09951"
                    },
                    "time": "2016-10-17T15:17:15.000000000Z",
                    "volume": 1
                },
                {
                    "complete": "false",
                    "mid": {
                        "c": "1.09961",
                        "h": "1.09961",
                        "l": "1.09958",
                        "o": "1.09954"
                    },
                    "time": "2016-10-17T15:17:20.000000000Z",
                    "volume": 3
                }
            ]
        }
        self.assertEqual("2016-10-17T15:17:15.000000000Z", txuslib.get_close_time_from_candle(self.json))

    def test_get_close_price_from_candle(self):
        self.json = {
            "candles": [
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09946",
                        "h": "1.09949",
                        "l": "1.09946",
                        "o": "1.09949"
                    },
                    "time": "2016-10-17T15:16:40.000000000Z",
                    "volume": 2
                },
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09947",
                        "h": "1.09947",
                        "l": "1.09947",
                        "o": "1.09946"
                    },
                    "time": "2016-10-17T15:16:45.000000000Z",
                    "volume": 1
                },
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09947",
                        "h": "1.09950",
                        "l": "1.09947",
                        "o": "1.09947"
                    },
                    "time": "2016-10-17T15:17:00.000000000Z",
                    "volume": 2
                },
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09951",
                        "h": "1.09951",
                        "l": "1.09951",
                        "o": "1.09947"
                    },
                    "time": "2016-10-17T15:17:05.000000000Z",
                    "volume": 1
                },
                {
                    "complete": "true",
                    "mid": {
                        "c": "1.09954",
                        "h": "1.09954",
                        "l": "1.09954",
                        "o": "1.09951"
                    },
                    "time": "2016-10-17T15:17:15.000000000Z",
                    "volume": 1
                },
                {
                    "complete": "false",
                    "mid": {
                        "c": "1.09961",
                        "h": "1.09961",
                        "l": "1.09958",
                        "o": "1.09954"
                    },
                    "time": "2016-10-17T15:17:20.000000000Z",
                    "volume": 3
                }
            ],
            "granularity": "S5",
            "instrument": "EUR/USD"
        }
        self.assertEqual(1.09954, txuslib.get_close_price_from_candle(self.json))

    def test_get_price_from_order(self):
        self.json = {
            "lastTransactionID": "6368",
            "orderCreateTransaction": {
                "accountID": "<ACCOUNT>",
                "batchID": "6367",
                "id": "6367",
                "instrument": "EUR_USD",
                "positionFill": "DEFAULT",
                "reason": "CLIENT_ORDER",
                "time": "2016-06-22T18:41:29.264030555Z",
                "timeInForce": "FOK",
                "type": "MARKET_ORDER",
                "units": "100",
                "userID": "USERID"
            },
            "orderFillTransaction": {
                "accountBalance": "43650.75945",
                "accountID": "<ACCOUNT>",
                "batchID": "6367",
                "financing": "0.00000",
                "id": "6368",
                "instrument": "EUR_USD",
                "orderID": "6367",
                "pl": "0.00000",
                "price": "1.13027",
                "reason": "MARKET_ORDER",
                "time": "2016-06-22T18:41:29.264030555Z",
                "tradeOpened": {
                    "tradeID": "6368",
                    "units": "100"
                },
                "type": "ORDER_FILL",
                "units": "100",
                "userID": "USERID"
            },
            "relatedTransactionIDs": [
                "6367",
                "6368"
            ]
        }
        self.assertEqual(1.13027, txuslib.get_price_from_order(self.json))

    def test_get_trade_id_from_order(self):
        self.json = {
            "lastTransactionID": "6368",
            "orderCreateTransaction": {
                "accountID": "<ACCOUNT>",
                "batchID": "6367",
                "id": "6367",
                "instrument": "EUR_USD",
                "positionFill": "DEFAULT",
                "reason": "CLIENT_ORDER",
                "time": "2016-06-22T18:41:29.264030555Z",
                "timeInForce": "FOK",
                "type": "MARKET_ORDER",
                "units": "100",
                "userID": "USERID"
            },
            "orderFillTransaction": {
                "accountBalance": "43650.75945",
                "accountID": "<ACCOUNT>",
                "batchID": "6367",
                "financing": "0.00000",
                "id": "6368",
                "instrument": "EUR_USD",
                "orderID": "6367",
                "pl": "0.00000",
                "price": "1.13027",
                "reason": "MARKET_ORDER",
                "time": "2016-06-22T18:41:29.264030555Z",
                "tradeOpened": {
                    "tradeID": "6368",
                    "units": "100"
                },
                "type": "ORDER_FILL",
                "units": "100",
                "userID": "USERID"
            },
            "relatedTransactionIDs": [
                "6367",
                "6368"
            ]
        }
        self.assertEqual("6368", txuslib.get_trade_id_from_order(self.json))

    def test_get_nb_seconds(self):
        self.end_date = '2020-01-01T00:00:00.000000000Z'
        self.start_date = '1970-01-01T00:00:00.000000000Z'
        self.assertEqual(1577836800, txuslib.get_nb_seconds(self.start_date, self.end_date))

    def test_get_nb_seconds_1900(self):
        self.date = '2020-01-01T00:00:00.000000000Z'
        self.assertEqual(3786825600, txuslib.get_nb_seconds_1900(self.date))

    def test_conv_unix_to_oanda_json(self):
        self.date = "2020-05-21 15:41:33.985456782"
        self.assertEqual("2020-05-21T15:41:33.985456782Z", txuslib.conv_unix_to_oanda_json(self.date))

    def test_conv_unix_to_x_www_form_url_encoded(self):
        self.date = "2020-05-21 15:41:33.985456782"
        self.assertEqual("2020-05-21T15%3A41%3A33.985456782Z", txuslib.conv_unix_to_x_www_form_url_encoded(self.date))

    def test_conv_oanda_json_to_x_www_form_url_encoded(self):
        self.date = "2020-05-21T15:41:33.985456782Z"
        self.assertEqual("2020-05-21T15%3A41%3A33.985456782Z",
                         txuslib.conv_oanda_json_to_x_www_form_url_encoded(self.date))

    def test_round_up(self):
        self.nb1 = 4.5
        self.nb2 = 6.3456
        self.nb3 = 897.54647
        self.nb4 = -567.456
        self.assertEqual(5, txuslib.round_up(self.nb1))
        self.assertEqual(5, txuslib.round_up(self.nb1, 0))
        self.assertEqual(6.35, txuslib.round_up(self.nb2, 2))
        self.assertEqual(7, txuslib.round_up(self.nb2, 0))
        self.assertEqual(900, txuslib.round_up(self.nb3, -2))
        self.assertEqual(898, txuslib.round_up(self.nb3, 0))
        self.assertEqual(-500, txuslib.round_up(self.nb4, -2))

    def test_round_down(self):
        self.nb1 = 1.45632
        self.nb2 = 1.4
        self.nb3 = 897.54647
        self.nb4 = -567.456
        self.nb5 = 0.9
        self.assertEqual(1, txuslib.round_down(self.nb1))
        self.assertEqual(0, txuslib.round_down(self.nb5))

    def test_oanda_date_format_break_down(self):
        self.date = '1983-08-25T07:35:27.456Z'
        self.assertEqual(int('1983'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['year'])
        self.assertEqual(int('08'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['month'])
        self.assertEqual(int('25'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['day'])
        self.assertEqual(int('07'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['hour'])
        self.assertEqual(int('35'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['minute'])
        self.assertEqual(int('27'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['second'])
        self.assertEqual(int('456'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['microsecond'])

    def test_get_nb_candles_two_dates(self):
        # TC 1: from Monday to Friday
        self.start_date = '2020-05-18T00:00:00.000000000Z'
        self.end_date = '2020-05-23T00:00:00.000000000Z'
        self.gran = 'S5'
        self.assertEqual(86400, txuslib.get_nb_candles_two_dates(self.start_date, self.end_date, self.gran))

        # TC 2: from Monday to Sunday (number of candles should be the same as in TC1, as WE do not count)
        self.end_date = '2020-05-25T00:00:00.000000000Z'
        self.assertEqual(86400, txuslib.get_nb_candles_two_dates(self.start_date, self.end_date, self.gran))

    def test_join_js_candles(self):
        self.orgnl_json = {
            "candles": [
                {
                    "ask": {
                        "c": "119.461",
                        "h": "120.473",
                        "l": "118.713",
                        "o": "120.245"
                    },
                    "complete": 'true',
                    "time": "2016-01-03T22:00:00.000000000Z",
                    "volume": 42744
                }
            ],
            "granularity": "D",
            "instrument": "USD/JPY"
        }
        self.new_json = {
            "candles": [
                {
                    "ask": {
                        "c": "119.080",
                        "h": "119.725",
                        "l": "118.801",
                        "o": "119.461"
                    },
                    "complete": 'true',
                    "time": "2016-01-04T22:00:00.000000000Z",
                    "volume": 33404
                }
            ],
            "granularity": "D",
            "instrument": "USD/JPY"
        }
        self.final_json = {
            "candles": [
                {
                    "ask": {
                        "c": "119.461",
                        "h": "120.473",
                        "l": "118.713",
                        "o": "120.245"
                    },
                    "complete": 'true',
                    "time": "2016-01-03T22:00:00.000000000Z",
                    "volume": 42744
                },
                {
                    "ask": {
                        "c": "119.080",
                        "h": "119.725",
                        "l": "118.801",
                        "o": "119.461"
                    },
                    "complete": 'true',
                    "time": "2016-01-04T22:00:00.000000000Z",
                    "volume": 33404
                }
            ],
            "granularity": "D",
            "instrument": "USD/JPY"
        }
        self.assertEqual(self.final_json, txuslib.join_js_candles(self.orgnl_json, self.new_json))

    def test_is_earlier_date(self):
        # TC 1: positive
        self.earlier_date = '2017-03-21T07:21:00.000000000Z'
        self.later_date = '2020-05-28T17:30:00.000000000Z'
        self.assertEqual(True, txuslib.is_earlier_date(self.earlier_date, self.later_date))

        # TC 2: negative
        self.earlier_date = '2016-02-21T04:34:00.000000000Z'
        self.later_date = '2021-12-23T23:30:00.000000000Z'
        self.assertEqual(False, txuslib.is_earlier_date(self.later_date, self.earlier_date))

    def test_weighted_slope_sum(self):
        self.data_array = np.array((1, 7, 5, 3, 8))
        self.assertEqual(0.5, txuslib.slope_sum(self.data_array))

    def test_linear_wma(self):
        self.data_array = np.arange(1,11)
        self.price = txuslib.linear_wma(pd.DataFrame({'prices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}), 2)
        pd.testing.assert_frame_equal(
            pd.DataFrame(
                {'prices':
                     [np.NaN, 1.66667, 2.66667, 3.66667, 4.66667, 5.66667, 6.66667, 7.66667, 8.66667, 9.66667]}),
            self.price)




if __name__ == '__main__':
    unittest.main()
