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
from pandas._testing import assert_frame_equal


class TestDataExtraction(unittest.TestCase):
    def test_inst_dict_constructor(self):
        print('[INFO] running', self.test_inst_dict_constructor)
        insts=['EUR_USD', 'EUR_JPY', 'EUR_AUD']
        keys = ['status_code', 'timestamp', 'open', 'higher', 'lower', 'close']
        de = dext.DataExtraction(insts=insts, comp='M', gran='S5')
        insts_dict = de.inst_dict_constructor()
        for inst in insts:
            self.assertIn(inst, insts_dict.keys())
            for key in keys:
                self.assertIn(key, insts_dict[inst].keys(), msg='[ERROR] test_inst_dict_constructor')

class TestDataPrediction(unittest.TestCase):
    # the test creates h5 and joblib files in the \test directory
    def test_lstm_trainer(self):
        print('[INFO] running', self.test_lstm_trainer)
        dataset_train = pd.read_csv(conf.TESTING_TRAIN_CSV)
        df_filt = dataset_train[['c']]
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close')
        score = float(dp.lstm_trainer(df_filt))
        self.assertTrue(score < 1, msg='[TEST FAIL] Calculation error is higher than 0.3')
        self.assertTrue(os.path.exists(conf.SCALER_PATH),
                        msg='[TEST FAIL] %s file does not exist' % conf.SCALER_PATH)
        self.assertTrue(os.path.exists(conf.MODEL_PATH), msg='[TEST FAIL] %s file does not exist' % conf.MODEL_PATH)
        try:
            shutil.rmtree(conf.MODEL_DIR)  # removes the folder after the test
        except OSError as e:
            print("[ERROR] test_lstm_trainer. Deletion of the directory %s failed : %s" % (conf.MODEL_DIR, e.strerror))


    def test_lstm_tester(self):
        print('[INFO] running', self.test_lstm_tester)
        try:
            os.makedirs(conf.MODEL_DIR)
        except OSError as e:
            print("[ERROR] test_lstm_trainer. Creation of the directory %s failed : %s " % (conf.MODEL_DIR, e.strerror))

        shutil.copyfile(conf.TESTING_MODEL_PATH, conf.MODEL_PATH)
        shutil.copyfile(conf.TESTING_SCALER_PATH, conf.SCALER_PATH)

        dataset_test = pd.read_csv(conf.TESTING_TEST_CSV)

        df_filt = dataset_test[['c']].iloc[-100:]
        df_filt = df_filt.reset_index(drop=True)
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close')
        ret = dp.lstm_tester(df_filt)
        try:
            shutil.rmtree(conf.MODEL_DIR)  # removes the folder after the test
        except OSError as e:
            print("[ERROR] test_lstm_tester. Deletion of the directory %s failed : %s" % (conf.MODEL_DIR, e.strerror))
        self.assertTrue(ret['test_score'] < 2, msg='[ERROR] test_lstm_tester. Calculation error is higher than 1%')
        self.assertTrue(type(ret['predicted_data']) == np.ndarray, msg='[ERROR] test_lstm_tester. Return is not a list')

    def test_dynamic_predictor(self):
        print('[INFO] running', self.test_dynamic_predictor)
        df = pd.read_csv(conf.TESTING_TRAIN_CSV, sep=',')
        df = pd.DataFrame(df)['c'].iloc[-60:]
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close')
        pred_price = dp.dynamic_predictor(
            pred_df=df,
            model_path=conf.TESTING_MODEL_PATH,
            scaler_path=conf.TESTING_SCALER_PATH)
        self.assertAlmostEqual(pred_price, 1.18, msg='[ERROR] test_dynamic_predictor', delta=0.01)

    def test_lstm_load_model(self):
        print('[INFO] running', self.test_lstm_load_model)
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close')
        loaded_model = dp.lstm_load_model(conf.TESTING_MODEL_PATH)
        print(type(loaded_model))
        self.assertEqual(
            '<class \'tensorflow.python.keras.engine.sequential.Sequential\'>',
            str(type(loaded_model)),
            msg='[ERROR] test_lstm_load_model')

    def test_lstm_load_scaler(self):
        print('[INFO] running', self.test_lstm_load_scaler)
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close')
        loaded_scaler = dp.lstm_load_scaler(conf.TESTING_SCALER_PATH)
        print(type(loaded_scaler))
        self.assertEqual(
            '<class \'sklearn.preprocessing._data.MinMaxScaler\'>',
            str(type(loaded_scaler)),
            msg='[ERROR] test_lstm_load_scaler')

    def test_dataframe_splitter(self):
        print('[INFO] running', self.test_dataframe_splitter)
        df = pd.DataFrame(np.random.random(15000), columns=['random'])
        dp = dpred.DataPrediction(insts=['EUR_USD'], meas='raw_price', comp='M', field='close_price')
        dp_train = dp.data_frame_splitter(df)['train_data']
        dp_test = dp.data_frame_splitter(df)['test_data']
        self.assertEqual(
            (dp_train.shape, dp_test.shape),
            ((13500, 1), (1500, 1)),
            msg='[ERROR] test_data_frame_splitter')


class TestDataTransformation(unittest.TestCase):
    # this test case is testing the business logic part of the method, not the API connection (tested in integration t.)
    def test_set_price_data_db(self):
        print('[INFO] running', self.test_set_price_data_db)
        inst_dict = {
            'EUR_USD':
                {'open': 1.3,
                 'higher': 1.5,
                 'lower': 1.2,
                 'close': 1.4},
            'EUR_CHF':
                {'open': 1.7,
                 'higher': 1.9,
                 'lower': 1.6,
                 'close': 1.8},
            'EUR_JPY':
                {'open': 200,
                 'higher': 400,
                 'lower': 100,
                 'close': 300}
        }
        dt = dtrans.DataTransformation(inst_dict=inst_dict,
                                       meas='raw_price',
                                       comp='M',
                                       bucket='test_env_bucket')
        actual_query = dt.set_price_data_db()
        expected_query = ['raw_price,inst=EUR_USD,comp=M open=1.3',
                          'raw_price,inst=EUR_USD,comp=M higher=1.5',
                          'raw_price,inst=EUR_USD,comp=M lower=1.2',
                          'raw_price,inst=EUR_USD,comp=M close=1.4',
                          'raw_price,inst=EUR_CHF,comp=M open=1.7',
                          'raw_price,inst=EUR_CHF,comp=M higher=1.9',
                          'raw_price,inst=EUR_CHF,comp=M lower=1.6',
                          'raw_price,inst=EUR_CHF,comp=M close=1.8',
                          'raw_price,inst=EUR_JPY,comp=M open=200',
                          'raw_price,inst=EUR_JPY,comp=M higher=400',
                          'raw_price,inst=EUR_JPY,comp=M lower=100',
                          'raw_price,inst=EUR_JPY,comp=M close=300']
        self.assertListEqual(actual_query, expected_query, '[ERROR] test_set_price_data_db failed')


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





class TestTxuslib(unittest.TestCase):
    def test_create_directory(self):
        self.path = r'C:\Users\Jesus Garcia\Downloads\carpeta_test'

        # Positive Test Case: folder does not exist
        txuslib.create_dir(self.path)
        os.chdir(r'C:\Users\Jesus Garcia\Downloads\carpeta_test')
        self.assertEqual(
            r'C:\Users\Jesus Garcia\Downloads\carpeta_test',
            os.getcwd(),
            msg='[ERROR] test_create_directory')

    def test_get_close_time_from_candle(self):
        print('[INFO] running', self.test_get_close_time_from_candle)
        self.json_file = {
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
        self.assertEqual("2016-10-17T15:17:15.000000000Z", txuslib.get_close_time_from_candle(self.json_file))

    def test_get_close_price_from_candle(self):
        print('[INFO] running', self.test_get_close_price_from_candle)
        self.json_file = {
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
                        "o": "1.09953"
                    },
                    "time": "2016-10-17T15:17:20.000000000Z",
                    "volume": 3
                }
            ],
            "granularity": "S5",
            "instrument": "EUR/USD"
        }
        self.assertEqual(1.09954, txuslib.get_close_price_from_candle(self.json_file))

    def test_get_price_from_order(self):
        print('[INFO] running', self.test_get_price_from_order)
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
        print('[INFO] running', self.test_get_trade_id_from_order)
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
        print('[INFO] running', self.test_get_nb_seconds)
        self.end_date = '2020-01-01T00:00:00.000000000Z'
        self.start_date = '1970-01-01T00:00:00.000000000Z'
        self.assertEqual(1577836800, txuslib.get_nb_seconds(self.start_date, self.end_date))

    def test_get_nb_seconds_1900(self):
        print('[INFO] running', self.test_get_nb_seconds_1900)
        self.date = '2020-01-01T00:00:00.000000000Z'
        self.assertEqual(3786825600, txuslib.get_nb_seconds_1900(self.date))

    def test_conv_unix_to_oanda_json(self):
        print('[INFO] running', self.test_conv_unix_to_oanda_json)
        self.date = "2020-05-21 15:41:33.985456782"
        self.assertEqual("2020-05-21T15:41:33.985456782Z", txuslib.conv_unix_to_oanda_json(self.date))

    def test_conv_unix_to_x_www_form_url_encoded(self):
        print('[INFO] running', self.test_conv_unix_to_x_www_form_url_encoded)
        self.date = "2020-05-21 15:41:33.985456782"
        self.assertEqual("2020-05-21T15%3A41%3A33.985456782Z", txuslib.conv_unix_to_x_www_form_url_encoded(self.date))

    def test_conv_oanda_json_to_x_www_form_url_encoded(self):
        print('[INFO] running', self.test_conv_oanda_json_to_x_www_form_url_encoded)
        self.date = "2020-05-21T15:41:33.985456782Z"
        self.assertEqual("2020-05-21T15%3A41%3A33.985456782Z",
                         txuslib.conv_oanda_json_to_x_www_form_url_encoded(self.date))

    def test_round_up(self):
        print('[INFO] running', self.test_round_up)
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
        print('[INFO] running', self.test_round_down)
        self.nb1 = 1.45632
        self.nb2 = 1.4
        self.nb3 = 897.54647
        self.nb4 = -567.456
        self.nb5 = 0.9
        self.assertEqual(1, txuslib.round_down(self.nb1))
        self.assertEqual(0, txuslib.round_down(self.nb5))

    def test_oanda_date_format_break_down(self):
        print('[INFO] running', self.test_oanda_date_format_break_down)
        self.date = '1983-08-25T07:35:27.456Z'
        self.assertEqual(int('1983'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['year'])
        self.assertEqual(int('08'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['month'])
        self.assertEqual(int('25'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['day'])
        self.assertEqual(int('07'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['hour'])
        self.assertEqual(int('35'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['minute'])
        self.assertEqual(int('27'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['second'])
        self.assertEqual(int('456'), txuslib.oanda_date_format_break_down('1983-08-25T07:35:27.456Z')['microsecond'])

    def test_get_nb_candles_two_dates(self):
        print('[INFO] running', self.test_get_nb_candles_two_dates)
        # TC 1: from Monday to Friday
        self.start_date = '2020-05-18T00:00:00.000000000Z'
        self.end_date = '2020-05-23T00:00:00.000000000Z'
        self.gran = 'S5'
        self.assertEqual(86400, txuslib.get_nb_candles_two_dates(self.start_date, self.end_date, self.gran))

        # TC 2: from Monday to Sunday (number of candles should be the same as in TC1, as WE do not count)
        self.end_date = '2020-05-25T00:00:00.000000000Z'
        self.assertEqual(86400, txuslib.get_nb_candles_two_dates(self.start_date, self.end_date, self.gran))

    def test_join_js_candles(self):
        print('[INFO] running', self.test_join_js_candles)
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
        print('[INFO] running', self.test_is_earlier_date)
        # TC 1: positive
        self.earlier_date = '2017-03-21T07:21:00.000000000Z'
        self.later_date = '2020-05-28T17:30:00.000000000Z'
        self.assertEqual(True, txuslib.is_earlier_date(self.earlier_date, self.later_date))

        # TC 2: negative
        self.earlier_date = '2016-02-21T04:34:00.000000000Z'
        self.later_date = '2021-12-23T23:30:00.000000000Z'
        self.assertEqual(False, txuslib.is_earlier_date(self.later_date, self.earlier_date))

    def test_weighted_slope_sum(self):
        print('[INFO] running', self.test_weighted_slope_sum)
        self.data_array = np.array((1, 7, 5, 3, 8))
        self.assertEqual(0.5, txuslib.slope_sum(self.data_array))

    def test_linear_wma(self):
        print('[INFO] running', self.test_linear_wma)
        self.data_array = np.arange(1, 11)
        self.price = txuslib.linear_wma(pd.DataFrame({'prices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}), shift=0, win_len=2)
        pd.testing.assert_frame_equal(
            pd.DataFrame(
                {'prices':
                    [np.NaN, 1.66667, 2.66667, 3.66667, 4.66667, 5.66667, 6.66667, 7.66667, 8.66667, 9.66667]}),
            self.price)

    def test_df_from_inst_dict(self):
        print('[INFO] running', self.test_df_from_inst_dict)
        inst_dict = {
            'EUR_USD':
                {'open': 1.3,
                 'higher': 1.5,
                 'lower': 1.2,
                 'close': 1.4},
            'EUR_CHF':
                {'open': 1.7,
                 'higher': 1.9,
                 'lower': 1.6,
                 'close': 1.8},
            'EUR_JPY':
                {'open': 200,
                 'higher': 400,
                 'lower': 100,
                 'close': 300}
        }
        assert_frame_equal(
            pd.DataFrame([[1.4, 1.8, 300]], columns=['EUR_USD', 'EUR_CHF', 'EUR_JPY']),
            txuslib.df_from_inst_dict(inst_dict, 'close'))


if __name__ == '__main__':
    unittest.main()
