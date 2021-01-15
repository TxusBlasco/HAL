import unittest
from data_extraction import data_extraction as dext
from data_transformation import data_transformation as dtrans
from data_prediction import data_prediction as dpred
import time
from datetime import datetime, timedelta
import schedule  # pip install necessary
from config_data import constants as cs
from txus_library import txuslib
from task_manager import task_manager as tm
import config_data.constants as conf
from data_loading import data_loading as dload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class IntegrationTestDataExtraction(unittest.TestCase):
    def test_get_last_candle_from_db(self):
        de = dext.DataExtraction(inst='EUR_USD', gran='S5', comp='M')
        price_dict = de.get_last_candle_from_db()
        self.assertEqual(price_dict['status_code'], 200)
        self.assertIsInstance(price_dict['timestamp'], str)
        self.assertIsInstance(price_dict['open_price'], float)
        self.assertIsInstance(price_dict['higher_price'], float)
        self.assertIsInstance(price_dict['lower_price'], float)
        self.assertIsInstance(price_dict['close_price'], float)


class IntegrationTestDataTransformation(unittest.TestCase):
    # requires ulterior recovery from data in Influx DB => DataLoading.get_training_df should work ok
    def test_set_price_data_db(self):
        for _ in range(10):
            price_dict = {'open_price': _,
                          'higher_price': 100 + _,
                          'lower_price': 200 + _,
                          'close_price': 300 + _}

            dt = dtrans.DataTransformation(price_dict=price_dict,
                                           insts=['EUR_USD', 'EUR_CHF', 'EUR_JPY'],
                                           meas='raw_price',
                                           comp='M',
                                           bucket='test_env_bucket')
            dt.set_price_data_db()
        field = [key for key in price_dict.keys()]
        aux_dict = {}
        for elem in field:
            dl = dload.DataLoading(insts=['EUR_USD', 'EUR_CHF', 'EUR_JPY'],
                                   meas='raw_price',
                                   comp='M',
                                   field=elem,
                                   bucket='test_env_bucket',
                                   start_query='-1m')
            dl_df = dl.get_training_df()  # dict of df
            aux_dict[elem] = dl_df
        expected_result = {
            {
                'open_price':
            }
        }
        self.assertEqual(aux_dict['close_price']['EUR_USD']['_value'].iloc[-1], expected_result)


class IntegrationTestDataLoading(unittest.TestCase):
    def test_get_training_df(self):
        dl = dload.DataLoading(insts=['EUR_USD'],
                               meas='raw_price',
                               comp='M',
                               field='close_price',
                               bucket='test_env_bucket')
        unittest_passed = dl.get_training_df()[1]
        print(unittest_passed)
        self.assertTrue(unittest_passed)

if __name__ == '__main__':
    unittest.main()
