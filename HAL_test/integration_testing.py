import unittest
from data_extraction import data_extraction as dext
from data_transformation import data_transformation as dtrans
from task_manager import task_manager as tman
import time
from datetime import datetime, timedelta
import schedule  # pip install necessary
from txus_library import txuslib
from task_manager import task_manager as tm
import config_data.constants as conf
from data_loading import data_loading as dload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class IntegrationTestDataExtraction(unittest.TestCase):
    def test_get_last_candle_from_db(self):
        de = dext.DataExtraction(insts=['EUR_USD'], gran='S5', comp='M')
        price_dict = de.get_last_candle_from_db('EUR_USD')
        self.assertEqual(price_dict['status_code'], 200)
        self.assertIsInstance(price_dict['timestamp'], str)
        self.assertIsInstance(price_dict['open_price'], float)
        self.assertIsInstance(price_dict['higher_price'], float)
        self.assertIsInstance(price_dict['lower_price'], float)
        self.assertIsInstance(price_dict['close_price'], float)


class IntegrationTestDataTransformationAndLoading(unittest.TestCase):
    # requires ulterior recovery from data in Influx DB => DataLoading.get_training_df should work ok
    def test_trans_load(self):
        inst_dict = {}
        for _ in range(10):
            inst_dict = {
                'EUR_USD':
                    {'open_price': _,
                     'higher_price': 100 + _,
                     'lower_price': 200 + _,
                     'close_price': 300 + _},
                'EUR_CHF':
                    {'open_price': 400 + _,
                     'higher_price': 500 + _,
                     'lower_price': 600 + _,
                     'close_price': 700 + _},
                'EUR_JPY':
                    {'open_price': 800 + _,
                     'higher_price': 900 + _,
                     'lower_price': 1000 + _,
                     'close_price': 1100 + _}
            }
            dt = dtrans.DataTransformation(inst_dict=inst_dict,
                                           meas='raw_price',
                                           comp='M',
                                           bucket='test_env_bucket')
            dt.set_price_data_db()
        inst_list = [key for key in inst_dict.keys()]
        aux_dict = {}
        for elem in ['open', 'higher', 'lower', 'close']:
            dl = dload.DataLoading(insts=inst_list,
                                   meas='raw_price',
                                   comp='M',
                                   field=elem,
                                   bucket='test_env_bucket',
                                   start_query='-1m')
            dl_dict = dl.get_training_df()  # dict of df
            aux_dict[elem] = dl_dict
        self.assertEqual(9, aux_dict['open']['EUR_USD']['_value'].iloc[-1])
        self.assertEqual(509, aux_dict['higher']['EUR_CHF']['_value'].iloc[-1])
        self.assertEqual(1009, aux_dict['lower']['EUR_JPY']['_value'].iloc[-1])
        self.assertEqual(309, aux_dict['close']['EUR_USD']['_value'].iloc[-1])

class IntegrationTestTaskManager(unittest.TestCase):
    def test_task_manager(self):
        insts = ['EUR_USD', 'EUR_CHF', 'EUR_JPY']
        tm = tman.TaskManager(insts=insts, meas='raw_price',comp='M', field='close', gran='S5')
        etp_dict_short = tm.etp_job(is_etl_testing=True)
        print('-------------------------------------------')
        print('[INFO] short etp:')
        print(etp_dict_short)
        print('-------------------------------------------')
        etp_dict_long = tm.etp_job(is_etlp_testing=True)
        print('[INFO] long etp:')
        print(etp_dict_long)
        print('-------------------------------------------')



if __name__ == '__main__':
    unittest.main()
