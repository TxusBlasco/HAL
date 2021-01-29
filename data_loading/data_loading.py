from influxdb_client import InfluxDBClient  # pip install influxdb-client[extra] for pandas data frame query
from txus_library import txuslib
import config_data.constants as conf
import pandas as pd
from datetime import datetime


class ApiQueryError(Exception):
    pass


class DataLoading:

    """
    ####################################################################################################################
    DataLoading recovers data from Influx DB
    ####################################################################################################################
    """

    def __init__(self, insts: list, meas: str, comp: str, field: str, bucket='bucket',
                 start_query='-21h', stop_query='now()'):
        self.token = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['token']
        self.org = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['org']
        self.bucket = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB'][bucket]
        self.url = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['url']
        self.insts = insts
        self.meas = meas
        self.comp = comp
        self.field = field  # o, h, l, c price
        self.start_query = start_query  # start query window
        self.stop_query = stop_query  # end query window

    # Extracts all the CLOSE candles from 6AM to 3AM of the previous day (price data from last day)
    # for each instrument on the list self.insts
    def get_training_df(self) -> dict:
        print('[INFO] running: %s at %s' % (self.get_training_df, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        query_api = client.query_api()
        inst_dict = {}
        try:
            for inst in self.insts:
                query = 'from(bucket: "%s") ' \
                         '|> range(start: %s, stop: %s) ' \
                         '|> filter(fn: (r) => r["_measurement"] == "%s") ' \
                         '|> filter(fn: (r) => r["_field"] == "%s") ' \
                         '|> filter(fn: (r) => r["inst"] == "%s") ' \
                         '|> filter(fn: (r) => r["comp"] == "%s") ' \
                         % (self.bucket, self.start_query, self.stop_query, self.meas, self.field, inst, self.comp)
                data_frame = query_api.query_data_frame(query)
                inst_dict[inst] = pd.DataFrame(data_frame)
                query_api.__del__()
                client.__del__()
        except ApiQueryError:
            print('[ERROR] Could not retrieve any data from Influx DB')
        else:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.expand_frame_repr', False)
            return inst_dict
            # dictionary of pandas data frames formed by insts requested
            # each data frame has the following columns:
            # result = _result
            # table = 0
            # _start = timestamp 2021-01-12 08:55:12.909639+00:00 -> start range for query
            # _stop = timestamp 2021-01-12 08:55:12.909639+00:00 -> stop query window
            # _time = timestamp 2021-01-12 08:55:12.909639+00:00 -> time for the request
            # _value = price value 1.23456
            # _field = close_price -> o, h, l, c
            # _measurement = raw_price -> could also be predicted_price
            # inst =EUR_USD -> instrument
            # price_comp = M -> price component mid, bid, ask: M, B, A


