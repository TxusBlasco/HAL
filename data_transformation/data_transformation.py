from influxdb_client import InfluxDBClient  # pip install influxdb-client
from influxdb_client.client.write_api import SYNCHRONOUS
from txus_library import txuslib
import config_data.constants as conf
import numpy as np


class ApiWritingError(Exception):
    pass


class DataTransformation:
    """
    ####################################################################################################################
    DataTransformation takes data, cleans it, transforms it and saves it in InfluxDB
    ####################################################################################################################
    """

    def __init__(self, price_dict: dict, insts: list, meas: str, comp: str, bucket='bucket'):
        self.price_dict = price_dict
        self.token = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['token']
        self.org = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['org']
        self.bucket = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB'][bucket]
        self.url = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['url']
        self.timestamp = price_dict['timestamp']  # time
        self.open_price = price_dict['open_price']  # field value
        self.higher_price = price_dict['higher_price']  # field value
        self.lower_price = price_dict['lower_price']  # field value
        self.close_price = price_dict['close_price']  # field value
        self.insts = insts  # list of instruments (ex: EUR_USD) (tag key)
        self.meas = meas  # meas (raw_price or pred_price)
        self.comp = comp  # comp (bid, ask or mid) (tag key)

    # Syntax
    # myMeasurement,tag1=value1,tag2=value2 fieldKey="fieldValue" 1556813561098000000
    # meas name : instrument
    # tag set : open (o), high (h), low (l), close (c)
    # data : price
    def set_price_data_db(self):
        print("[INFO] Running ", self.set_price_data_db)
        client = InfluxDBClient(url=self.url, token=self.token)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        query_list = []
        for inst in self.insts:
            # Line protocol:
            o = '%s,inst=%s,price_comp=%s open_price=%s' % (self.meas, inst, self.comp, self.open_price)
            h = '%s,inst=%s,price_comp=%s higher_price=%s' % (self.meas, inst, self.comp, self.higher_price)
            l = '%s,inst=%s,price_comp=%s lower_price=%s' % (self.meas, inst, self.comp, self.lower_price)
            c = '%s,inst=%s,price_comp=%s close_price=%s' % (self.meas, inst, self.comp, self.close_price)
            aux_list = [o, h, l, c]
            query_list.append(aux_list)
        query_list_reshaped = list(np.reshape(query_list, len(self.insts)*4))
        try:
            write_api.write(self.bucket, self.org, query_list_reshaped)
            write_api.__del__()
            client.__del__()
        except ApiWritingError:
            print('[ERROR] Could not write to Influx DB API')
        finally:
            return query_list_reshaped

    # TODO
    # gets price data and provides the real time tendency value (slope)
    def set_slope(self):
        pass

    # TODO
    # gets price data and provides the real time variance value
    def set_variance(self):
        pass

    # TODO
    # filters price time series to smoothen the curve with fourier transform
    def fourier_filter(self):
        pass

    # filters price time series to smoothen the curve with least squares approach
    def least_square_filter(self):
        pass
