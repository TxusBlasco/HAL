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

    def __init__(self, inst_dict: dict, meas: str, comp: str, bucket='bucket'):
        self.inst_dict = inst_dict
        self.token = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['token']
        self.org = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['org']
        self.bucket = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB'][bucket]
        self.url = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['InfluxDB']['url']
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
        for inst in self.inst_dict.keys():
            # Line protocol:
            o = '%s,inst=%s,comp=%s open=%s' % (self.meas, inst, self.comp, self.inst_dict[inst]['open'])
            h = '%s,inst=%s,comp=%s higher=%s' % (self.meas, inst, self.comp, self.inst_dict[inst]['higher'])
            l = '%s,inst=%s,comp=%s lower=%s' % (self.meas, inst, self.comp, self.inst_dict[inst]['lower'])
            c = '%s,inst=%s,comp=%s close=%s' % (self.meas, inst, self.comp, self.inst_dict[inst]['close'])
            aux_list = [o, h, l, c]
            query_list.append(aux_list)
        query_list_reshaped = list(np.reshape(query_list, len(self.inst_dict)*4))
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
