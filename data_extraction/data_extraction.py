import requests  # pip install
import json
from txus_library import txuslib
import config_data.constants as conf


class DataExtraction:

    """
    ####################################################################################################################
    DataExtraction recovers all data price related from OANDA Broker REST API
    ####################################################################################################################
    """

    class Deco:
        # receives a function that returns a dictionary with price data
        # and turns the value to numeric for the timestamp key
        # example: 2020-11-19T06:48:00.000000000Z --> 20201119064800
        # dictionary must have the keys: timestamp, open, higher, lower and close
        # DEPRECATED
        @classmethod
        def timestamp_to_num(cls, f):
            def wrapper(*args, **kwargs):
                original = f(*args, **kwargs)
                d = original['timestamp']
                year = d.split('-')[0]
                month = d.split('-')[1]
                rest = d.split('-')[2]
                day = rest.split('T')[0]
                rest_day = rest.split('T')[1]
                hour = rest_day.split(':')[0]
                minute = rest_day.split(':')[1]
                rest_minute = rest_day.split(':')[2]
                second = rest_minute.split('.')[0]
                num_d = year + month + day + hour + minute + second
                original['timestamp'] = num_d
                return original
            return wrapper

        # TODO pending unit testing
        @classmethod
        def timestamp_to_rfc3339(cls, f):
            def wrapper(*args, **kwargs):
                original = f(*args, **kwargs)
                d = original['timestamp']
                rfc_1 = d.split('.')[0]
                rfc = rfc_1 + '.00Z'
                original['timestamp'] = rfc
                return original
            return wrapper

    # inst: trading instrument (EUR_USD, EUR_GBP, etc.)
    # gran: granularity (S5, S10, etc.)
    # comp: price comp: A (Ask), B (Bid), M (mid)
    # units : amount of an instrument to be bought
    def __init__(self, insts: list, gran: str, comp: str, start_date=0, end_date=0):
        self.insts = insts
        self.gran = gran
        self.comp = comp
        self.start_date = start_date
        self.end_date = end_date
        self.TOKEN = txuslib.get_data_from_yaml(conf.OANDA_TOKEN_PATH)['Oanda']['token']
        self.URL = conf.OANDA_TEST_ENV_URL

    # Fetch last candlestick data for a given instrument and returns the json file
    # returns a list of two elements: the timestamp and the price for that timestamp
    @Deco.timestamp_to_rfc3339
    def get_last_candle_from_db(self, inst: str) -> dict:
        print("[INFO] Running ", self.get_last_candle_from_db)
        head = {'Authorization': 'Bearer {}'.format(self.TOKEN)}
        get_url = \
            self.URL + r'/v3/instruments/' + inst + r'/candles?price=' + self.comp \
            + r'&granularity=' + self.gran + r'&count=2'
        try:
            response = requests.get(get_url, headers=head)
            js = json.loads(response.content)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("[ERROR] Error: " + str(e))
        else:
            price_dict = {
                'status_code': response.status_code,
                'timestamp': js['candles'][0]['time'],
                'open': float(js['candles'][0]['mid']['o']),
                'higher': float(js['candles'][0]['mid']['h']),
                'lower': float(js['candles'][0]['mid']['l']),
                'close': float(js['candles'][0]['mid']['c'])
            }
            return price_dict

    # builds a dictionary made of instruments where each inst presents the status code, the timestamp and ohlc prices
    # { 'EUR_USD:
    #       {   'status_code': 200
    #           'timestamp': 2020-11-19T06:48:00.000000000Z
    #           'open': 1.6721,
    #           'higher': 1.7554,
    #           'lower': 1.3454,
    #           'close': 1.6543
    #       },
    # { 'EUR_JPY:
    #       {   'status_code': 200
    #           'timestamp': 2020-11-19T06:48:00.000000000Z
    #           'open': 1.6721,
    #           'higher': 1.7554,
    #           'lower': 1.3454,
    #           'close': 1.6543
    #       },
    # { 'EUR_AUD:
    #       {   'status_code': 200
    #           'timestamp': 2020-11-19T06:48:00.000000000Z
    #           'open': 1.6721,
    #           'higher': 1.7554,
    #           'lower': 1.3454,
    #           'close': 1.6543
    #       }
    # }
    def inst_dict_constructor(self):
        inst_dict = {}
        for key in self.insts:
            inst_dict[key] = self.get_last_candle_from_db(key)
        return inst_dict


    # gets a set of price of an instrument between two dates
    # start_date: time where to start the price collection in OANDA format: YYYY-MM-DDThh:mm:ss.000000000Z
    # end_date: time where to end the price collection in OANDA format: YYYY-MM-DDThh:mm:ss.000000000Z
    # returns a dictionary with the prices related to their time
    # DEPRECATED
    def get_bulk_price_data_set(self):
        print("[INFO] Running ", self.get_bulk_price_data_set)
        new_json = {
            "candles": [],
            "granularity": self.gran,
            "instrument": self.inst
        }
        head = {'Authorization': 'Bearer ' + self.TOKEN}

        _start_date = self.start_date
        _end_date = self.end_date

        while txuslib.is_earlier_date(_start_date, _end_date):  # add 5000 items in each loop
            get_url = \
                self.URL + r'/v3/insts/' + self.inst + r'/candles?price=' + self.comp + r'&from=' + \
                txuslib.conv_oanda_json_to_x_www_form_url_encoded(_start_date) + \
                r'&count=5000&granularity=' + self.gran
            response = requests.get(get_url, headers=head)
            js = json.loads(response.content)
            response.raise_for_status()
            print("[INFO] Status code: {}".format(response.status_code))
            last_date_from_json = txuslib.get_close_time_from_candle(js)
            _start_date = last_date_from_json
            new_json = txuslib.join_js_candles(new_json, js)

        last_date_from_json = txuslib.get_close_time_from_candle(new_json)
        while txuslib.is_earlier_date(_end_date, last_date_from_json):  # remove the items surpassing the end date
            new_json['candles'].pop(-1)
            last_date_from_json = txuslib.get_close_time_from_candle(new_json)

        return new_json



