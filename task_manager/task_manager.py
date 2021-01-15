from data_extraction import data_extraction as dext
from data_transformation import data_transformation as dtrans
from data_loading import data_loading as dload
from data_prediction import data_prediction as dpred
import time
from datetime import date, datetime, timedelta
import schedule  # pip install schedule
import config_data.constants as conf
import pandas as pd


class DataExtractionError(Exception):
    pass


class DataTransformationError(Exception):
    pass


class AlgorithmTrainingError(Exception):
    pass


class DataPredictionError(Exception):
    pass


class TaskManager:

    def __init__(self,
                 insts: list,
                 gran: str,
                 comp: str,
                 meas: str,
                 field: str,
                 training_time=conf.TRAINING_TIME,
                 trading_time=conf.TRADING_TIME,
                 sunday_trade_time=conf.SUNDAY_TRADE_TIME,
                 end_of_trading_day=conf.END_OF_TRADING_DAY,
                 friday_end_trade_time=conf.FRIDAY_END_TRADE_TIME):

        self.insts = insts
        self.gran = gran
        self.comp = comp
        self.meas = meas
        self.field = field
        self.training_time = training_time
        self.trading_time = trading_time
        self.sunday_trade_time = sunday_trade_time
        self.end_of_trading_day = end_of_trading_day
        self.friday_end_trade_time = friday_end_trade_time

    # gets the last value of price from the broker API
    def data_extract(self, inst: str) -> dict:
        de = dext.DataExtraction(inst=inst, gran=self.gran, comp=self.comp)
        try:
            price_dict = de.get_last_candle_from_db()
            return price_dict
        except DataExtractionError:
            print('[ERROR] Data could not be extracted')

    # TODO return transformed price
    # saves extracted and transformed data in the Influx DB database
    def data_transform(self, inst: str, price_dict: dict):
        dt = dtrans.DataTransformation(inst=inst, price_dict=price_dict, meas=self.meas, comp=self.comp)
        try:
            dt.set_price_data_db()
            trans_price = 'pending'
            return trans_price
        except DataTransformationError:
            print('[ERROR] Data could not be transformed')

    # based on the last 60 values of data, predicts the next price value
    def data_predict(self, pred_df: pd.DataFrame) -> float:
        dp = dpred.DataPrediction(insts=self.insts, meas=self.meas, comp=self.comp, field=self.field)
        try:
            pred_price = dp.dynamic_predictor(pred_df=pred_df, model_path=conf.MODEL_PATH, scaler_path=conf.SCALER_PATH)
            return pred_price
        except DataPredictionError:
            print('[ERROR] Data could not be predicted')

    # will be true every day at training time (3:00 AM)
    def is_training_time(self):
        if datetime.now().strftime("%H:%M") == self.training_time:
            print('[INFO] Finished trading. Starting training now at ', self.training_time)
            test_passed = True
            return True
        else:
            return False

    # will be true on Fridays at the end of New York session
    def is_end_of_trading_time(self):
        if datetime.now().strftime("%A %H:%M") == self.end_of_trading_day + " " + self.friday_end_trade_time:
            print('[INFO] Finished weekly trading. Next start Sunday at ', self.sunday_trade_time)
            return True
        else:
            return False

    # for each instrument, extracts stream data from Oanda, saves it in InfluxDB and predicts next price value
    def etp_job(self) -> dict:  # job: Extract, Transform and Predict data
        print('[INFO] running: ', self.etp_job)
        print('[INFO] Starting trading now at ', datetime.now())
        pred_df = pd.DataFrame()
        ext_df = pd.DataFrame()
        ext_dict = {}
        pred_dict = {}
        data_ext_counter = 0
        while True:
            for inst in self.insts:  # perform ETL for each instrument
                ext_price_dict = self.data_extract(inst=inst)
                ext_dict[inst] = ext_price_dict['c']
                trans_price = self.data_transform(inst=inst, price_dict=ext_price_dict['c'])
            aux_ext_df = pd.DataFrame.from_dict(ext_dict)
            ext_df = ext_df.append(aux_ext_df, ignore_index=True)
            if data_ext_counter >= 60:
                for inst in self.insts:  # perform prediction for each instrument
                    # TODO: now data_predict can only predict one instrument, not a dict of instruments
                    # TODO: however, the rest of ETPL is prepared to manage a dictionary of instruments
                    pred_price = self.data_predict(ext_df[inst].iloc[-60:])
                    pred_dict[inst] = pred_price
            aux_pred_df = pd.DataFrame.from_dict(pred_dict)
            pred_df = pred_df.append(aux_pred_df, ignore_index=True)

            print('[INFO] extracted data: ', ext_df.iloc[-1:])
            print('[INFO] predicted data: ', pred_df.iloc[-1:])



            if self.is_training_time():
                break
            if self.is_end_of_trading_time():
                break
            data_ext_counter += 1
            time.sleep(5)
        return {'current_date_time': datetime.now().strftime("%Y-%M-%D_%H:%M"),
                'extracted_data': ext_df,
                'predicted_data': pred_df}

    # train the algorithm with data from the previous day
    # should finish the training before the planned daily start of trading
    def train_job(self):
        print('[INFO] running: ', self.train_job)
        try:
            dp = dpred.DataPrediction(insts=self.insts, comp=self.comp, meas=self.meas, field=self.field)
            dp.daily_train_test()
        except AlgorithmTrainingError:
            print('[ERROR] The prediction algorithm could not be trained')

    # plan all the weekly tasks
    def cron(self):
        print('[INFO] running: ', self.cron)
        schedule.every().sunday.at(self.sunday_trade_time).do(self.etp_job)  # Sydney session opens
        schedule.every().monday.at(self.training_time).do(self.train_job)
        schedule.every().monday.at(self.trading_time).do(self.etp_job)
        schedule.every().tuesday.at(self.training_time).do(self.train_job)
        schedule.every().tuesday.at(self.trading_time).do(self.etp_job)
        schedule.every().wednesday.at(self.training_time).do(self.train_job)
        schedule.every().wednesday.at(self.trading_time).do(self.etp_job)
        schedule.every().thursday.at(self.training_time).do(self.train_job)
        schedule.every().thursday.at(self.trading_time).do(self.etp_job)
        schedule.every().friday.at(self.training_time).do(self.train_job)
        schedule.every().friday.at(self.trading_time).do(self.etp_job)  # New york session closes at 22h


def main():
    # TODO in future releases, the list of insts will be decided by an expert system
    # TODO now, only eur/usd trading is applied
    insts = ['EUR_USD']
    gran = "S5"  # 5 seconds per API request
    comp = "M"
    meas = "raw_price"
    price_field = 'close_price'
    tm = TaskManager(insts, gran, comp, meas, price_field)
    tm.cron()
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()
