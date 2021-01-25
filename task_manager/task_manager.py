from data_extraction import data_extraction as dext
from data_transformation import data_transformation as dtrans
from data_loading import data_loading as dload
from data_prediction import data_prediction as dpred
import time
from datetime import date, datetime, timedelta
import schedule  # pip install schedule
import config_data.constants as conf
import pandas as pd
from txus_library import txuslib


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
    def data_extract(self) -> dict:
        de = dext.DataExtraction(insts=self.insts, gran=self.gran, comp=self.comp)
        try:
            inst_dict = de.inst_dict_constructor()
            return inst_dict
        except DataExtractionError:
            print('[ERROR] Data could not be extracted')

    # TODO return transformed price
    # saves extracted and transformed data in the Influx DB database
    def data_transform(self, inst_dict: dict):
        dt = dtrans.DataTransformation(inst_dict=inst_dict, meas=self.meas, comp=self.comp)
        try:
            dt.set_price_data_db()
            trans_price = 'pending'
            return trans_price
        except DataTransformationError:
            print('[ERROR] Data could not be transformed')

    # based on the last 60 values of data, predicts the next price value
    def data_predict(self, pred_df: pd.DataFrame, model_path=conf.MODEL_PATH, scaler_path=conf.SCALER_PATH) -> float:
        dp = dpred.DataPrediction(insts=self.insts, meas=self.meas, comp=self.comp, field=self.field)
        try:
            pred_price = dp.dynamic_predictor(pred_df=pred_df, model_path=model_path, scaler_path=scaler_path)
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
    def etp_job(self, is_etl_testing=False, is_etlp_testing=False) -> dict:  # job: Extract, Transform and Predict data
        print('[INFO] running: ', self.etp_job)
        print('[INFO] running: %s at %s' % (self.etp_job, datetime.now().strftime("%Y-%M-%D_%H:%M")))
        pred_df = pd.DataFrame()
        ext_df = pd.DataFrame()
        ext_dict = {}
        pred_dict = {}
        data_ext_counter = 0
        while True:
            ext_inst_dict = self.data_extract()
            trans_price = self.data_transform(inst_dict=ext_inst_dict)
            aux_ext_df = txuslib.df_from_inst_dict(inst_dict=ext_inst_dict, field=self.field)
            ext_df = ext_df.append(aux_ext_df, ignore_index=True)
            if data_ext_counter >= 60:
                for inst in self.insts:  # perform prediction for each instrument
                    # TODO: now data_predict can only predict one instrument, not a dict of instruments
                    # TODO: however, the rest of ETPL is prepared to manage a dictionary of instruments
                    if is_etlp_testing:
                        pred_price = self.data_predict(
                            pred_df=ext_df[inst].iloc[-60:],
                            model_path=conf.TESTING_MODEL_PATH,
                            scaler_path=conf.TESTING_SCALER_PATH)
                    else:
                        pred_price = self.data_predict(ext_df[inst].iloc[-60:])
                    pred_dict[inst] = pred_price
            aux_pred_df = pd.DataFrame.from_dict(pred_dict)
            pred_df = pred_df.append(aux_pred_df, ignore_index=True)
            print('---------------------------------------------------------------------------------------------------')
            print('[INFO] extracted data:')
            print(ext_df.iloc[-1:])
            print('---------------------------------------------------------------------------------------------------')
            print('[INFO] predicted data:')
            print(pred_df.iloc[-1:])
            print('---------------------------------------------------------------------------------------------------')
            if self.is_training_time():
                break
            if self.is_end_of_trading_time():
                break
            if is_etl_testing:
                break
            if is_etlp_testing and data_ext_counter > 65:
                break

            data_ext_counter += 1
            time.sleep(5)
        return {'current_date_time': datetime.now().strftime("%Y-%M-%D_%H:%M"),
                'extracted_data': ext_df,
                'predicted_data': pred_df}

    # train the algorithm with data from the previous day
    # should finish the training before the planned daily start of trading
    def train_job(self):
        print('[INFO] running: %s at %s' % (self.train_job, datetime.now().strftime("%Y-%M-%D_%H:%M")))
        try:
            dp = dpred.DataPrediction(insts=self.insts, comp=self.comp, meas=self.meas, field=self.field)
            dp.daily_train_test()
        except AlgorithmTrainingError:
            print('[ERROR] The prediction algorithm could not be trained')

    # plan all the weekly tasks
    def cron(self):
        print('[INFO] running: ', self.cron)
        schedule.every().sunday.at(self.sunday_trade_time).do(self.etp_job).tag('trade')  # Sydney session opens
        schedule.every().monday.at(self.training_time).do(self.train_job).tag('train')
        schedule.every().monday.at(self.trading_time).do(self.etp_job).tag('trade')
        schedule.every().tuesday.at(self.training_time).do(self.train_job).tag('train')
        schedule.every().tuesday.at(self.trading_time).do(self.etp_job).tag('trade')
        schedule.every().wednesday.at(self.training_time).do(self.train_job).tag('train')
        schedule.every().wednesday.at(self.trading_time).do(self.etp_job).tag('trade')
        schedule.every().thursday.at(self.training_time).do(self.train_job).tag('train')
        schedule.every().thursday.at(self.trading_time).do(self.etp_job).tag('trade')
        schedule.every().friday.at(self.training_time).do(self.train_job).tag('train')
        schedule.every().friday.at(self.trading_time).do(self.etp_job).tag('trade')   # New york session closes at 22h
        print('[INFO] List of scheduled jobs: ')
        for job in schedule.jobs:
            print('[INFO]', job)
        print('[INFO] Next run is on: ', schedule.next_run())
        if conf.TRAINING_TIME in str(schedule.next_run()):
            print('[INFO] Populating data for the next training job')
            self.etp_job()


def main():
    # TODO in future releases, the list of insts will be decided by an expert system
    # TODO now, only eur/usd trading is applied
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    insts = ['EUR_USD', 'EUR_JPY', 'EUR_AUD', 'EUR_CHF', 'EUR_GBP']
    gran = "S5"  # 5 seconds per API request
    comp = "M"
    meas = "raw_price"
    field = 'close'
    tm = TaskManager(insts=insts, gran=gran, comp=comp, meas=meas, field=field)
    tm.cron()
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()
