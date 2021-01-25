OANDA_TOKEN_PATH = r'C:\Users\Jesus Garcia\Desktop\HAL_local_files\config.yaml'
OANDA_TEST_ENV_URL = r'https://api-fxpractice.oanda.com'
PRICE_DATA_FILE = 'price_data.csv'


# NON-MODIFIABLE DATES AND TIMES FOR START AND STOP TRADING AND TRAINING
# Only use 00 minutes, example: 22:00 or 14:00, never 22:30 or 14:10
# If these values are modified, then task_manager parameters need to be updated
TRAINING_TIME = '03:00'  # 3AM Madrid time UTC/GMT +1
TRADING_TIME = '06:00'  # 6AM Madrid time UTC/GMT +1
END_OF_TRADING_DAY = 'Friday'  # Last session of the week is New York, closing at 22h on Friday
FRIDAY_END_TRADE_TIME = '22:00'  # Last session of the week is New York, closing at 22h on Friday
WEEK_START_TRADE_DAY = 'Sunday'
SUNDAY_TRADE_TIME = '22:00'  # trade from 22:00 on Sunday to 03:00 on Monday due to Sidney session opening

# Directories
MODEL_DIR = r'model_data'
SCALER_PATH = r'model_data\scaler.joblib'
MODEL_PATH = r'model_data\lstm_model.h5'
TESTING_TRAIN_CSV = r'testing_files_repo\data_source\EUR_USD_2019_S5_TRAIN_UNITTEST.csv'
TESTING_TEST_CSV = r'testing_files_repo\data_source\EUR_USD_2019_S5_TEST_UNITTEST.csv'
TESTING_SCALER_PATH = r'testing_files_repo\model_testing_templates\scaler.joblib'
TESTING_MODEL_PATH = r'testing_files_repo\model_testing_templates\lstm_model.h5'

# Formats
granularity_to_seconds = {
    'S5': 5,
    'S10': 10,
    'S15': 15,
    'S30': 30,
    'M1': 60,
    'M2': 120,
    'M4': 240,
    'M5': 300,
    'M10': 600,
    'M15': 900,
    'M30': 1800,
    'H1': 3600,
    'H2': 7200,
    'H4': 14400,
    'H6': 21600,
    'H8': 28800,
    'H12': 43200,
    'D': 86400,
    'W': 604800,
    'M': 2592000
}