from src.utils.allutils import EXP , GetConfig
import argparse
import pandas as pd
from loguru import logger 
exp = EXP()


class CleanUpData(object):
    def __init__(self,cfg):
        config = GetConfig(cfg)
        exp.rawtrainpath = config.get_file_or_directory_path("DataStore","data","raw","train")
        exp.rawtestpath = config.get_file_or_directory_path("DataStore","data","raw","test")
        config.mkdir_return_path("DataStore","data","preprocess")
        exp.pretrainpath = config.get_file_or_directory_path("DataStore","data","preprocess","train")
        exp.pretestpath = config.get_file_or_directory_path("DataStore","data","preprocess","test")

    # @logger.catch
    def __call__(self):
        train = pd.read_csv(exp.rawtrainpath)
        test = pd.read_csv(exp.rawtestpath)

        rename = {'vendor+AF8-id': "vendor_id",
                        'pickup+AF8-loc': "pickup_loc",
                        'drop+AF8-loc' : "drop_loc",
                        'driver+AF8-tip': "driver_tip", 
                        'mta+AF8-tax' : "mta_tax",
                        'pickup+AF8-time' : "pickup_time",
                        'drop+AF8-time' : "drop_time", 
                        'num+AF8-passengers' : "num_passengers",
                        'toll+AF8-amount' : "toll_amount",
                        'payment+AF8-method' : "payment_method",
                        'rate+AF8-code' : "rate_code",
                        'stored+AF8-flag' : "stored_flag",
                        'extra+AF8-charges' : "extra_charges",
                        'improvement+AF8-charge' : "improvement_charge",
                        'total+AF8-amount': "total_amount"
                        }
        train = train.rename(columns=rename)
        train = self._trip_duration(train)
        test = self._trip_duration(test)
        
        train = self._drop_columns(train)
        test = self._drop_columns(test)

        train.to_csv(exp.pretrainpath,index=False)
        test.to_csv(exp.pretestpath,index=False)
    

    def _trip_duration(self,df):
        df_ = df.copy()
        df_['pickup_time']=pd.to_datetime(df_['pickup_time'])
        df_['drop_time']=pd.to_datetime(df_['drop_time'])
        df_['trip_duration']=df_['drop_time']-df_['pickup_time']
        df_['trip_duration'] = df_['trip_duration'].astype('timedelta64[s]')
        return df_


    def _drop_columns(self, df):
        df_ = df.copy()
        df_ = df_.drop(['pickup_time', "drop_time",'ID', "vendor_id", "drop_loc", "pickup_loc", "stored_flag", "mta_tax", "improvement_charge" ], axis = 1)
        return df_



# @logger.catch
def clean_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg','-c', type=str, default="config/config.yml", help='config yml path')
    opt = parser.parse_args()

    logger.info(f"..Starting Clean the Data set..  args: {opt}")
    data = CleanUpData(**vars(opt))
    logger.info(f"Arguments: {exp.get_attributes()}")

    data()
    logger.info('.....Dataset was successfully cleanup.....')

if __name__ == '__main__':
    clean_data()


