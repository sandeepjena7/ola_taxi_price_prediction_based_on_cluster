from src.utils.allutils import EXP , GetConfig
import argparse
import pandas as pd
import shutil
from loguru import logger 
from sklearn.model_selection import train_test_split

exp = EXP()


class RemoveNull(object):
    def __init__(self,cfg):
        config = GetConfig(cfg)
        exp.trainpath = config.get_file_or_directory_path('DataStore','data','preprocess','train')

    def __call__(self):

        train = pd.read_csv(exp.trainpath)        
        cat_cols = ["driver_tip", "toll_amount", "extra_charges", "total_amount"]
        num_cols = [c for c in train.columns if c not in cat_cols]

        for c in num_cols :
            train[c] = train[c].fillna((train[c].mean()))

        for c in cat_cols :
            train[c] = pd.to_numeric(train[c], errors='coerce')

        train = train.dropna()

        train.to_csv(exp.trainpath,index=False)


# @logger.catch
def remove_null_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg','-c', type=str, default="config/config.yml", help='config yml path')
    opt = parser.parse_args()

    logger.info(f"..Starting remove Null values..  args: {opt}")
    data = RemoveNull(**vars(opt))
    logger.info(f"Arguments: {exp.get_attributes()}")

    data()
    logger.info('.....Dataset was successfully Remove null value.....')

if __name__ == '__main__':
    remove_null_data()