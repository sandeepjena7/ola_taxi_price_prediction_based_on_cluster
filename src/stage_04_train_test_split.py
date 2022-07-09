from src.utils.allutils import EXP , GetConfig
import argparse
import pandas as pd
import shutil
from loguru import logger 
from sklearn.model_selection import train_test_split

exp = EXP()


class SplitData(object):
    def __init__(self,cfg):
        config = GetConfig(cfg)
        exp.trainpath = config.get_file_or_directory_path('DataStore','data','preprocess','train')
        exp.testpath = config.get_file_or_directory_path('DataStore','data','preprocess','test')

        config.mkdir_return_path('DataStore','data','final')
        exp.splittrainpath = config.get_file_or_directory_path('DataStore','data','final','train')
        exp.splitvalidpath = config.get_file_or_directory_path('DataStore','data','final','valid')
        exp.finaltest = config.get_file_or_directory_path('DataStore','data','final','test')
        exp.test_size = config.dictconfig.params.test_size
        exp.state = config.dictconfig.params.random_state

    def __call__(self):
        train = pd.read_csv(exp.trainpath)
        sptrain,spvalid = train_test_split(train,test_size=exp.test_size,random_state=exp.state)

        shutil.copyfile(exp.testpath,exp.finaltest)
        sptrain.to_csv(exp.splittrainpath,index=False)
        spvalid.to_csv(exp.splitvalidpath,index=False)



# @logger.catch
def split_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg','-c', type=str, default="config/config.yml", help='config yml path')
    opt = parser.parse_args()

    logger.info(f"..Starting Split Data..  args: {opt}")
    data = SplitData(**vars(opt))
    logger.info(f"Arguments: {exp.get_attributes()}")

    data()
    logger.info('.....Dataset was successfully Split.....')

if __name__ == '__main__':
    split_data()
