from src.utils.allutils import EXP , GetConfig
import os.path as osp
import os
import shutil
from zipfile import ZipFile
import argparse
from loguru import logger 


exp = EXP()

class DownloadData(object):
    # @logger.catch
    def __init__(self,cfg):
        config = GetConfig(cfg)
        path = config.dictconfig.DataSource.path
        exp.trainfile = osp.join(path,config.dictconfig.DataSource.train)
        exp.testfile = osp.join(path,config.dictconfig.DataSource.test)
        exp.rawfloder = config.mkdir_return_path("DataStore",'data',"raw")
    
    # @logger.catch
    def __call__(self):
        if not osp.isfile(exp.trainfile) and  not osp.isfile(exp.testfile):
            raise  FileNotFoundError('File train or test file does not exist')
        
        for source in [exp.trainfile,exp.testfile]:
            basename = osp.basename(source)
            destination = osp.join(exp.rawfloder,basename)
            try:
                shutil.copyfile(source, destination)
                print("File copied successfully.")

            except shutil.SameFileError:
                print("Source and destination represents the same file.")
            except IsADirectoryError:
                print("Destination is a directory.")
            except PermissionError:
                print("Permission denied.")
            except:
                print("Error occurred while copying file.")

        fileExt = r".zip"
        files = [file for file in os.listdir(exp.rawfloder) if file.endswith(fileExt)]
        if not len(files):
            raise Exception(f"Given directory {exp.rawfloder}  there is not found .zip file")

        for file in files:
            with ZipFile(osp.join(exp.rawfloder, file),'r') as zip_ref:
                zip_ref.extractall(exp.rawfloder)

        
# @logger.catch
def download():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg','-c', type=str, default="config/config.yml", help='config yml path')
    opt = parser.parse_args()

    logger.info(f"...Downloading...  args: {opt}")    
    data = DownloadData(**vars(opt))
    logger.info(f"Arguments: {exp.get_attributes()}")

    data()
    logger.info(f"..Downloading complete..  args: {opt}")  
    # except Exception as e:
    #     logger.exception(e)
    #     raise
      
        

if __name__ == '__main__':
    download()