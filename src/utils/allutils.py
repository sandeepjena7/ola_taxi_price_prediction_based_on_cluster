import os
import yaml
from easydict import EasyDict as edict
import os.path as osp
import errno
import warnings
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import json



class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    @staticmethod
    def yml_params(modelname,**kwargs):
        """if mlflow is same prams pass two model diffrent value he get error to avoid created this function
        """
        return {f"{modelname}_{key}":value for key,value in kwargs.items()}


class _YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super().__init__(cfg_dict)


class GetConfig:
    def __init__(self, filename):
        if not isinstance(filename,str):
            raise ValueError(f"Given {filename} should be a string ")
        
        GetConfig._check_isfile(filename)
        self.dictconfig = _YamlParser( config_file=filename)

    def mkdir_return_path(self,*args):
        """
        You have pass like this
         - demo.yaml 
         
        artifact:
            preproces: dataset
            train: train
        
        mkdir_return_path(artifact,preproces,train)

        createdir - PWD/dataset/train
        return: 
            fullpathPWD/dataset/train
        """
        if not len(args) >= 2:
            raise AttributeError('no of arguments should be greater or equal to 2')
        if not all(isinstance(n, str) for n in args):
            raise TypeError('you should pass all value in str')


        key = list(args)
        dir_path = []

        if key[0] not in  self.dictconfig.keys():
            raise KeyError(f"Give {key[0]} key not present Yaml")

        for dir in key[1:]:
            if not len(dir_path):

                if dir not in self.dictconfig[key[0]].keys():
                    raise KeyError(f"Give {dir} key not present Yaml")
                createdir = GetConfig._mkdir_if_missing(self.dictconfig[key[0]][dir])
                dir_path.append(createdir)
                
            else:
                if dir not in self.dictconfig[key[0]].keys():
                    raise KeyError(f"Give {dir} key not present Yaml")
                    
                dir_path.append(self.dictconfig[key[0]][dir])
                dir_name = os.path.join(*dir_path)
                if dir == key[-1]:
                    return GetConfig._mkdir_if_missing(dir_name)
                GetConfig._mkdir_if_missing(dir_name)

    def get_file_or_directory_path(self,*args):
        """
        You have pass like this
         - demo.yaml 

        artifact:
            preproces: dataset
            train: train

        get_file_or_directory_path(artifact,preproces,train)
        
        return: 
            fullpathPWD/dataset/train
        """
        if not len(args) >= 2:
            raise AttributeError('no of arguments should be greater or equal to 2')
        if not all(isinstance(n, str) for n in args):
            raise TypeError('you should pass all value in str')
        key = list(args)
        dir_path = []

        if key[0] not in  self.dictconfig.keys():
            raise KeyError(f"Given {key[0]} key not present Yaml")
        for dictkey in key[1:]:
            if dictkey not in   self.dictconfig[key[0]].keys():
                raise KeyError(f'Given {dictkey} not present yaml')  
            dir_path.append(self.dictconfig[key[0]][dictkey])

        return osp.join(*dir_path)    
        



    @staticmethod
    def _mkdir_if_missing(dirname):
        if not osp.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        return dirname
    

    @staticmethod
    def _check_isfile(fpath):

        isfile = osp.isfile(fpath)
        if not isfile:
            warnings.warn('No file found at "{}"'.format(fpath))
            raise FileNotFoundError(f"{fpath} is not present System")

class EXP:
    def get_attributes(self):
        return (dict(self.__dict__.items()))




