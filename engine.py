import pandas as pd
import json
from joblib import load
import os
import logging
import logging.config
import yaml
import warnings
warnings.simplefilter("ignore")

class InputNotInRange(Exception):
    def __init__(self,value,message="number not in range"):
        self.value = value
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f" Given value: {self.value} >>> {self.message} "

class log():
    def __init__(self,loggingfilepath):
        self.default_config = os.path.join(os.path.dirname(
            os.path.abspath('__file__')),loggingfilepath)

    def setup_logging(self, default_level=logging.info):
        path = self.default_config
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.captureWarnings(True)
        else:
            logging.basicConfig(level=default_level)


log("config/logging.yml").setup_logging()
logger = logging.getLogger(__name__)
# logger = logging.getLogger("streamlite")



def load_model():
    if not os.path.isfile('model/model.pkl') :
        logger.error("model/model.pkl not found file path")
        raise FileNotFoundError("Run the tools/load.py ")

    models = load("model/model.pkl")
    logging.info("model was load successfully in the memory")
    return models

def manager(userin:dict,models) -> float:
    logger.info("="*30)
    assert isinstance(userin,dict)
    if not os.path.isfile("report/schema_input.json"):
        logger.error("schema_input.json path not found")
        raise FileNotFoundError("Go to notebook generate schema_input.json and put report folder")

    with open("report/schema_input.json",'r') as f:
        schema = json.load(f)
    logger.info("schema was loadded")
    if len(schema) != len(userin):
        logger.error("User given input data and schema length not match")
        raise ValueError("no of feature should be 8")

    for key,value in userin.items():
        if not isinstance(value,int):
            logger.error("user given input type not int")
            raise ValueError(f"Given {key} value should be interger")
        if key not in schema.keys():
            logger.error("user given feature not in present schema file")
            raise AttributeError(f"in schema this feature {key} not in present")
        if (schema[key]['min'] > value) or (schema[key]['max'] < value):
            logger.error('user given value not between min and max')
            raise InputNotInRange(value,f"In {key} not in range between {schema[key]['min']} & {schema[key]['max']} ")

    logger.info("user given input is valid input")
    data = [userin[feature] for feature in schema.keys()]
    df = pd.Series(data).to_frame().T
    
    logger.info("Model start predict")
    clusters = int(models['kmean'].predict(df))
    result = float(models[clusters].predict(df))
    logger.info("model prediction complete")
    logger.info("="*30)
    return result


if __name__ == '__main__':
    data = {"driver_tip" : 3
            ,"distance" : 5
            ,"num_passengers" : 9
            ,"trip_duration" : 4
            ,"payment_method" : 4
            ,"rate_code":76
            ,"extra_charges" : 5
            ,"toll_amount" : 10}

    result = manager(data,load_model())    
    print(result)