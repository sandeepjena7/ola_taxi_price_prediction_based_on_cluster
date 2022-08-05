import argparse
from distutils import filelist
from pathlib import Path
import sys
import os.path as osp
import os
from jsonschema import ValidationError
import joblib
import mlflow
import json
from loguru import logger 
import warnings
warnings.simplefilter("ignore")

logger.add("logs/tools.log", backtrace=True, diagnose=True,rotation="5 MB",retention="900 days")  # onec complete then uncommet this thing
logger.info("-"*100)
ROOT = Path(osp.relpath(Path.cwd())) 
METRIC_CONFIG = ['mse','mae','r2']




@logger.catch
def run(
    id,
    metrics,
    metrics_path,
    model_path
    ):

    # if model_path is None:
    #     if not osp.exists(model_path):
    #         os.makedirs(save_model,exist_ok=True)
    # else:
    #     if not osp.exists(model_path):
    #         os.makedirs(model_path,exist_ok=True)
    id = [i for i in os.listdir("mlruns/0") if id in i][0]
    
    if not osp.isfile(metrics_path):
        logger.error("metrics file path not found")
        raise FileNotFoundError(f"file not found {metrics_path}")
    
    files = f"mlruns/0/{id}"
    if not len([i for i in os.listdir(f"{files}/metrics") if  "kneelocator" in i.lower()]):
        logger.error("you should first successfully run your experiment")
        raise ValueError('you should first successfully run your experiment')
    
    with open(f"mlruns/0/{id}/metrics/kneelocator",'r') as f:
        clusters = int(f.read().split()[1])

   
    models = {}
    best_model_cluster = {}
    best_model_cluster["mlflow_id"] = id
    
    logger.info("Total no of cluster is {}", clusters)
    for clu in range(clusters):
        if not len([i for i in os.listdir(f"{files}/metrics") if metrics in i]) :
            logger.error("you run firs mlfow this id {} successfully",id)
            raise ValueError('you should first successfully run your experiment')
        
        value = {}
        for met in os.listdir(f"{files}/metrics"):
             if (metrics in met) & (str(clu) in met):
                with open(f"mlruns/0/{id}/metrics/{met}", 'r') as f:
                    value[met] = float(f.read().split()[1])

        best_model = min(value, key=value.get)
        best_model_cluster[str(clu)] = best_model.split("_")[1]
        best_model_cluster[f"{metrics}_{clu}"] = min(value.values())

        logged_model = None
        
        for model in os.listdir(f"mlruns/0/{id}/artifacts"):
            if (model.split("_")[0] in best_model.lower()) & (str(clu) in model) :
                logged_model = f"runs:/{id}/{model}"

        if not len(logged_model):
            raise Exception # when deebug understand which types of error may occur then put this error name 

        logger.info("MLflow models is {}",logged_model)
        model = mlflow.pyfunc.load_model(logged_model)
        models[clu] = model
        # put the all the metrics metrics.json file

    kmean = mlflow.pyfunc.load_model(f"runs:/{id}/kmeancluster")
    models["kmean"] = kmean
    save_model(model_path,models)
    logger.info("model is save sucessfuly at {}", model_path)
    save_metrics(metrics_path,best_model_cluster)
    logger.info("deployment metrics and cluster model save at {}",metrics_path)
    



def save_metrics(path, data):
    assert isinstance(data,dict)

    with open(path) as f:
        file = f.read()
    if not len(file):
        with open(path,'w') as f:
            json.dump([data], f, indent=4)
    else:
        with open(path) as f:
            jsonfile = json.load(f)
        jsonfile.append(data)
        with open(path,'w') as f:
            json.dump(jsonfile, f, indent=4)

def save_model(path,data):
    try:
        joblib.dump(data,path)
    except Exception as e:
        logger.error("model was not saved")
        raise IOError (f'Error saving model in disk {e}')


def check_mlflow_id(id):

    
    exp_id = os.listdir("mlruns/0")
    check_id = [i for i in exp_id if id in i]

    if not len(check_id):
        raise ValidationError("give unique run id")
    
    if len(check_id)>2:
        raise Exception("given unique id match more than '1' at mlrun dir")
    logger.info("give id {} is found in mlflow dir sucessfully",id)
    




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, help='mlfow best run id start letter and id length must be > 3 , ex. 3re3, 69ndp93')
    parser.add_argument("-m", "--metrics", type=str, default='mse', choices=METRIC_CONFIG, help="Which metric to choose to best model loaded" )
    parser.add_argument("--metrics-path", type=str, default=ROOT / "report/metrics.json", help='all the mlfow train metrics store')
    parser.add_argument("--model-path", type=str, default=ROOT / "model/model.pkl", help='inference model was saved to file')
    opt = parser.parse_args()
    return opt


def main(opt):
    check_mlflow_id(opt.id)
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
