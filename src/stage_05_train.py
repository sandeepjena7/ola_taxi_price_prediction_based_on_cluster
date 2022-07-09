import lightgbm as lgb
import xgboost as xgb 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from kneed import KneeLocator
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn
from collections import OrderedDict

from src.utils.allutils import EXP , GetConfig,RunBuilder,save_json
from src.utils.metrics import metrics
from src.utils.mlflowlogmodel import dump_model_mlflow
import argparse
import pandas as pd
from loguru import logger 
import warnings

warnings.simplefilter("ignore")

exp = EXP()

class Model(object):
    def __init__(self,cfg):
        config = GetConfig(cfg)
        exp.params = config.dictconfig.params

        exp.train = config.get_file_or_directory_path("DataStore","data","final","train")
        exp.valid = config.get_file_or_directory_path("DataStore","data","final","valid")
        exp.test = config.get_file_or_directory_path("DataStore","data","final","test")

    
    def __call__(self):
             
        train_y = pd.read_csv(exp.train)
        test = pd.read_csv(exp.test)
        valid_y = pd.read_csv(exp.valid)

        train = train_y.drop('total_amount',axis=1)
        valid = valid_y.drop('total_amount',axis=1)

        wcss = []
        for i in range(1,exp.params.cluster_range):
            kmean = KMeans(n_clusters=i,random_state=exp.params.random_state)
            kmean.fit(train)
            wcss.append(kmean.inertia_)
        
        kn = KneeLocator(
                        range(1,exp.params.cluster_range),
                        wcss,
                        curve='convex',
                        direction='decreasing')

          

        cluster = KMeans(n_clusters=kn.knee,random_state=exp.params.random_state)
        cluster.fit(train)
        labels = cluster.labels_
        df_labels = pd.DataFrame(labels, columns=['labels'])
        train_df = pd.concat([train_y,df_labels], axis=1)

        pred_valid = cluster.predict(valid)
        df_valid_labels = pd.DataFrame(pred_valid, columns=['labels'])
        valid_df = pd.concat([valid_y,df_valid_labels], axis=1)

        # with mlflow.start_run() as run:
        #     print(run)
        
        mlflow.log_metric("kneelocator",kn.knee)    
        mlflow.sklearn.log_model(cluster,'kmeancluster')   

        mlflow.log_params(RunBuilder.yml_params("GBR",**exp.params.GradientBoostingRegressor))
        mlflow.log_params(RunBuilder.yml_params("LGR",**exp.params.LGBMRegressor))       
        mlflow.log_params(RunBuilder.yml_params("XGR",**exp.params.XGBRegressor))
    

        GBModel = GradientBoostingRegressor(**exp.params.GradientBoostingRegressor)
        LGModel = lgb.LGBMRegressor(**exp.params.LGBMRegressor)
        XGModel = xgb.XGBRegressor(**exp.params.XGBRegressor)
    
        
        models = [GBModel,LGModel,XGModel]
    

        for cluster in range(kn.knee):
            cluster_train = train_df[train_df['labels']==cluster]
            cluster_x_train = cluster_train.drop(['labels','total_amount'], axis=1)
            cluster_y_train = cluster_train['total_amount']

            cluster_valid = valid_df[valid_df['labels']==cluster]
            cluster_x_valid = cluster_valid.drop(['labels','total_amount'], axis=1)
            cluster_y_valid = cluster_valid['total_amount']

            for model in models:
                model.fit(cluster_x_train, cluster_y_train)
                y_pred = model.predict(cluster_x_valid)

                metrics_ = metrics(
                                    cluster_y_valid,
                                    y_pred,
                                    type(model).__name__,
                                    cluster)

                mlflow.log_metrics(metrics_)
                dump_model_mlflow(model,cluster)

            




# @logger.catch
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg','-c', type=str, default="config/config.yml", help='config yml path')
    opt = parser.parse_args()

    logger.info(f"..Training Starting..  args: {opt}")
    data = Model(**vars(opt))
    logger.info(f"Arguments: {exp.get_attributes()}")

    data()
    logger.info('.....Training Ended.....')

if __name__ == '__main__':
    train()






            



