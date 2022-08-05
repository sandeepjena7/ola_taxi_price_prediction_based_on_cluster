import xgboost
import sklearn
import lightgbm
import mlflow

def dump_model_mlflow(model,cluster):

    assert isinstance(model,(lightgbm.sklearn.LGBMRegressor,
                                sklearn.ensemble._gb.GradientBoostingRegressor,
                                xgboost.sklearn.XGBRegressor))

    name = type(model).__name__.lower()[:3]
    if name in 'xgb':
        mlflow.xgboost.log_model(model,f"{name}_{cluster}")
    
    elif name in 'lgb':
        mlflow.lightgbm.log_model(model,f"{name}_{cluster}")

    elif name in  'gra':
        mlflow.sklearn.log_model(model,f"{name}_{cluster}")