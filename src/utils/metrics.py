from sklearn.metrics import mean_squared_error,mean_absolute_error ,r2_score

def metrics(y_true,y_pred,model_name,cluster):
    mse = mean_squared_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)    
    metric = {f"mse_{model_name}_{cluster}": mse,
              f"mae_{model_name}_{cluster}": mae,
              f"r2_{model_name}_{cluster}": r2}
    return metric