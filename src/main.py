from loguru import logger 
import mlflow
import logging
import warnings
warnings.simplefilter("ignore")
logger.add("logs/src.log", backtrace=True, diagnose=True,rotation="50 MB",retention="100 days")  # onec complete then uncommet this thing
logger.info(">>>."*100)

@logger.catch
def main():
    with mlflow.start_run() as run:
        # mlflow.run(".",'get_data', use_conda=False)
        # mlflow.run(".", "clean_data", use_conda=False)
        # mlflow.run(".", "remove_null", use_conda=False)
        # mlflow.run(".", "split_data", use_conda=False)
        mlflow.run(".", "train", use_conda=False)


if __name__ == '__main__':
    main()
    