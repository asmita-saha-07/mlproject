import os 
import sys 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from src.utils import evaluate_models, save_object
@dataclass
class ModelTrainConfig:
    model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrain:
    def __init__(self):
        self.model_train_config=ModelTrainConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
      try:
        X_train,y_train,X_test,y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1], test_arr[:,-1])
        models={
            'RandomForestRegressor':RandomForestRegressor(),
            'AdaBoostRegressor':AdaBoostRegressor(),
            'GradientBoostingRegressor':GradientBoostingRegressor(),
            'LinearRegression':LinearRegression(),
            'DecisionTreeRegressor':DecisionTreeRegressor(),
            'XGBRegressor':XGBRegressor(),
            'KNeighborsRegressor':KNeighborsRegressor(),
            'CatBoostRegressor':CatBoostRegressor(verbose=False)
        }

        params={
           'RandomForestRegressor':{'n_estimators':[50,100]},
           'AdaBoostRegressor':{'learning_rate':[0.1,0.5]},
           'GradientBoostingRegressor':{'loss':['squared_error']},
           'LinearRegression':{},
           'DecisionTreeRegressor':{'criterion':['squared_error']},
           'XGBRegressor':{'n_estimators':[50,100]},
           'KNeighborsRegressor':{'n_neighbors':[5,10]},
           'CatBoostRegressor':{'learning_rate':[0.1,0.5]}
        }

        model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)
        best_model_score=max(model_report.values())
        best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model=models[best_model_name]

        if best_model_score<0.6:
            raise Exception("No best model")

        save_object(
            filepath=self.model_train_config.model_file_path,
            obj=best_model
        )
        logging.info("Model saved successfully")
        pred=best_model.predict(X_test)
        r2=r2_score(y_test, pred)
        logging.info("R2 score calculated")
        return r2
      except Exception as e:
         raise CustomException(e,sys)      