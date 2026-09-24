import os
import sys 
import dill
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(filepath, obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs=GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred=model.predict(X_test)
            r2=r2_score(y_test, y_pred)
            report[list(models.keys())[i]]=r2
        return report

    except Exception as e:
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        with open(filepath,'rb') as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e,sys)