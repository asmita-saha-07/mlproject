import os
import sys 
import dill
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(filepath, obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train, y_train)
            y_pred=model.predict(X_test)
            r2=r2_score(y_test, y_pred)
            report[list(models.keys())[i]]=r2
        return report

    except Exception as e:
        raise CustomException(e,sys)