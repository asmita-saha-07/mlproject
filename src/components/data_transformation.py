import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline=Pipeline([
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())]
            )

            cat_pipeline=Pipeline([
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("onehotencoder",OneHotEncoder(sparse_output=False)),
                ("scaler",StandardScaler())]
            )

            logging.info(f"Numerical columns : {numerical_columns}")
            logging.info(f"Categorical columns : {categorical_columns}")

            preprocessor=ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and Test data read successfully")

            preprocessing_obj=self.get_transformer_object()

            target_column_name="math_score"

            input_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_train_df=train_df[target_column_name]

            input_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_test_df=test_df[target_column_name]

            logging.info("Extracted train and test datframes")

            input_train_arr=preprocessing_obj.fit_transform(input_train_df)
            input_test_arr=preprocessing_obj.transform(input_test_df)

            train_arr=np.c_[input_train_arr, np.array(target_train_df)]
            test_arr=np.c_[input_test_arr, np.array(target_test_df)]

            save_object(
                filepath=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj
            )
            logging.info('Saved preprocessor object as pkl file')
            
            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

