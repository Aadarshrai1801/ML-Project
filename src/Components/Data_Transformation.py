import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Src.Exception import CustomException
from Src.Logger import logging
from Src.Utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("Artifacts", "Preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            num_features = ["writing_score", "reading_score"]
            cat_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"                        
            ]
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")), #type: ignore
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical Columns: {num_features}")
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {cat_features}")
            
            preprocessor = ColumnTransformer(
                [
                    ("Numerical Pipeline", num_pipeline, num_features),
                    ("Categorical Pipeline", cat_pipeline, cat_features)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys) #type: ignore
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read the Train and Test data")
            logging.info("Obtaining preprocessor object")
            
            preprocessor_obj = self.get_data_transformer_obj()
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1) #type: ignore
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1) #type: ignore
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(input_feature_test_df)
            ]
            
            logging.info(f"Saved Preprocessing object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e, sys) #type: ignore
            