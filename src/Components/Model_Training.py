import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from Src.Exception import CustomException
from Src.Logger import logging
from Src.Utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "Model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_arr, test_arr):
        try: 
            logging.info("Splitting of Training and Test input data")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )
            
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Nieghbour Regressor" : KNeighborsRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "Catboosting Regressor" : CatBoostRegressor(),
                "Adaboost Regressor" : AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get the best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys) #type: ignore
            logging.info(f"Best found model on both training and test dataset")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2score = r2_score(y_test, predicted)
            return r2score
        
        except Exception as e:
            raise CustomException(e, sys) #type: ignore