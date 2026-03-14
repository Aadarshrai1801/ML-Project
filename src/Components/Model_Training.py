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
                "Catboosting Regressor" : CatBoostRegressor(verbose=0),
                "Adaboost Regressor" : AdaBoostRegressor()
            }
            
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200, 500],
                    "subsample": [0.8, 1.0],
                    "max_depth": [3, 5, 7]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "K-Nieghbour Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"],
                    "p": [1, 2]
                },

                "XGB Regressor": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [100, 200, 500],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },

                "Catboosting Regressor": {
                    "iterations": [100, 200, 500],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "depth": [4, 6, 10],
                    "l2_leaf_reg": [1, 3, 5]
                },

                "Adaboost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }
            }
            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get the best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            
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