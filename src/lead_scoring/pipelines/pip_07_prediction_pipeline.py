
import pandas as pd
import numpy as np
import os
import sys
import joblib

from dataclasses import dataclass
from pathlib import Path
import logging

from src.lead_scoring.logger import logger
from src.lead_scoring.constants import PREDICTION_PIPELINE_CONFIG_FILEPATH
from src.lead_scoring.utils.commons import read_yaml
from src.lead_scoring.exception import CustomException


@dataclass
class PredictionConfig:
    model_path: Path
    data_preprocessor: Path


class ConfigurationManager:
    def __init__(self, prediction_pipeline_config = PREDICTION_PIPELINE_CONFIG_FILEPATH):
        try:
          self.prediction_config = read_yaml(prediction_pipeline_config)
        except FileNotFoundError as e:
             logger.error(f"Error: Prediction pipeline config file not found. {e}")
             raise CustomException(e, sys)
    
    def get_prediction_pipeline_config(self) -> PredictionConfig:
        logger.info("Getting prediction pipeline configuration")
        try:
            predict_config = self.prediction_config.prediction_pipeline
            return PredictionConfig(
                model_path = Path(predict_config.model_path),
                data_preprocessor = Path(predict_config.data_preprocessor)
        )
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise CustomException(e, sys)

class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def make_predictions(self, features):                           
        try:
             logging.info("Making Predictions")
             predictions = self._predict_internal(features) 
             return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise CustomException(e, sys)
    
    def _predict_internal(self, features):
        try:
             # Load the trained mode
            logging.info(f"Loading Model from: {self.config.model_path}")
            model_path = Path(self.config.model_path)
            model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model from {self.config.model_path}: {e}")
            raise CustomException(e, sys)
        
        try:
             # Load the data preprocessor
            logging.info(f"Loading Data Preprocessor from: {self.config.data_preprocessor}")
            data_preprocessor_path = Path(self.config.data_preprocessor)
            data_preprocessor = joblib.load(data_preprocessor_path)
        except Exception as e:
            logger.error(f"Error loading preprocessor from {self.config.data_preprocessor}: {e}")
            raise CustomException(e, sys)

        try:
            # Transform features
            logging.info("Transforming features")
            features_transformed = data_preprocessor.transform(features)
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            raise CustomException(e,sys)

        try:
            # Make predictions
            predictions = model.predict(features_transformed)

            # Convert Numpy int64 to python int for JSON serialization
            predictions = [int(pred) for pred in predictions]
        except Exception as e:
             logger.error(f"Error making predictions {e}")
             raise CustomException(e, sys)
        
        return predictions

class InputDataHandler:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


    def get_data_as_df(self):
        try:
            logger.info("Converting data object to a dataframe")
            data_dict = {key: [getattr(self, key)] for key in vars(self)}
            df = pd.DataFrame(data_dict)
            return df
        except Exception as e:
            logger.error(f"Error converting data to dataframe: {str(e)}")
            raise CustomException(e, sys)