import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")


import pandas as pd
import numpy as np
import joblib
import os

from pathlib import Path
from dataclasses import dataclass 
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple

from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_features_path: Path
    train_targets_path: Path
    model_name: str
    model_params: Dict[str, Any]


class ConfigurationManager:
    def __init__(self, 
                 model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH, 
                 model_params_config: Path = PARAMS_CONFIG_FILEPATH):
        try:
          self.training_config = read_yaml(model_training_config)
        except Exception as e:
           logger.error(f"Error loading model training config file: {str(e)}")
           raise CustomException(e, sys)

        try:
           self.model_params_config = read_yaml(model_params_config)
        except Exception as e:
            logger.error(f"Error loading model parameters config: {str(e)}")
            raise CustomException(e, sys)
        
        try:
           if 'artifacts_root' in self.training_config:
               artifacts_root = self.training_config.artifacts_root
               create_directories([artifacts_root])
           else:
               logger.error("artifacts_root not defined in the configuration.")
               raise CustomException("artifacts_root not defined in the configuration.", sys)
        except Exception as e:
           logger.error(f"Error creating directories: {str(e)}")
           raise CustomException(e, sys)

    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
          trainer_config = self.training_config['model_trainer']
          model_params = self.model_params_config['XGBClassifier_params']

          # Creates all necessary directories
          create_directories([Path(trainer_config.root_dir)])
          
          return ModelTrainerConfig(
              root_dir = Path(trainer_config.root_dir),
              train_features_path = Path(trainer_config.train_features_path),
              train_targets_path = Path(trainer_config.train_targets_path),
              model_name = trainer_config.model_name,
              model_params = model_params

          )
        except Exception as e:
           logger.error(f"Error getting model training config: {str(e)}")
           raise CustomException(e, sys)
    
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @staticmethod
    def load_data(train_features_path: Path, train_targets_path: Path) -> Tuple[Any, pd.DataFrame]:
        """Loads the training data from the given file paths."""
        try:
            with open(train_features_path, 'rb') as f:
                X_train_transformed = joblib.load(f)
            y_train = pd.read_parquet(train_targets_path.as_posix())

            logger.info("Training data loaded successfully")
            return X_train_transformed, y_train

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
    def train_model(self, X_train_transformed, y_train):
        try:
            if not self.config.model_params:
                raise ValueError("Model parameters are empty.")
            xgb_model = XGBClassifier(**self.config.model_params)
            xgb_model.fit(X_train_transformed, y_train)

            # Save the model
            model_path = Path(self.config.root_dir) / self.config.model_name
            joblib.dump(xgb_model, model_path) 
            logger.info(f"Model trained and saved at: {model_path}") 

            return xgb_model 
        except Exception as e:
            logger.error(f"Error training model: {str(e)}") 
            raise CustomException(e, sys)



if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        model_trainer = ModelTrainer(config = model_training_config)

        X_train_transformed, y_train = model_trainer.load_data(
            model_training_config.train_features_path, model_training_config.train_targets_path
        )
        model = model_trainer.train_model(X_train_transformed, y_train) 

        logger.info("Model Training Completed Successfully")
    except CustomException as e:
        logger.error(f"Error in model training: {str(e)}")
        sys.exit(1)