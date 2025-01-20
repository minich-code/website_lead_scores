import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")


import pandas as pd

import joblib


from pathlib import Path

from xgboost import XGBClassifier
from typing import Any, Tuple

from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *
from src.lead_scoring.config_entity.config_params import ModelTrainerConfig


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


