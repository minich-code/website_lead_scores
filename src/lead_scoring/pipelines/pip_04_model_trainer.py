

import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

from src.lead_scoring.components.c_04_model_trainer import ModelTrainer
from src.lead_scoring.logger import logger
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.exception import CustomException
from typing import List, Callable

PIPELINE_NAME = "MODEL TRAINER PIPELINE"

class ModelTrainerPipeline:
    def __init__(self):
        pass 


    def run(self):
        try:
            config_manager = ConfigurationManager()
            model_training_config = config_manager.get_model_training_config()
            model_trainer = ModelTrainer(config = model_training_config)

            X_train_transformed, y_train = model_trainer.load_data(
                model_training_config.train_features_path, model_training_config.train_targets_path
            )
            model = model_trainer.train_model(X_train_transformed, y_train) 

        except CustomException as e:
            logger.error(f"Error in {PIPELINE_NAME}: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    try:
        logger.info(f"## ============== {PIPELINE_NAME} ## =================")
        model_trainer_pipeline = ModelTrainerPipeline()
        model_trainer_pipeline.run()
        logger.info(f" # =================={PIPELINE_NAME} Completed Successfully ===========#")

    except Exception as e:
        logger.error(f"Error during {PIPELINE_NAME} pipeline execution: {e}")
        raise CustomException(e, sys.exc_info())