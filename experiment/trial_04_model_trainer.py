# import sys
# sys.path.append("/home/western/DS_Projects/website_lead_scores")


# import pandas as pd
# import numpy as np
# import joblib
# import os

# from pathlib import Path
# from dataclasses import dataclass 
# from xgboost import XGBClassifier
# from typing import Dict, Any, Tuple

# from src.lead_scoring.exception import CustomException
# from src.lead_scoring.logger import logger
# from src.lead_scoring.constants import *
# from src.lead_scoring.utils.commons import *

# # Wandb
# import wandb
# wandb.require("core")
# #Moved wandb init to train_model to ensure each training run is logged separately
# #wandb.init(
# #    project = 'leads-scoring'
# #)


# @dataclass
# class ModelTrainerConfig:
#     root_dir: Path
#     train_features_path: Path
#     train_targets_path: Path
#     model_name: str
#     model_params: Dict[str, Any]
#     project_name: str #Added Project Name
#     val_features_path: Path
#     val_targets_path: Path



# class ConfigurationManager:
#     def __init__(self, 
#                  model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH, 
#                  model_params_config: Path = PARAMS_CONFIG_FILEPATH,
#                  hyperparameter_config: Path =HYPERPARAMETER_SEARCH_CONFIG_FILEPATH):
#         try:
#           self.training_config = read_yaml(model_training_config)
#         except Exception as e:
#            logger.error(f"Error loading model training config file: {str(e)}")
#            raise CustomException(e, sys)

#         try:
#            self.model_params_config = read_yaml(model_params_config)
#            self.wandb_config = read_yaml(hyperparameter_config)
#         except Exception as e:
#             logger.error(f"Error loading model parameters config: {str(e)}")
#             raise CustomException(e, sys)
        
#         try:
#            if 'artifacts_root' in self.training_config:
#                artifacts_root = self.training_config.artifacts_root
#                create_directories([artifacts_root])
#            else:
#                logger.error("artifacts_root not defined in the configuration.")
#                raise CustomException("artifacts_root not defined in the configuration.", sys)
#         except Exception as e:
#            logger.error(f"Error creating directories: {str(e)}")
#            raise CustomException(e, sys)

#     def get_model_training_config(self) -> ModelTrainerConfig:
#         logger.info("Getting model training configuration")
#         try:
#           trainer_config = self.training_config['model_trainer']
#           model_params = self.model_params_config['XGBClassifier_params']

#           # Creates all necessary directories
#           create_directories([Path(trainer_config.root_dir)])
          
#           return ModelTrainerConfig(
#               root_dir = Path(trainer_config.root_dir),
#               train_features_path = Path(trainer_config.train_features_path),
#               train_targets_path = Path(trainer_config.train_targets_path),
#               model_name = trainer_config.model_name,
#               model_params = model_params,
#               project_name = trainer_config.project_name, # Added project_name to config
#               val_features_path = Path(trainer_config.val_features_path),
#               val_targets_path = Path(trainer_config.val_targets_path)  # Added val_targets_path to config
#           )
#         except Exception as e:
#            logger.error(f"Error getting model training config: {str(e)}")
#            raise CustomException(e, sys)
    
# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config

#     @staticmethod
#     def load_data(train_features_path: Path, train_targets_path: Path, val_features_path: Path, val_targets_path: Path ) -> Tuple[Any, pd.DataFrame]:
#         """Loads the training data from the given file paths."""
#         try:
#             with open(train_features_path, 'rb') as f:
#                 X_train_transformed = joblib.load(f)
#             y_train = pd.read_parquet(train_targets_path.as_posix())

#             with open(val_features_path, 'rb') as f:
#                 X_val_transformed = joblib.load(f)
#             y_val = pd.read_parquet(val_targets_path.as_posix)  # Added val_targets_path to config

#             logger.info("Training data loaded successfully")
#             return X_train_transformed, y_train, y_val, X_val_transformed

#         except FileNotFoundError as fnf_error:
#             logger.error(f"File not found: {str(fnf_error)}")
#         except Exception as e:
#             logger.error(f"Unexpected error loading data: {str(e)}")
#     def train_model(self, X_train_transformed, y_train):
#         try:
#             if not self.config.model_params:
#                 raise ValueError("Model parameters are empty.")
            
#             run = wandb.init(project=self.config.project_name, #Use configured project name
#                              config = {**self.config.model_params
                                 
#                              })
#             xgb_model = XGBClassifier(**self.config.model_params)

#             xgb_model.fit(X_train_transformed, y_train)


#             # Save model 
#             model_path = Path(self.config.root_dir) / self.config.model_name
#             joblib.dump(xgb_model, model_path) 
#             logger.info(f"Model trained and saved at: {model_path}") 

            

#             artifact = wandb.Artifact('model', type='model')
#             artifact.add_file(model_path)
#             run.log_artifact(artifact) #Correctly log the artifact 


#             run.finish()

#             return xgb_model 
#         except Exception as e:
#             logger.error(f"Error training model: {str(e)}") 
#             raise CustomException(e, sys)
        



# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         model_training_config = config_manager.get_model_training_config()
#         model_trainer = ModelTrainer(config = model_training_config)

#         X_train_transformed, y_train = model_trainer.load_data(
#             model_training_config.train_features_path, model_training_config.train_targets_path
#         )
#         model = model_trainer.train_model(X_train_transformed, y_train) 

#         logger.info("Model Training Completed Successfully")
#     except CustomException as e:
#         logger.error(f"Error in model training: {str(e)}")
#         wandb.finish() # Ensure the run is ended incase of an exception
#         sys.exit(1)

# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         model_training_config = config_manager.get_model_training_config()
#         model_trainer = ModelTrainer(config = model_training_config)

#         X_train_transformed, y_train = model_trainer.load_data(
#             model_training_config.train_features_path, model_training_config.train_targets_path
#         )
#         model = model_trainer.train_model(X_train_transformed, y_train) 

#         logger.info("Model Training Completed Successfully")

#         trigger = input("Enter 'train' for model training or 'sweep' for hyperparameter search: ")
        
#         if trigger.lower() == 'train':
#             model_trainer = ModelTrainer(config = model_training_config)
#             model_trainer.train_model(X_train_transformed, y_train)

#         elif trigger.lower() =='sweep':
#             # Hyperparameter search code goes here
#             sweep_configuration = config.wandb_config

#             # Start the hyperparameter sweep 
#             sweep_id = wandb.sweep(

#                 sweep = sweep_configuration,
#                 project = model_training_config.project_name,
#                 entity = 'leads_score' # replace with your wandb username
#             )
#             # Define function to run each sweep trial 
#             def train_and_evaluate():
#                 with wandb.init():
#                     config = wandb.config

#                     xgb_model = XGBClassifier(**self.config.model_params)


#                     xgb_model.fit(X_train_transformed, y_train)
#                     y_val_pred = xgb_model.predict(X_val_transformed)
#                     val_f1 = f1_score(y_val_pred, y_val, average='macro')
#                     wandb.log({"validation_f1_score": val_f1})

#             # Run the sweep 
#             wandb.agent(sweep_id, function=train_and_evaluate, count=15)

#         else:
#             print("Invalid input. Please enter 'train' or'sweep'.")

#     except Exception as e:
#         logger.error(f"Error in model training: {str(e)}")
#         wandb.finish() # Ensure the run is ended in case of an exception
#         sys.exit(1)

import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Local Modules
from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *

# Wandb
import wandb
wandb.require("core")


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_features_path: Path
    train_targets_path: Path
    model_name: str
    model_params: Dict[str, Any]
    project_name: str
    val_features_path: Path
    val_targets_path: Path


class ConfigurationManager:
    def __init__(
        self,
        model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH,
        model_params_config: Path = PARAMS_CONFIG_FILEPATH,
        hyperparameter_config: Path = HYPERPARAMETER_SEARCH_CONFIG_FILEPATH,
    ):
        try:
            self.training_config = read_yaml(model_training_config)
        except Exception as e:
            logger.error(f"Error loading model training config file: {str(e)}")
            raise CustomException(e, sys)

        try:
            self.model_params_config = read_yaml(model_params_config)
            self.wandb_config = read_yaml(hyperparameter_config)
        except Exception as e:
            logger.error(f"Error loading model parameters config: {str(e)}")
            raise CustomException(e, sys)

        try:
            if "artifacts_root" in self.training_config:
                artifacts_root = self.training_config.artifacts_root
                create_directories([artifacts_root])
            else:
                logger.error("artifacts_root not defined in the configuration.")
                raise CustomException(
                    "artifacts_root not defined in the configuration.", sys
                )
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise CustomException(e, sys)

    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
            trainer_config = self.training_config["model_trainer"]
            model_params = self.model_params_config["XGBClassifier_params"]

            # Creates all necessary directories
            create_directories([trainer_config.root_dir])

            return ModelTrainerConfig(
                root_dir=Path(trainer_config.root_dir),
                train_features_path=Path(trainer_config.train_features_path),
                train_targets_path=Path(trainer_config.train_targets_path),
                model_name=trainer_config.model_name,
                model_params=model_params,
                project_name=trainer_config.project_name,
                val_features_path=Path(trainer_config.val_features_path),
                val_targets_path=Path(trainer_config.val_targets_path),
            )
        except Exception as e:
            logger.error(f"Error getting model training config: {str(e)}")
            raise CustomException(e, sys)


class DataManager:
    @staticmethod
    def load_data(
        train_features_path: Path,
        train_targets_path: Path,
        val_features_path: Path,
        val_targets_path: Path,
    ) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Any]:
        """Loads the training and validation data from the given file paths."""
        try:
            with open(train_features_path, "rb") as f:
                X_train_transformed = joblib.load(f)
            y_train = pd.read_parquet(train_targets_path.as_posix())

            with open(val_features_path, "rb") as f:
                X_val_transformed = joblib.load(f)
            y_val = pd.read_parquet(val_targets_path.as_posix())

            logger.info("Training and validation data loaded successfully")
            return X_train_transformed, y_train, y_val, X_val_transformed

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise CustomException(e, sys)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _train(self, X_train_transformed, y_train):
        try:
            if not self.config.model_params:
                raise ValueError("Model parameters are empty.")

            run = wandb.init(
                project=self.config.project_name,
                config={**self.config.model_params},
            )
            xgb_model = XGBClassifier(**self.config.model_params)
            xgb_model.fit(X_train_transformed, y_train)

            # Save model
            model_path = Path(self.config.root_dir) / self.config.model_name
            joblib.dump(xgb_model, model_path)
            logger.info(f"Model trained and saved at: {model_path}")

            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(model_path)
            run.log_artifact(artifact)

            run.finish()

            return xgb_model
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)

    def _train_and_evaluate(
        self, X_train_transformed, y_train, X_val_transformed, y_val
    ):
        with wandb.init() as run:
            config = wandb.config
            xgb_model = XGBClassifier(**config)
            xgb_model.fit(X_train_transformed, y_train)
            y_val_pred = xgb_model.predict(X_val_transformed)
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            wandb.log({"validation_f1_score": val_f1})

    def train_with_sweep(
      self,
      X_train_transformed,
      y_train,
      X_val_transformed,
      y_val,
      sweep_configuration, #Take sweep configuration as a parameter
    ):

        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project=self.config.project_name
        )
        wandb.agent(
            sweep_id,
            function=lambda: self._train_and_evaluate(
                X_train_transformed, y_train, X_val_transformed, y_val
            ),
            count=1,
        )
if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        data_manager = DataManager()
        X_train_transformed, y_train, y_val, X_val_transformed = data_manager.load_data(
            model_training_config.train_features_path,
            model_training_config.train_targets_path,
            model_training_config.val_features_path,
            model_training_config.val_targets_path,
        )
        model_trainer = ModelTrainer(config=model_training_config)
        trigger = input(
            "Enter 'train' for model training or 'sweep' for hyperparameter search: "
        )

        if trigger.lower() == "train":
            model = model_trainer._train(X_train_transformed, y_train)
            logger.info("Model Training Completed Successfully")
        elif trigger.lower() == "sweep":
            model_trainer.train_with_sweep(
                X_train_transformed,
                y_train,
                X_val_transformed,
                y_val,
                config_manager.wandb_config.get("sweep", {}), #Pass sweep config
            )
        else:
            print("Invalid input. Please enter 'train' or 'sweep'.")

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        wandb.finish()  # Ensure the run is ended in case of an exception
        sys.exit(1)

