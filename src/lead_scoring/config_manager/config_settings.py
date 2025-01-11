
import pymongo
import os
import sys

from dotenv import load_dotenv
from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import DATA_INGESTION_CONFIG_FILEPATH
from src.lead_scoring.utils.commons import read_yaml, create_directories
from src.lead_scoring.config_entity.config_params import DataIngestionConfig

# Load environment variables from .env file
load_dotenv()


class ConfigurationManager:
    def __init__(self, data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH):
        """
        Initializes the ConfigurationManager.

        Args:
            data_ingestion_config (str): Path to the data ingestion configuration file.
               
        """
        try:
            logger.info(f"Initializing ConfigurationManager with config file: {data_ingestion_config}")
            self.ingestion_config = read_yaml(data_ingestion_config)
            create_directories([self.ingestion_config.artifacts_root])
            logger.info("Configuration directories created successfully.")
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Returns the data ingestion configuration object."""
        try:
            data_config = self.ingestion_config.data_ingestion
            create_directories([data_config['root_dir']])
            logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
            data_config['mongo_uri'] = os.environ.get('MONGO_URI')
            return DataIngestionConfig(config_data=data_config)
        except Exception as e:
            logger.error(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)
