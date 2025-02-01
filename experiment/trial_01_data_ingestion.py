
import sys
sys.path.append('/home/western/DS_Projects/website_lead_scores')

from dataclasses import dataclass
from pathlib import Path
import pymongo
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import DATA_INGESTION_CONFIG_FILEPATH
from src.lead_scoring.utils.commons import read_yaml, create_directories

# Load environment variables from .env file
load_dotenv()

@dataclass
class DataIngestionConfig:
    config_data: dict

class ConfigurationManager:
    def __init__(self, data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH):
        """
        Initializes the ConfigurationManager.
        """
        try:
            logger.info(f"Initializing ConfigurationManager with config file: {data_ingestion_config}")
            self.ingestion_config = read_yaml(data_ingestion_config)
            create_directories([self.ingestion_config['artifacts_root']])
            logger.info("Configuration directories created successfully.")
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_config = self.ingestion_config['data_ingestion']
            create_directories([data_config['root_dir']])
            logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
            data_config['mongo_uri'] = os.environ.get('MONGO_URI')
            return DataIngestionConfig(config_data=data_config)
        except Exception as e:
            logger.error(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)
    
    def get_user_name(self):
        try:
            return self.ingestion_config['data_ingestion'].get('get_user_name', 'DefaultUser')
        except Exception as e:
            logger.error(f"Error getting user name from config: {e}")
            raise CustomException(e, sys)

class MongoDBConnection:
    """Handles MongoDB connections with retry logic."""
    def __init__(self, uri, db_name, collection_name, retries=3):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.retries = retries
        self.client = None
        self.db = None
        self.collection = None

    def __enter__(self):
        attempt = 0
        while attempt < self.retries:
            try:
                logger.info(f"Connecting to MongoDB (Attempt {attempt+1}/{self.retries})")
                self.client = pymongo.MongoClient(self.uri)
                self.db = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
                logger.info("Connected to MongoDB Database")
                return self.collection
            except Exception as e:
                logger.error(f"Error connecting to MongoDB: {e}")
                attempt += 1
                time.sleep(2)
        raise CustomException("Failed to connect to MongoDB after multiple attempts.", sys)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig, user_name: str):
        self.config = config
        self.user_name = user_name
        self.mongo_connection = MongoDBConnection(
            config.config_data['mongo_uri'],
            config.config_data['database_name'],
            config.config_data['collection_name']
        )

    def import_data_from_mongodb(self):
        start_time = time.time()
        start_timestamp = datetime.now()
        try:
            logger.info("Starting data ingestion...")
            with self.mongo_connection as collection:
                all_data = self._fetch_all_data(collection)
                if all_data.empty:
                    logger.warning("No data found in MongoDB.")
                    return
                cleaned_data = self._clean_data(all_data)
                output_path = self._save_data(cleaned_data)
                self._save_metadata(start_time, start_timestamp, len(cleaned_data), output_path)
                logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e)

    def _fetch_all_data(self, collection, max_retries=3) -> pd.DataFrame:
        for attempt in range(max_retries):
            try:
                logger.info("Fetching data from MongoDB...")
                batch_size = self.config.config_data.get('batch_size', 1000)
                
                def data_generator():
                    skip_count = 0
                    while True:
                        batch = list(collection.find({}, {'_id': 0}).skip(skip_count).limit(batch_size))
                        if not batch:
                            break
                        yield pd.DataFrame(batch)
                        skip_count += batch_size
                
                all_data = list(data_generator())
                return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            except Exception as e:
                logger.warning(f"Retrying data fetch (Attempt {attempt + 1}/{max_retries})... Error: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        logger.error("Max retries reached. Could not fetch data from MongoDB.")
        raise CustomException("MongoDB data fetch failed.")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            logger.info("Data cleaning completed successfully.")
            return df
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise CustomException(e)

    def _save_data(self, df: pd.DataFrame) -> Path:
        try:
            output_path = Path(self.config.config_data['root_dir']) / "website_visitors.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise CustomException(e)

    def _save_metadata(self, start_time: float, start_timestamp: datetime, total_records: int, output_path: Path):
        try:
            metadata = {
                'start_time': start_timestamp.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': time.time() - start_time,
                "total_records": total_records,
                "data_source": self.config.config_data['collection_name'],
                "ingested_by": self.user_name,
                "output_path": str(output_path)
            }
            metadata_path = Path(self.config.config_data['root_dir']) / "data-ingestion-metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info("Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        user_name = config_manager.get_user_name() # Get user name from config
        data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
        data_ingestion.import_data_from_mongodb()
        logger.info("Data ingestion process completed successfully.")
    except CustomException as e:
        logger.error(f"Error during data ingestion: {e}")
        logger.info("Data ingestion process failed.")