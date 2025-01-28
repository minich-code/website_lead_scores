# import sys
# sys.path.append('/home/western/DS_Projects/website_lead_scores')

# from dataclasses import dataclass
# from pathlib import Path
# import pymongo
# import pandas as pd
# import numpy as np
# import os
# import json
# import time
# import logging
# from datetime import datetime
# from dotenv import load_dotenv
# from pymongo import MongoClient
# from src.lead_scoring.exception import CustomException
# from src.lead_scoring.logger import logger
# from src.lead_scoring.constants import DATA_INGESTION_CONFIG_FILEPATH
# from src.lead_scoring.utils.commons import read_yaml, create_directories

# # Load environment variables from .env file
# load_dotenv()

# @dataclass
# class DataIngestionConfig:
#     config_data: dict

# class ConfigurationManager:
#     def __init__(self, data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH):
#         """
#         Initializes the ConfigurationManager.

#         Args:
#             data_ingestion_config (str): Path to the data ingestion configuration file.
#         """
#         try:
#             logger.info(f"Initializing ConfigurationManager with config file: {data_ingestion_config}")
#             self.ingestion_config = read_yaml(data_ingestion_config)
#             create_directories([self.ingestion_config.artifacts_root])
#             logger.info("Configuration directories created successfully.")
#         except Exception as e:
#             logger.error(f"Error initializing ConfigurationManager: {e}")
#             raise CustomException(e, sys)
    
#     def get_data_ingestion_config(self) -> DataIngestionConfig:
#         try:
#             data_config = self.ingestion_config.data_ingestion
#             create_directories([data_config['root_dir']])
#             logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
#             data_config['mongo_uri'] = os.environ.get('MONGO_URI')
#             return DataIngestionConfig(config_data=data_config)
#         except Exception as e:
#             logger.error(f"Error loading data ingestion configuration: {e}")
#             raise CustomException(e, sys)
        
# class MongoDBConnection:
#     """Handles MongoDB connections."""
#     def __init__(self, uri, db_name, collection_name):
#         """
#         Initializes MongoDBConnection with the given URI, database name, and collection name.

#         Args:
#             uri (str): MongoDB URI.
#             db_name (str): Name of the MongoDB database.
#             collection_name (str): Name of the MongoDB collection.
#         """
#         self.uri = uri
#         self.db_name = db_name
#         self.collection_name = collection_name
#         self.client = None
#         self.db = None
#         self.collection = None


#     def __enter__(self):
#         try:
#             logger.info(f"Connecting to MongoDB Database")
#             self.client = pymongo.MongoClient(self.uri)
#             self.db = self.client[self.db_name]
#             self.collection = self.db[self.collection_name]
#             logger.info(f"Connected to MongoDB Database")

#             return self.collection
#         except Exception as e:
#             logger.error(f"Error connecting to MongoDB: {e}")
#             raise CustomException(e, sys)
        
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.client:
#             self.client.close()
#             logger.info("MongoDB connection closed.")
        
# class DataIngestion:
#     def __init__(self, config: DataIngestionConfig, user_name: str):
#         """
#         Initializes DataIngestion with the configuration.

#         Args:
#             config (DataIngestionConfig): Data ingestion configuration.
#              user_name (str): The name of the person doing ingestion
#         """
#         self.config = config
#         self.user_name = user_name
#         config_data = self.config.config_data
#         self.mongo_connection = MongoDBConnection(
#             config_data['mongo_uri'],
#             config_data['database_name'],
#             config_data['collection_name']
#         )

#     def import_data_from_mongodb(self):
#         start_time = time.time()
#         start_timestamp = datetime.now()

#         try:
#             logger.info("Starting data ingestion from MongoDB.")
#             with self.mongo_connection as collection:
#                 all_data = self._fetch_all_data(collection)
#                 cleaned_data = self._clean_data(all_data)
#                 output_path = self._save_data(cleaned_data)
#                 total_records = len(cleaned_data)
#                 logger.info(f"Total records fetched: {total_records}")
#                 self._save_metadata(start_time, start_timestamp, total_records, output_path)
#                 logger.info("Data ingestion completed successfully.")
#         except Exception as e:
#             logger.error(f"Error during data ingestion: {e}")
#             raise CustomException(e, sys)

#     def _fetch_all_data(self, collection) -> pd.DataFrame:
#         try:
#             logger.info("Fetching all data from MongoDB in batches.")
#             batch_size = self.config.config_data['batch_size']

#             def data_generator():
#                 skip_count = 0
#                 while True:
#                     logger.debug(f"fetching batch with skip: {skip_count}, limit:{batch_size}")
#                     cursor = collection.find({}, {'_id': 0}).skip(skip_count).limit(batch_size)
#                     batch = list(cursor)
                    
#                     if not batch:
#                         logger.info("No more data found")
#                         break

#                     if len(batch) < batch_size:
#                         logger.info("Last batch of records")
#                         yield pd.DataFrame(batch)
#                         break    
                        
#                     yield pd.DataFrame(batch)
#                     skip_count += batch_size



#             all_data = list(data_generator())
#             final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
#             logger.info(f"Data fetched successfully. Total records: {len(final_df)}")

#             return final_df
        
#         except Exception as e:
#             logger.error(f"Error fetching data from MongoDB: {e}")
#             raise CustomException(e, sys)
        

#     def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         try:
#             logger.info(f"Cleaning data from MongoDB")
#             df.replace([np.inf, -np.inf], np.nan, inplace=True)
#             if df.isnull().values.any():
#                 logger.warning("Data contains NaN values. Drop rows with NaN values")
#                 df = df.dropna()
#             logger.info(f"Data cleaning completed successfully.")
            
#             return df
        
#         except Exception as e:
#             logger.error(f"Error cleaning data: {e}")
#             raise CustomException(e, sys)
    
#     def _save_data(self, all_data:pd.DataFrame) -> Path:
#         try:
#             output_path = Path(self.config.config_data['root_dir']) / "website_visitors.parquet"
#             all_data.to_parquet(output_path, index=False)
#             logger.info(f"Data saved to {output_path}")

#             return output_path
        
#         except Exception as e:
#             logger.error(f"Error saving data: {e}")
#             raise CustomException(e, sys)
        
#     def _save_metadata(self, start_time: float, start_timestamp: datetime, total_records: int, output_path: Path):
#         try:
#             end_time = time.time()
#             end_timestamp = datetime.now()
#             duration = end_time - start_time
#             metadata = {
#                'start_time': start_timestamp.isoformat(),
#                 'end_time': end_timestamp.isoformat(),
#                 'duration_seconds': duration,
#                 "total_records": total_records,
#                 "data_source": self.config.config_data['collection_name'],
#                 "ingested_by": self.user_name,
#                 "output_path": str(output_path)

#             }

#             metadata_path = Path(self.config.config_data['root_dir']) / "data-ingestion-metadata.json"

#             with open(metadata_path, 'w') as f:
#                 json.dump(metadata, f, indent=4)

#             logger.info(f"Metadata saved successfully")

#         except Exception as e:
#             logger.error(f"Error saving metadata: {e}")
#             raise CustomException(e, sys)
            


# if __name__ == "__main__":
#     try:
#         user_name = input("Enter your name: ")
#         config_manager = ConfigurationManager()
#         data_ingestion_config = config_manager.get_data_ingestion_config()
#         data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
#         data_ingestion.import_data_from_mongodb()
#         logger.info("Data ingestion process completed successfully.")
#     except CustomException as e:
#         logger.error(f"Error during data ingestion: {e}")
#         logger.info("Data ingestion process failed.")

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
import logging
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
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
        try:
            data_config = self.ingestion_config.data_ingestion
            create_directories([data_config['root_dir']])
            logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
            data_config['mongo_uri'] = os.environ.get('MONGO_URI')
            return DataIngestionConfig(config_data=data_config)
        except Exception as e:
            logger.error(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)
        
class MongoDBConnection:
    """Handles MongoDB connections."""
    def __init__(self, uri, db_name, collection_name):
        """
        Initializes MongoDBConnection with the given URI, database name, and collection name.

        Args:
            uri (str): MongoDB URI.
            db_name (str): Name of the MongoDB database.
            collection_name (str): Name of the MongoDB collection.
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None


    def __enter__(self):
        try:
            logger.info(f"Connecting to MongoDB Database")
            self.client = pymongo.MongoClient(self.uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info(f"Connected to MongoDB Database")

            return self.collection
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise CustomException(e, sys)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")
        
class DataIngestion:
    def __init__(self, config: DataIngestionConfig, user_name: str):
        """
        Initializes DataIngestion with the configuration.

        Args:
            config (DataIngestionConfig): Data ingestion configuration.
             user_name (str): The name of the person doing ingestion
        """
        self.config = config
        self.user_name = user_name
        config_data = self.config.config_data
        self.mongo_connection = MongoDBConnection(
            config_data['mongo_uri'],
            config_data['database_name'],
            config_data['collection_name']
        )

    def import_data_from_mongodb(self):
        start_time = time.time()
        start_timestamp = datetime.now()

        try:
            logger.info("Starting data ingestion from MongoDB.")
            with self.mongo_connection as collection:
                all_data = self._fetch_all_data(collection)
                cleaned_data = self._clean_data(all_data)
                output_path = self._save_data(cleaned_data)
                total_records = len(cleaned_data)
                logger.info(f"Total records fetched: {total_records}")
                self._save_metadata(start_time, start_timestamp, total_records, output_path)
                logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)
        
    def _fetch_all_data(self, collection) -> pd.DataFrame:
        try:
            logger.info("Fetching all data from MongoDB in batches.")
            batch_size = self.config.config_data['batch_size']

            def data_generator():
                skip_count = 0
                while True:
                    logger.debug(f"fetching batch with skip: {skip_count}, limit:{batch_size}")
                    cursor = collection.find({}, {'_id': 0}).skip(skip_count).limit(batch_size)
                    batch = list(cursor)

                    if not batch:
                        logger.info("No more data found")
                        break
                    
                    if len(batch) < batch_size:
                         logger.info("Last batch of records")
                         yield pd.DataFrame(batch)
                         break    
                     

                    yield pd.DataFrame(batch)
                    skip_count += batch_size


            all_data = list(data_generator())
            final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            logger.info(f"Data fetched successfully. Total records: {len(final_df)}")

            return final_df
        
        except Exception as e:
            logger.error(f"Error fetching data from MongoDB: {e}")
            raise CustomException(e, sys)
        

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(f"Cleaning data from MongoDB")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            if df.isnull().values.any():
                logger.warning("Data contains NaN values. Drop rows with NaN values")
                df = df.dropna()
            logger.info(f"Data cleaning completed successfully.")
            
            return df
        
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise CustomException(e, sys)
    
    def _save_data(self, all_data:pd.DataFrame) -> Path:
        try:
            output_path = Path(self.config.config_data['root_dir']) / "website_visitors.parquet"
            all_data.to_parquet(output_path, index=False)
            logger.info(f"Data saved to {output_path}")

            return output_path
        
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise CustomException(e, sys)
        
    def _save_metadata(self, start_time: float, start_timestamp: datetime, total_records: int, output_path: Path):
        try:
            end_time = time.time()
            end_timestamp = datetime.now()
            duration = end_time - start_time
            metadata = {
               'start_time': start_timestamp.isoformat(),
                'end_time': end_timestamp.isoformat(),
                'duration_seconds': duration,
                "total_records": total_records,
                "data_source": self.config.config_data['collection_name'],
                "ingested_by": self.user_name,
                "output_path": str(output_path)

            }

            metadata_path = Path(self.config.config_data['root_dir']) / "data-ingestion-metadata.json"

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"Metadata saved successfully")

        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise CustomException(e, sys)
            


if __name__ == "__main__":
    try:
        user_name = input("Enter your name: ")
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
        data_ingestion.import_data_from_mongodb()  # This is the only call
        logger.info("Data ingestion process completed successfully.")
    except CustomException as e:
        logger.error(f"Error during data ingestion: {e}")
        logger.info("Data ingestion process failed.")