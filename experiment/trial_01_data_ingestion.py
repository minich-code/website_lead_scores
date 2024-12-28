
import sys
# Add the project's source directory to the Python path for importing modules.
sys.path.append('/home/western/DS_Projects/website_lead_scores')

# Import dataclass for creating data classes, pathlib for file system paths
from dataclasses import dataclass
from pathlib import Path
import pymongo
import pandas as pd
import numpy as np
# Import os for operating system related functions
import os
import json
import time


from datetime import datetime
from pymongo import MongoClient
from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *

# Define a data class to store data ingestion configuration
@dataclass
class DataIngestionConfig:
    # Define the root directory for artifacts
    root_dir: Path
    mongo_uri: str
    database_name: str
    collection_name: str
    batch_size: int

# Define a class to manage configurations
class ConfigurationManager:
    # Initialize ConfigurationManager with data_ingestion config file
    def __init__(self, data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH):
        try:
            # Log the initialization of the ConfigurationManager
            logger.info(f"Initializing ConfigurationManager with config file: {data_ingestion_config}")
            # Load the config from the yaml file
            self.ingestion_config = read_yaml(data_ingestion_config)
            # Get the root directory of the artifacts
            artifacts_root = self.ingestion_config.artifacts_root
            # Create all the necessary directories
            create_directories([artifacts_root])
            # Log that the configuration directories have been created successfully
            logger.info("Configuration directories created successfully.")
        except Exception as e:
            # Log the error and raise a custom exception if there is an error during initialization
            logger.error(f"Error initializing ConfigurationManager: {str(e)}")
            raise CustomException(e, sys)
    # Function to get data ingestion configuration
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns the data ingestion configuration object.
        """
        try:
            # Gets the data ingestion section of the configuration
            data_config = self.ingestion_config.data_ingestion
            # creates all necessary directories
            create_directories([data_config.root_dir])
            # Log the configuration has been loaded from the file
            logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
            # Return an instance of DataIngestionConfig with parameters from the file
            return DataIngestionConfig(
                root_dir=Path(data_config.root_dir),
                mongo_uri=data_config.mongo_uri,
                database_name=data_config.database_name,
                collection_name=data_config.collection_name,
                batch_size=data_config.batch_size
            )
        except Exception as e:
            # Log the error and raise a custom exception if an error occurs while loading config
            logger.error(f"Error loading data ingestion configuration: {str(e)}")
            raise CustomException(e, sys)

# Define a class for data ingestion
class DataIngestion:
    # Initialize DataIngestion with the configuration
    def __init__(self, config: DataIngestionConfig):
        # set the config for the class
        self.config = config
        try:
             # Log the connection to MongoDB with masked URI
            logger.info(f"Connecting to MongoDB using URI: {self.config.mongo_uri} (masked for security)")
            # Connect to the MongoDB client using the URI from the config
            self.client = pymongo.MongoClient(self.config.mongo_uri)
            # Get the database from the client
            self.db = self.client[self.config.database_name]
             # Get the collection from the database
            self.collection = self.db[self.config.collection_name]
             # Log successful connection to the database and collection
            logger.info(f"Connected to MongoDB database: {self.config.database_name}, collection: {self.config.collection_name}")
        except Exception as e:
            # Log the error and raise a custom exception if there is error connecting to MongoDB
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise CustomException(e, sys)

    # Function to import data from MongoDB
    def import_data_from_mongodb(self):
        """
        Imports data from MongoDB to a Pandas DataFrame.
        """
        # Get the start time for tracking performance
        start_time = time.time()
        # Get the start timestamp for the metadata
        start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            # Log that the data ingestion has started
            logger.info("Starting data ingestion from MongoDB.")
            # Fetch all the data from MongoDB
            all_data = self.fetch_all_data()

            # Save the data to a Parquet file
            output_path = self.save_data(all_data)
            # Get the total records imported
            total_records = len(all_data)
            # Log that total records fetched
            logger.info(f"Total records fetched: {total_records}")
             # Save the metadata of the data ingestion
            self._save_metadata(start_time, start_timestamp, total_records, output_path)
            # Log that data ingestion has completed successfully
            logger.info("Data ingestion completed successfully.")
        except Exception as e:
            # Log the error and raise a custom exception if there was an error during data ingestion
            logger.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)

    # Function to fetch data from MongoDB
    def fetch_all_data(self):
        try:
            # Log that fetching all data has started
            logger.info("Fetching all data from MongoDB collection.")
             # Get all documents from the MongoDB collection, exclude the `_id` field
            data_cursor = self.collection.find({}, {'_id': 0})
             # Create list from the cursor
            data = list(data_cursor)
            # Create a Pandas DataFrame from the retrieved data
            df = pd.DataFrame(data)

            # Replace infinite values with NaN and drop all rows with NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Check if there are any NaN values
            if df.isnull().values.any():
                # Log a warning if there is any NaN values present
                logger.warning("Data contains NaN values. Dropping rows with NaN values.")
            # Remove rows that contains NaN
            df = df.dropna()

            # Log that data fetched successfully with the total number of records
            logger.info(f"Data fetched successfully. Total records: {len(df)}")
            # return the dataframe
            return df
        except Exception as e:
             # Log the error and raise a custom exception if there was an issue fetching from MongoDB
            logger.error(f"Error fetching data from MongoDB: {str(e)}")
            raise CustomException(e, sys)

    # Function to save data into parquet format
    def save_data(self, all_data):
        try:
            # Create the path for the parquet file
            output_path = str(Path(self.config.root_dir) / f"website_visitors.parquet")
            # Save the DataFrame to a Parquet file without index
            all_data.to_parquet(output_path, index=False)
            # Log that data has been saved to parquet
            logger.info(f"Data saved to Parquet file: {output_path}")
             # return the output path
            return output_path
        except Exception as e:
            # Log the error and raise a custom exception if there is an error saving the file to parquet
            logger.error(f"Error saving data to Parquet: {str(e)}")
            raise CustomException(e, sys)

    # Function to save the metadata of data ingestion
    def _save_metadata(self, start_time, start_timestamp, total_records, output_path):
        try:
            # Get end time for tracking performance
            end_time = time.time()
            # Get end time stamp of ingestion process
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Calculate duration of ingestion process
            duration = end_time - start_time
             # Create a metadata object
            metadata = {
                'start_time': start_timestamp,
                'end_time': end_timestamp,
                'duration': duration,
                'total_records': total_records,
                'data_source': self.config.collection_name,
                'output_path': output_path
            }
            # Create the path to save the metadata
            metadata_path = str(Path(self.config.root_dir) / "data-ingestion-metadata.json")
            # Open a file and write metadata into it
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
             # Log that metadata saved successfully and the path
            logger.info(f"Metadata saved successfully to: {metadata_path}")
            # Log metadata content as debug
            logger.debug(f"Metadata content: {metadata}")
        except Exception as e:
            # Log the error and raise a custom exception if metadata saving fails
            logger.error(f"Error during metadata saving: {str(e)}")
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    try:
        # Create an instance of ConfigurationManager
        config_manager = ConfigurationManager()
        # Get data ingestion configuration from ConfigurationManager
        data_ingestion_config = config_manager.get_data_ingestion_config()
         # Create instance of the DataIngestion
        data_ingestion = DataIngestion(config=data_ingestion_config)
        # Import data from mongodb
        data_ingestion.import_data_from_mongodb()
         # Log that data ingestion is completed successfully
        logging.info("Data ingestion process completed successfully.")
    except CustomException as e:
        # Log the error if an exception occurs during data ingestion process and log that process has failed
        logger.error(f"Error during data ingestion: {str(e)}")
        logger.info("Data ingestion process failed.")