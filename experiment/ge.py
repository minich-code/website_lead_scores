import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

from typing import Tuple, Optional, Dict, List, Any
from collections import defaultdict
import pandas as pd
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import json
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *
from src.lead_scoring.exception  import CustomException
from soda.scan import Scan
from ydata_profiling import ProfileReport


@dataclass
class DataValidationConfig:
    """
    Configuration for data validation process.

    """
    root_dir: Path
    val_status: str
    data_dir: Path
    all_schema: Dict[str, Any]
    critical_columns: List[str]

class ConfigurationManager:
    def __init__(
              self,
              data_validation_config: Path = DATA_VALIDATION_CONFIG_FILEPATH,
              schema_config: Path = SCHEMA_CONFIG_FILEPATH) -> None:
        
        logger.info(f"Initializing the configuration manager") 

        try:
            self.data_val_config = read_yaml(data_validation_config)
            self.schema = read_yaml(schema_config) 
            create_directories([self.data_val_config.artifacts_root]) 
        except Exception as e:
            logger.exception(f"Error creating directories") 
            raise CustomException(e, sys) 

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_valid_config = self.data_val_config.data_validation 
            schema_dict = self._process_schema()

            create_directories([Path(data_valid_config.root_dir)]) 
            logger.info(f"Data Validation Config Loaded") 

            return DataValidationConfig(
                root_dir = Path(data_valid_config.root_dir), 
                val_status = data_valid_config.val_status, 
                data_dir = Path(data_valid_config.data_dir), 
                all_schema = schema_dict,
                critical_columns = data_valid_config.critical_columns
            )
        except Exception as e: 
            logger.exception(f"Error getting data validation configuration: {str(e)}") 
            raise CustomException(e, sys)

    def _process_schema(self) -> Dict[str, str]:
        schema_columns = self.schema.get("columns", {})
        target_column = self.schema.get("target_column", [])
        schema_dict = {col['name']: col['type'] for col in schema_columns}
        schema_dict.update({col['name']: col['type'] for col in target_column})
        return schema_dict

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)  
        self.logger.info(f"Data Validation initialized")  
        self.logger.debug(f"Data validation config: {self.config}") 

    def _generate_profile_report(self, data: pd.DataFrame) -> str:
        """
        Generates a pandas profile report and returns the path to the HTML report
        
        Args:
            data (pd.DataFrame): The DataFrame to profile
            
        Returns:
            str: Path to the generated HTML report
        """
        try:
            report_path = Path(self.config.root_dir) / "data_profile_report.html"
            
            profile = ProfileReport(data, title="Data Profiling Report")
            profile.to_file(str(report_path))
            
            self.logger.info(f"Profile report generated at: file://{report_path.absolute()}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating or saving pandas profile report: {e}")
            raise CustomException(e, sys)

    def check_cardinality(self, data):
        """"Check  and drop columns with unique values"""

        unique_counts = data.nunique()
        drop_columns = [col for col in data.columns if unique_counts[col] == len(data)]
        if drop_columns: 
            logger.warning(f"Dropping columns with unique values: {drop_columns}")
    
        try:
            data.drop(columns=drop_columns, inplace=True)
            logger.debug(f"Dropped columns with unique values: {drop_columns}")
        except Exception as e:
            logger.error(f"Error occurred while dropping columns: {e}")
        
        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validates data using Soda Core and returns the validation status.
        """
        try:
            # Initialize Soda scan
            scan = Scan()
            # Add config and check files
            #scan.add_configuration_yaml_file("configuration.yml")
            scan.add_checks_yaml_file("checks/checks.yaml")
            #Set datasource to pandas and pass in dataframe
            scan.set_data_source_scan_results(data_source_name="pandas_dataframe", scan_results={"pandas_dataframe":data})
            # Execute the scan
            scan.execute()
            # Get results
            scan_results = scan.get_scan_results()
            # Extract status from scan results
            overall_status =  scan_results["has_failures"] is False

            # Generate and save the profile report
            report_path = self._generate_profile_report(data)
            self.logger.info(f"To view the profile report, open this link in your browser: file://{Path(report_path).absolute()}")
            
            # Validate or sanitize file paths
            val_status_path = Path(self.config.val_status).resolve(strict=False)
            root_dir_path = Path(self.config.root_dir).resolve(strict=False)
            
            # Save results to a file
            try:
              with open(val_status_path, 'w') as f:
                json.dump(scan_results, f, indent=4)
              logger.info(f"Soda scan results saved to {val_status_path}")
            except Exception as e:
              logger.error(f"Failed to save soda scan results: {e}")
            
            # Check and drop columns with unique values
            self.check_cardinality(data)


             # Save the data to a parquet file only if the validation passed
            if overall_status:
              try:
                output_path = str(root_dir_path / 'validated_data.parquet')
                with open(output_path, 'wb') as f:
                    data.to_parquet(f, index=False)
                logger.info(f"Validated data saved to {output_path}")
              except Exception as e:
                logger.error(f"Failed to save validated data: {e}")
            else:
              logger.warning(f"Data validation failed. Check {val_status_path} for more details")
            
            return overall_status
        
        except Exception as e:
            self.logger.error(f"Error during Soda Scan: {e}")
            return False


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        
        print(f"Data Directory: {data_validation_config.data_dir}") #Check that the data directory is correct
        data = pd.read_parquet(data_validation_config.data_dir)
        print("Loaded DataFrame Shape:", data.shape)
        
        logger.info("Starting data validation process") 
        validation_status = data_validation.validate_data(data)

        if validation_status:
            logger.info("Data Validation Completed Successfully!")
        else:
            logger.warning("Data Validation Failed. Check the status file for more details.")

    except Exception as e:
        logger.error(f"Data validation process failed: {e}")
        raise CustomException(e, sys)