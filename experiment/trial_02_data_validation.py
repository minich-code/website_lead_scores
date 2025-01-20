   

import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

from typing import Tuple, Optional, Dict, List, Any
from collections import defaultdict
import pandas as pd
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import json
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *
from src.lead_scoring.exception  import CustomException
from ydata_profiling import ProfileReport  # Updated import

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

        try: #Added: Error handling for read_yaml
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

    def validate_columns(self, data: pd.DataFrame) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate that all expected columns are present and no extra columns exist.

        Parameters:
        - data (pd.DataFrame): The DataFrame to validate.

        Returns:
        - Tuple[bool, Optional[Dict[str, Any]]]: A tuple containing:
            - bool: True if validation passes, False otherwise
            - Optional[Dict[str, Any]]: None if validation passes, error details if validation fails
        """
        try:
            all_cols = frozenset(data.columns)
            all_schema = frozenset(self.config.all_schema.keys())

            missing_columns = list(all_schema - all_cols)
            extra_columns = list(all_cols - all_schema)

            if missing_columns or extra_columns:
                error_message = {
                    'missing_columns': missing_columns,
                    'extra_columns': extra_columns
                }
                self.logger.error(f"Column validation failed: {error_message}")
                return False, error_message

            self.logger.info("Column validation passed")
            return True, None

        except Exception as e:
            self.logger.error(f"An error occurred during column validation: {e}")
            return False, {"error": str(e)}

    def validate_data_types(self, data):
        """Validate that all columns have the correct data types"""

        dtype_mapping = {
            "string": ["object", "category", "string"],
            "integer": ["int32", "int64"],
            "float": ["float32", "float64"],
            "boolean": ["bool"],
            "datetime": ["datetime64[ns]", "datetime64"]
        }

        all_schema = self.config.all_schema
        type_mismatches = {}
        validation_status = True

        for col, expected_type in all_schema.items():
            if col not in data.columns:
                continue

            actual_type = str(data[col].dtype)
            if isinstance(expected_type, dict):
                expected_type = expected_type.get("type")

            if not isinstance(expected_type, str):
                logger.debug(f"Invalid data type for column {col}: {expected_type}")
                continue

            expected_pandas_type = dtype_mapping.get(expected_type)
            if expected_pandas_type and actual_type not in expected_pandas_type:
                type_mismatches[col] = {"expected": expected_type, "actual": actual_type}
                validation_status = False

        if not validation_status:
            logger.error(f"Data type validation failed: {type_mismatches}")
            return validation_status, type_mismatches
        else:
            logger.info("Data type validation passed")
            return validation_status, None

    def _validate_missing_values(self, data: pd.DataFrame) -> Tuple[bool, Optional[Dict[str, int]]]:
        """Validate that columns have any missing values"""
        if not isinstance(data, pd.DataFrame):
            logger.error("Input data is not a DataFrame")
            return False, {"error": "Input data is not a DataFrame"}

        missing_values = {
            col: ("Column not present" if col not in data.columns else data[col].isnull().sum())
            for col in self.config.critical_columns
            if col not in data.columns or data[col].isnull().sum() > 0
        }

        if missing_values:
            logger.debug(f"Missing value check failed: {missing_values}")
            raise CustomException(f"Missing critical columns: {missing_values}")
        logger.info(f"Missing value check passed") 
        return True, None
    
    def validate_string_values(self, data: pd.DataFrame) -> Tuple[bool, Optional[Dict[str, List[str]]]]:
        """Validate that string-type columns contain only allowed values"""
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Input data is not a DataFrame")
            return False, {"error": "Input data is not a DataFrame"}
        try:
            all_schema = self.config.all_schema.get("columns", {})
            invalid_values = {}

            def extract_enum_values(constraints: List[Dict[str, Any]]) -> Optional[List[str]]:
                for constraint in constraints:
                    if constraint["type"] == "enum":
                        return constraint["values"]
                return None

            for col, col_schema in all_schema.items():
                col_type = col_schema.get("type", None)
                constraints = col_schema.get("constraints", [])

                if col in data.columns and col_type == "string":
                    enum_values = extract_enum_values(constraints)

                    if enum_values:
                        enum_set = set(enum_values)
                        if not data[col].isin(enum_set).all():
                            invalid_values[col] = data[~data[col].isin(enum_set)][col].tolist()
            if invalid_values:
                self.logger.error(f"String value validation failed: {invalid_values}") 
                return False, invalid_values
            self.logger.info(f"String value validation passed") 
            return True, None
        except Exception as e:
            self.logger.exception(f"An error occurred during string value validation: {e}")
            return False, None
    
        #logger = logging.getLogger(__name__)

    def validate_numeric_ranges(self, data: pd.DataFrame) -> Tuple[bool, Optional[Dict[str, List[Any]]]]:
        """Validate that numeric-type columns fall within the specified range"""
        all_schema = self.config.all_schema.get("columns", {})
        range_violations = defaultdict(lambda: {"min_violation": [], "max_violation": []})

        for col, col_schema in all_schema.items():
            col_type = col_schema.get("type", None)
            constraints = col_schema.get("constraints", [])

            if col_type in ["integer", "float"] and col in data.columns:
                range_violations.update(self._check_range_violations(data, col, constraints))

        if range_violations:
            logger.debug(f"Numeric range validation failed: {range_violations}") 
            return False, range_violations
        logger.info(f"Numeric range validation passed") 
        return True, None

    def _check_range_violations(self, data: pd.DataFrame, col: str, constraints: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Helper method to check range violations for a specific column"""
        range_violations = {}
        range_constraints = next(
            (c for c in constraints if c["type"] == "range"), {}
        )
        min_value = range_constraints.get("min")
        max_value = range_constraints.get("max")

        if min_value is not None or max_value is not None:
            mask = ~data[col].between(min_value, max_value, inclusive='both')
            if mask.any():
                range_violations[col] = data.loc[mask, col].tolist()

        return range_violations
    
    def validate_numeric_values(self, data: pd.DataFrame) -> Tuple[bool, Optional[Dict[str, List[Any]]]]:
        """Validate that numeric-type columns contain only allowed discreet values"""

        if not isinstance(data, pd.DataFrame):
            logger.error("Input data is not a DataFrame")
            return False, {"error": "Input data is not a DataFrame"}

        all_schema = self.config.all_schema.get("columns", {})
        
        invalid_values = {
            col: data[~data[col].isin(enum_values)][col].tolist()
            for col, col_schema in all_schema.items()
            if (col_type := col_schema.get("type", None)) in ["integer", "float"]
            and col in data.columns
            and (enum_values := next((c["values"] for c in col_schema.get("constraints", []) if c["type"] == "enum"), None))
            and not data[~data[col].isin(enum_values)].empty
        }

        if invalid_values:
            logger.debug(f"Invalid numeric values found: {invalid_values}")
            logger.warning(f"Numeric value validation failed: {invalid_values}")
            return False, invalid_values

        logger.info(f"Numeric value validation passed")
        return True, None
    
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
        
    # And update the validate_data method to use the returned path:
    def validate_data(self, data: pd.DataFrame):
        """Validate the data and return a status, save the metadata of the validation,
        and save the data if validation is passed
        """
        validation_results = {}
        validation_methods = [
            ("validate_columns", self.validate_columns),
            ("validate_data_types", self.validate_data_types),
            ("validate_missing_values", self._validate_missing_values),
            ("validate_string_values", self.validate_string_values),
            ("validate_numeric_values", self.validate_numeric_values),
            ("validate_numeric_ranges", self.validate_numeric_ranges)
        ]

        for key, method in validation_methods:
            status, errors = method(data)
            validation_results[key] = {"status": status, "errors": errors}
        
        # Generate and save the profile report
        report_path = self._generate_profile_report(data)
        self.logger.info(f"To view the profile report, open this link in your browser: file://{Path(report_path).absolute()}")

        # Validate or sanitize file paths
        val_status_path = Path(self.config.val_status).resolve(strict=False)
        root_dir_path = Path(self.config.root_dir).resolve(strict=False)
        
        # Save results to a file
        try:
            with open(val_status_path, 'w') as f:
                json.dump(validation_results, f, indent=4)
            logger.info(f"Validation results saved to {val_status_path}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

        overall_status = all(result["status"] for result in validation_results.values())

        # Check and Drop columns with unique values
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