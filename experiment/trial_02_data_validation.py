

import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
# Added: Importing json for saving validation results
import json
# Added: Importing sys for CustomException
import sys

from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *
from src.lead_scoring.exception  import CustomException


@dataclass
class DataValidationConfig:
    root_dir: Path
    val_status: str
    data_dir: Path
    all_schema: dict
    critical_columns: list


class ConfigurationManager:
    def __init__(
              self,
              data_validation_config: Path = DATA_VALIDATION_CONFIG_FILEPATH,
              schema_config: Path = SCHEMA_CONFIG_FILEPATH) -> None:
        
        logger.info(f"Initializing the configuration manager") # Modified: Added f-string for clarity

        try: #Added: Error handling for read_yaml
            self.data_val_config = read_yaml(data_validation_config)
            self.schema = read_yaml(schema_config) # Added: Error handling for read_yaml
            create_directories([self.data_val_config.artifacts_root]) #Modified: Passing to create_directories as a list
        except Exception as e:
           logger.error(f"Error creating directories") # Modified: Added f-string for clarity
           raise CustomException(e, sys) # Added: Raise CustomException

    def get_data_validation_config(self) -> DataValidationConfig:
        try: # Added: Error Handling
            data_valid_config = self.data_val_config.data_validation # Modified: Access `data_validation` here
            schema_columns = self.schema.get("columns", {})
            target_column = self.schema.get("target_column", [])

            # Combine columns and target column into a single dictionary 
            schema_dict = {col['name']: col['type'] for col in schema_columns}
            schema_dict.update({col['name']: col['type'] for col in target_column})

            create_directories([data_valid_config.root_dir]) #Modified: Passing to create_directories as a list
            logger.debug(f"Data Validation Config Loaded") # Modified: Added f-string for clarity

            return DataValidationConfig(
                root_dir = Path(data_valid_config.root_dir), # Modified: Access `data_validation` here
                val_status = data_valid_config.val_status, # Modified: Access `data_validation` here
                data_dir = Path(data_valid_config.data_dir), # Modified: Access `data_validation` here
                all_schema = schema_dict,
                critical_columns = data_valid_config.critical_columns
            )
        except Exception as e: #Added: Catching all exceptions
            logger.error(f"Error getting data validation configuration: {str(e)}") # Modified: Added f-string for clarity
            raise CustomException(e, sys) # Added: Raise CustomException

class DataValidation:
    def __init__ (self, config: DataValidationConfig):
        self.config = config
        logger.info(f"Data Validation initialized")  # Modified: Added f-string for clarity
        logger.debug(f"Data validation config: {self.config}") #Added: Logging the config


    def validate_columns(self, data):
        """Validate that all expected columns are present and no extra columns exist"""

        all_cols = list(data.columns)
        all_schema = list(self.config.all_schema.keys())

        missing_columns = [col for col in all_schema if col not in all_cols]
        extra_columns = [col for col in all_cols if col not in all_schema]

        error_message = {'missing_columns': missing_columns, 'extra_columns': extra_columns}

        if missing_columns or extra_columns:
            logger.error(f"Column validation failed: {error_message}") # Modified: Added f-string for clarity
            return False, error_message
        logger.info(f"Column validation passed") # Added: Log when the validation passes
        return True, None

# Check on this    
    def validate_data_types(self, data):
        """Validate that all columns have the correct data types"""

        dtype_mapping = {
            "string": ["object", "category", 'string'],
            "integer": ["int32", "int64"],
            "float": ["float32", "float64"],
            "boolean": ["bool"],
            "datetime": ["datetime64[ns]", "datetime64"]
        }

        all_schema = self.config.all_schema
        type_mismatches = {}
        validation_status = True

        for col, expected_type in all_schema.items(): # Modified: Getting items
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if isinstance(expected_type, dict):
                    expected_type = expected_type.get("type", None)

                if not isinstance(expected_type, str):
                    logger.debug(f"Invalid data type for column {col}: {expected_type}")  # Modified: Added f-string for clarity
                    expected_type = None
                expected_pandas_type = dtype_mapping.get(expected_type, None)

                if expected_pandas_type and actual_type not in expected_pandas_type:
                    type_mismatches[col] = {"expected": expected_type, "actual": actual_type}
                    validation_status = False


        if type_mismatches:
            logger.error(f"Data type validation failed: {type_mismatches}") # Modified: Added f-string for clarity
        else:
          logger.info(f"Data type validation passed") # Added: Logging when it passes

        return validation_status, type_mismatches

    def _validate_missing_values(self, data):
        """Validate that columns have any missing values"""
        missing_values = {}

        for col in self.config.critical_columns:
            if col not in data.columns:
                missing_values[col] = "Column not present"

            elif data[col].isnull().sum() > 0:
                missing_values[col] = data[col].isnull().sum()


        if missing_values:
            logger.error(f"Missing value check failed: {missing_values}") # Modified: Added f-string for clarity
            return False, missing_values
        logger.info(f"Missing value check passed") # Added: Log when validation is correct
        return True, None
    
    def validate_string_values(self, data):
        """Validate that string-type columns contain only allowed values"""
        all_schema = self.config.all_schema.get("columns", {})
        invalid_values = {}

        for col, col_schema in all_schema.items():
          col_type = col_schema.get("type", None)
          constraints = col_schema.get("constraints", [])

          if col_type == "string" and col in data.columns:
              enum_values = next(
                  (c["values"] for c in constraints if c["type"] == "enum"), None
              )

              if enum_values:
                  invalid_entries = data[~data[col].isin(enum_values)]
                  if not invalid_entries.empty:
                      invalid_values[col] = invalid_entries[col].tolist()
        if invalid_values:
             logger.error(f"String value validation failed: {invalid_values}") # Modified: Added f-string for clarity
             return False, invalid_values
        logger.info(f"String value validation passed") # Added: Log when validation passes
        return True, None
    
    def validate_numeric_ranges(self, data):
        """Validate that numeric-type columns fall within the specified range"""
        all_schema = self.config.all_schema.get("columns", {})
        range_violations = {}

        for col, col_schema in all_schema.items():
            col_type = col_schema.get("type", None)
            constraints = col_schema.get("constraints", [])

            if col_type in ["integer", "float"] and col in data.columns:
                range_constraints = next(
                    (c for c in constraints if c["type"] == "range"), {}
                )
                min_value = range_constraints.get("min")
                max_value = range_constraints.get("max")

                if min_value is not None and data[col].min() < min_value:
                    range_violations[col] = {
                        "min_violation": data[data[col] < min_value][col].tolist()
                    }
                if max_value is not None and data[col].max() > max_value:
                    range_violations[col].setdefault("max_violation", []).extend(
                        data[data[col] > max_value][col].tolist()
                    )

        if range_violations:
            logger.error(f"Numeric range validation failed: {range_violations}") # Modified: Added f-string for clarity
            return False, range_violations
        logger.info(f"Numeric range validation passed") # Added: Log when validation passes
        return True, None
    def validate_numeric_values(self, data):
        """Validate that numeric-type columns contain only allowed discreet values"""

        all_schema = self.config.all_schema.get("columns", {})
        invalid_values = {}

        for col, col_schema in all_schema.items():
            col_type = col_schema.get("type", None)
            constraints = col_schema.get("constraints", [])

            if col_type in ["integer", "float"] and col in data.columns:
                enum_values = next(
                    (c["values"] for c in constraints if c["type"] == "enum"), None
                )

                if enum_values:
                    invalid_entries = data[~data[col].isin(enum_values)]
                    if not invalid_entries.empty:
                        invalid_values[col] = invalid_entries[col].tolist()

        if invalid_values:
            logger.error (f"Numeric value validation failed: {invalid_values}")
            return False, invalid_values

        logger.info(f"Numeric value validation passed")
        return True, None

    
    def check_cardinality(self, data):
        """"Check  and drop columns with unique values"""

        drop_columns = [col for col in data.columns if data[col].nunique() == len(data)]
        if drop_columns: # Only print if there are columns to drop
            logger.info(f"Dropping columns with unique values: {drop_columns}") # Modified: Added f-string for clarity
        
        data.drop(columns=drop_columns, inplace=True)
        logger.debug(f"Dropped columns with unique values: {drop_columns}") # Modified: Added f-string for clarity

    def validate_data(self, data):
      """Validate the data and return a status, save the metadata of the validation,
         and save the data if validation is passed
      """
      validation_results = {}
      # Validate columns
      status, error_message = self.validate_columns(data)
      validation_results["validate_columns"] = {"status": status, "errors": error_message}

      # Validate data types
      status, type_mismatches = self.validate_data_types(data)
      validation_results["validate_data_types"] = {"status": status, "mismatches": type_mismatches}

      # Validate missing values
      status, missing_values = self._validate_missing_values(data)
      validation_results["validate_missing_values"] = {"status": status, "errors": missing_values}

      # Validate string values
      status, string_errors = self.validate_string_values(data)
      validation_results["validate_string_values"] = {"status": status, "errors": string_errors}

      # Validate numeric values
      status, numeric_errors = self.validate_numeric_values(data)
      validation_results['validate_numeric_values'] = {"status": status, "errors": numeric_errors}

      # Validate numeric ranges
      status, numeric_errors = self.validate_numeric_ranges(data)
      validation_results["validate_numeric_ranges"] = {"status": status, "errors": numeric_errors}

      # Save results to a file
      with open(self.config.val_status, 'w') as f:
          json.dump(validation_results, f, indent=4)
      logger.info(f"Validation results saved to {self.config.val_status}")

      overall_status = all(result["status"] for result in validation_results.values())

      # Check and Drop columns with unique values
      self.check_cardinality(data)

       # Save the data to a parquet file only if the validation passed
      if overall_status:
        output_path = str(Path(self.config.root_dir) / 'validated_data.parquet')
        data.to_parquet(output_path, index=False)
        logger.info(f"Validated data saved to {output_path}")
      else:
        logger.warning(f"Data validation failed. Check {self.config.val_status} for more details")
      
      return overall_status



if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data = pd.read_parquet(data_validation_config.data_dir)

        logger.info("Starting data validation process") # Modified: Added f-string for clarity
        validation_status = data_validation.validate_data(data)

        if validation_status:
            logger.info("Data Validation Completed Successfully!")
        else:
            logger.warning("Data Validation Failed. Check the status file for more details.")

    except Exception as e:
        logger.error(f"Data validation process failed: {e}")
        raise CustomException(e, sys) # Added: Raise CustomException