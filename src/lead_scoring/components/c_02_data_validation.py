
import pandas as pd
import json
import sys
from src.lead_scoring.logger import logger
from src.lead_scoring.exception import CustomException
from ydata_profiling import ProfileReport
from src.lead_scoring.config_entity.config_params import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.dtype_mapping = {
            "string": ["object", "category", "string"],
            "integer": ["int32", "int64"],
            "number": ["float32", "float64"],  
            "float": ["float32", "float64"], 
            "boolean": ["bool"],
            "datetime": ["datetime64[ns]", "datetime64"]
        }

    def validate_all_columns(self) -> bool:
        try:
            overall_status = True
            validation_results = {}

            try:
                data = pd.read_parquet(self.config.data_dir)
            except Exception as e:
                logger.error(f"Error reading Parquet file: {e}")
                raise CustomException(f"Error reading Parquet file: {e}", sys)

            all_cols = list(data.columns)
            all_schema = self.config.all_schema

            for col in all_cols:
                schema = all_schema.get(col)
                if not schema:
                    validation_results[col] = "Column missing in schema"
                    overall_status = False
                    logger.error(f"Column {col} missing in schema")
                    continue

                actual_dtype = str(data[col].dtype)
                expected_dtype = schema['type']
                if actual_dtype not in self.dtype_mapping.get(expected_dtype, []):
                    validation_results[col] = f"Incorrect dtype: expected {expected_dtype}, got {actual_dtype}"
                    overall_status = False
                    logger.error(f"Column {col}: {validation_results[col]}")
                    continue

                constraints = schema.get('constraints', {})
                if 'enum' in constraints:
                    allowed_values = set(constraints['enum'])
                    invalid_values = set(data[col]) - allowed_values
                    if invalid_values:
                        validation_results[col] = f"Invalid categorical values: {invalid_values}"
                        overall_status = False
                        logger.error(f"Column {col}: {validation_results[col]}")
                        continue

                if 'minimum' in constraints and 'maximum' in constraints:  
                    min_val, max_val = constraints['minimum'], constraints['maximum']
                    out_of_range = ~((data[col] >= min_val) & (data[col] <= max_val))
                    if out_of_range.any():
                        validation_results[col] = f"Out of range: [{min_val}, {max_val}]"
                        overall_status = False
                        logger.error(f"Column {col}: {validation_results[col]}")
                        continue

                validation_results[col] = "Valid"

            self._save_validation_results(validation_results)

            if overall_status:
                overall_status = self._save_validated_data(data)

            return overall_status

        except Exception as e:
            logger.exception(f"Error during validation: {e}")
            raise CustomException(e, sys)

    def _save_validation_results(self, validation_results: dict):
        val_status_path = self.config.val_status
        try:
            with open(val_status_path, 'w') as f:
                json.dump(validation_results, f, indent=4)
            logger.info(f"Validation results saved to {val_status_path}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
            raise CustomException(f"Failed to save validation results: {e}", sys)

    def _save_validated_data(self, data: pd.DataFrame) -> bool:
        try:
            output_path = self.config.validated_data
            data.to_parquet(output_path, index=False)
            logger.info(f"Validated data saved to {output_path}")

            profile = ProfileReport(data, title="Validated Data Profile")
            report_path = self.config.root_dir / self.config.profile_report_name
            profile.to_file(report_path)
            logger.info(f"Profile report generated at: {report_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to save validated data or generate profile report: {e}")
            raise CustomException(f"Failed to save validated data or generate profile report: {e}", sys)
