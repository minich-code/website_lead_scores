import sys

sys.path.append('/home/western/ds_projects/website_lead_scores')


import os
import pandas as pd
from typing import List, Callable, Dict, Any
from src.lead_scoring.components.c_02_data_validation import DataValidation, DataValidationConfig
from src.lead_scoring.logger import logger
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.exception import CustomException

PIPELINE_NAME = "DATA VALIDATION PIPELINE"


class PipelineStep:
    """Represents a step in the data validation pipeline."""

    def __init__(self, name: str, step_function: Callable):
        self.name = name
        self.step_function = step_function

    def execute(self, pipeline_data: Dict[str, Any]):
        """Executes the pipeline step."""
        try:
            logger.info(f"Executing step: {self.name}")
            result = self.step_function(pipeline_data)
            logger.info(f"Step {self.name} completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error executing step {self.name}: {e}")
            raise CustomException(e, sys)


class DataValidationPipeline:
    """Orchestrates the data validation pipeline."""

    def __init__(self, pipeline_name: str = PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self, initial_data: Dict[str, Any] = {}):
        """Executes the data validation pipeline."""
        try:
            logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")

            pipeline_data = initial_data

            for step in self.steps:
                # Ensure the data pipeline passes data correctly between steps
                pipeline_data = step.execute(pipeline_data)  # Pass pipeline_data directly

            logger.info(
                f"## ================ {self.pipeline_name} pipeline completed successfully =======================")

            return pipeline_data  # Return the final pipeline data
        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise


def create_data_loading_step(name: str) -> PipelineStep:
    """Creates a pipeline step for loading data."""

    def step_function(pipeline_data: Dict[str, Any]):
        data_validation_config = pipeline_data['data_validation_config']  # Access from pipeline_data
        logger.info(f"Loading data from: {data_validation_config.data_dir}")

        if not os.path.exists(data_validation_config.data_dir):
            raise FileNotFoundError(f"Data file not found at {data_validation_config.data_dir}")

        data = pd.read_parquet(data_validation_config.data_dir)
        logger.info(f"Loaded DataFrame Shape: {data.shape}")

        pipeline_data['data'] = data  # Store the data in the pipeline data dictionary
        return pipeline_data

    return PipelineStep(name=name, step_function=step_function)


def create_data_validation_step(name: str) -> PipelineStep:
    """Creates a pipeline step for data validation."""

    def step_function(pipeline_data: Dict[str, Any]):
        data = pipeline_data['data']  # Access data from pipeline_data
        data_validation_config = pipeline_data['data_validation_config']  # Access from pipeline_data
        data_validation = pipeline_data['data_validation']  # Access from pipeline_data

        logger.info("Initializing Data Validation")
        logger.info("Starting data validation process")

        validation_status = data_validation.validate_all_columns()  # Changed method name

        if validation_status:
            logger.info("Data Validation Completed Successfully!")
        else:
            logger.warning("Data Validation Failed. Check the status file for more details.")

        pipeline_data['validation_status'] = validation_status  # Store in pipeline data
        return pipeline_data

    return PipelineStep(name=name, step_function=step_function)


if __name__ == "__main__":
    try:
        # Initialize Configuration Manager and Data Validation Config
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()

        # Initialize DataValidation here and pass it to the steps
        data_validation = DataValidation(config=data_validation_config)

        # Create and Run Data Validation Pipeline
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.add_step(create_data_loading_step("Load Data"))
        data_validation_pipeline.add_step(create_data_validation_step("Validate Data"))

        # Initial data for the pipeline
        initial_data = {
            'data_validation_config': data_validation_config,
            'data_validation': data_validation  # Pass the DataValidation object
        }

        # Run the pipeline and capture the results
        pipeline_results = data_validation_pipeline.run(initial_data=initial_data)

        # Access results from the returned pipeline data dictionary
        if pipeline_results.get("validation_status", False):
            print("Data validation successful!")
        else:
            print("Data validation failed.")


    except Exception as e:
        print(f"Error: {e}")