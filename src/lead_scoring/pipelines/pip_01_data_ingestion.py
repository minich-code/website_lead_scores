
import sys
sys.path.append('/home/western/DS_Projects/website_lead_scores')


import sys
import logging
from typing import List, Callable, Dict
from src.lead_scoring.components.c_01_data_ingestion import DataIngestion, DataIngestionConfig
from src.lead_scoring.logger import logger
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.exception import CustomException

PIPELINE_NAME = "DATA INGESTION PIPELINE"

class PipelineStep:
    """Represents a single step in the data ingestion pipeline."""

    def __init__(self, name: str, step_function: Callable, logger: logging.Logger):
        self.name = name
        self.step_function = step_function
        self.logger = logger

    def execute(self, **kwargs) -> Dict:
        """Executes the pipeline step."""
        try:
            self.logger.info(f"Executing step: {self.name}")
            result = self.step_function(**kwargs)
            self.logger.info(f"Step {self.name} completed successfully.")
            return result
        except Exception as e:
            self.logger.error(f"Error executing step {self.name}: {e}")
            raise CustomException(e, sys)


class DataIngestionPipeline:
    """Orchestrates the data ingestion pipeline."""

    def __init__(self, pipeline_name: str, config: dict, logger: logging.Logger):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []
        self.config = config
        self.logger = logger

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self, user_name: str):
        """Executes the data ingestion pipeline."""
        try:
            self.logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")
            pipeline_data = {"config": self.config, "user_name": user_name}

            for step in self.steps:
                pipeline_data.update(step.execute(**pipeline_data))

            self.logger.info(f"## ================ {self.pipeline_name} pipeline completed successfully =======================")
        except CustomException as e:
            self.logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise



def create_data_ingestion_step(name: str, logger: logging.Logger) -> PipelineStep:
    """Creates a pipeline step that performs the core data ingestion from MongoDB."""

    def step_function(config: dict, user_name: str):
        logger.info("Initializing Data Ingestion")
        data_ingestion_config = config["data_ingestion_config"]
        data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
        ingested_data = data_ingestion.import_data_from_mongodb()
        return {"ingested_data": ingested_data}

    return PipelineStep(name=name, step_function=step_function, logger=logger)


if __name__ == "__main__":
    try:
        # Initialize Configuration and Logger
        config_manager = ConfigurationManager()
        user_name = input("Enter your name: ") 
        config = {
            "data_ingestion_config": config_manager.get_data_ingestion_config()
        }

        # Create and Run Pipeline
        pipeline_logger = logger  # Use centralized logger instance
        data_ingestion_pipeline = DataIngestionPipeline(pipeline_name=PIPELINE_NAME, config=config, logger=pipeline_logger)

        # Add steps dynamically
        data_ingestion_pipeline.add_step(create_data_ingestion_step("Import Data from MongoDB", pipeline_logger))

        # Run the pipeline
        data_ingestion_pipeline.run(user_name=user_name)  # Pass user_name as an argument to the pipeline
     

    except Exception as e:
        logger.error(f"Error during {PIPELINE_NAME} pipeline execution: {e}")
        raise CustomException(e, sys)
