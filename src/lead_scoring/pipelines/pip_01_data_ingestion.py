
import sys
sys.path.append('/home/western/DS_Projects/website_lead_scores')

from src.lead_scoring.components.c_01_data_ingestion import DataIngestion, DataIngestionConfig
from src.lead_scoring.logger import logger
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.exception import CustomException
import sys
from typing import List, Callable
import logging

PIPELINE_NAME = "DATA INGESTION PIPELINE"


class PipelineStep:
    """Represents a single step in the data ingestion pipeline."""

    def __init__(self, name: str, step_function: Callable):
        self.name = name
        self.step_function = step_function

    def execute(self, **kwargs):
        """Executes the pipeline step."""
        try:
            logger.info(f"Executing step: {self.name}")
            result = self.step_function(**kwargs)
            logger.info(f"Step {self.name} completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error executing step {self.name}: {e}")
            raise CustomException(e, sys)


class DataIngestionPipeline:
    """Orchestrates the data ingestion pipeline."""

    def __init__(self, pipeline_name: str = PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self):
        """Executes the data ingestion pipeline."""
        try:
            logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")

            user_name = input("Enter your name: ")
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()

            # Initial data for steps
            pipeline_data = {
                 'data_ingestion_config':data_ingestion_config,
                 'user_name':user_name
            }

            for step in self.steps:
              pipeline_data = step.execute(**pipeline_data)

            logger.info(f"## ================ {self.pipeline_name} pipeline completed successfully =======================")
        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise

def create_data_ingestion_step(name: str) -> PipelineStep:
    """Creates a pipeline step that performs the core data ingestion from mongodb"""

    def step_function(data_ingestion_config:DataIngestionConfig, user_name: str):
            logger.info("Initializing Data Ingestion")
            data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
            data_ingestion.import_data_from_mongodb()
            return {} # Pass output of each step to the next one
    return PipelineStep(name=name, step_function=step_function)

if __name__ == "__main__":
    try:
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.add_step(create_data_ingestion_step("Import Data from MongoDB"))
        data_ingestion_pipeline.run()
    except Exception as e:
       logger.error(f"Error during {PIPELINE_NAME} pipeline execution: {e}")
       raise CustomException(e, sys)