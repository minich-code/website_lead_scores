import sys
import time
sys.path.append('/home/western/ds_projects/website_lead_scores')

from src.lead_scoring.logger import logger
from src.lead_scoring.exception import CustomException
from src.lead_scoring.config_manager.config_settings import ConfigurationManager # import config manager
from src.lead_scoring.components.c_01_data_ingestion import DataIngestion
from typing import Callable, List, Dict, Any
from dataclasses import dataclass

PIPELINE_NAME = "DATA INGESTION PIPELINE"

@dataclass
class PipelineData:
    """Represents the data passed between pipeline steps."""
    data_ingestion_config: Any
    ingested_data: Any = None


class PipelineStep:
    """Represents a step in the data ingestion pipeline."""
    
    def __init__(self, step_function: Callable, name: str, config: Dict = None):
        self.step_function = step_function
        self.name = name  # Added name attribute
        self.config = config or {}

    def execute(self, pipeline_data: PipelineData) -> PipelineData:
        try:
            logger.info(f"Executing step: {self.name}")
            result = self.step_function(pipeline_data, **self.config)
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
        self.steps.append(step)

    def run(self):
        try:
            logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")

            # Fetch configurations
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()

            pipeline_data = PipelineData(
                data_ingestion_config=data_ingestion_config,
            )

            for step in self.steps:
                pipeline_data = step.execute(pipeline_data)

            logger.info(f"## ================ {self.pipeline_name} pipeline completed successfully =======================")

        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise


def create_data_ingestion_step(name: str) -> PipelineStep:
    """Creates a pipeline step for data ingestion with a retry mechanism."""

    def step_function(pipeline_data: PipelineData, **config) -> PipelineData:
        # Removed user_name from here and DataIngestion initialization
        logger.info("Initializing data ingestion")

        data_ingestion = DataIngestion(
            config=pipeline_data.data_ingestion_config,
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} - Fetching data from MongoDB...")
                data_ingestion.import_data_from_mongodb()
                logger.info("Data ingestion completed successfully.")
                break  # Exit loop if successful
            except Exception as e:
                logger.error(f"Data ingestion failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    raise CustomException(f"Data ingestion failed after {max_retries} attempts.", sys)

        return pipeline_data

    return PipelineStep(name=name, step_function=step_function)


if __name__ == "__main__":
    try:
        # Instantiate pipeline
        data_ingestion_pipeline = DataIngestionPipeline()

        # Add steps to pipeline
        data_ingestion_pipeline.add_step(create_data_ingestion_step("Ingest Data"))

        # Run pipeline
        data_ingestion_pipeline.run()

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME}: {e}")
        sys.exit(1)