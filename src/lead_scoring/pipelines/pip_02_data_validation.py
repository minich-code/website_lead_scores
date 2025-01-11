
import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

from src.lead_scoring.components.c_02_data_validation import DataValidation, DataValidationConfig
from src.lead_scoring.logger import logger
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.exception import CustomException
from typing import List, Callable
import pandas as pd



PIPELINE_NAME = "DATA VALIDATION PIPELINE"


class PipelineStep:
    """
    Represents a step in the data validation pipeline.

    Attributes:
        name (str): The name of the pipeline step.
        step_function (Callable): The function to execute for this step.

    Methods:
        execute(**kwargs): Executes the pipeline step, logging its progress and handling exceptions.
    """

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


class DataValidationPipeline:
    """Orchestrates the data validation pipeline."""
    """
    Class to manage and execute a data validation pipeline.

    Attributes:
        pipeline_name (str): The name of the pipeline.
        steps (List[PipelineStep]): A list of steps to be executed in the pipeline.

    Methods:
        add_step(step: PipelineStep):
            Adds a step to the pipeline.

        run():
            Executes all steps in the data validation pipeline, logging the process and handling exceptions.
    """

    def __init__(self, pipeline_name: str = PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self):
        """Executes the data validation pipeline."""
        try:
             logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")

             config_manager = ConfigurationManager()
             data_validation_config = config_manager.get_data_validation_config()
          
             pipeline_data = {
              'data_validation_config':data_validation_config,
            }

             for step in self.steps:
                pipeline_data = step.execute(**pipeline_data)

             logger.info(f"## ================ {self.pipeline_name} pipeline completed successfully =======================")

        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise

def create_data_validation_step(name: str) -> PipelineStep:
    
    """
    Creates a pipeline step for data validation.

    Parameters:
    - name (str): The name of the pipeline step.

    Returns:
    - PipelineStep: An instance of PipelineStep configured to perform data validation.
    """
    def step_function(data_validation_config: DataValidationConfig):
            logger.info("Initializing Data Validation")
            data_validation = DataValidation(config=data_validation_config)
            data = pd.read_parquet(data_validation_config.data_dir)

            logger.info("Starting data validation process") 
            validation_status = data_validation.validate_data(data)

            if validation_status:
                logger.info("Data Validation Completed Successfully!")
            else:
                logger.warning("Data Validation Failed. Check the status file for more details.")

            return {} 
    return PipelineStep(name=name, step_function=step_function)


if __name__ == "__main__":
    try:
         data_validation_pipeline = DataValidationPipeline()
         data_validation_pipeline.add_step(create_data_validation_step("Validate Data"))
         data_validation_pipeline.run()
    except Exception as e:
        logger.error(f"Error during {PIPELINE_NAME} pipeline execution: {e}")
        raise CustomException(e, sys)