
import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

from src.lead_scoring.components.c_03_data_transformation import DataTransformation, DataTransformationConfig
from src.lead_scoring.logger import logger
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.exception import CustomException
from typing import List, Callable

PIPELINE_NAME = "DATA TRANSFORMATION PIPELINE"

class PipelineStep:
    """Represents a single step in the data transformation pipeline."""

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


class DataTransformationPipeline:
    """Orchestrates the data transformation pipeline."""

    def __init__(self, pipeline_name: str = PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self):
        """Executes the data transformation pipeline."""
        try:
            logger.info(f"## ============== Starting {self.pipeline_name} pipeline ====================")

            # Initialize configuration manager and load configs
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()

            # Pipeline data storage for passing data between steps
            pipeline_data = {
                'data_transformation_config': data_transformation_config,
            }

            # Execute each step in the pipeline sequentially
            for step in self.steps:
                pipeline_data = step.execute(**pipeline_data)

            logger.info(f"## ============ {self.pipeline_name} pipeline completed successfully =================")

        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during {self.pipeline_name} pipeline execution: {e}")
            raise CustomException(e, sys)


def create_data_transformation_step(name: str) -> PipelineStep:
    """Creates a pipeline step that performs the core data transformation."""

    def step_function(data_transformation_config: DataTransformationConfig):
        logger.info("Initializing Data Transformation")
        data_transformation = DataTransformation(config=data_transformation_config)

        # Train-Validation-Test Split
        X_train, X_val, X_test, y_train, y_val, y_test = data_transformation.train_val_test_split()
        logger.info("Data split into train, validation, and test sets")

        # Data Transformation
        (
            preprocessor,
            X_train_transformed,
            X_val_transformed,
            X_test_transformed,
            y_train,
            y_val,
            y_test,
        ) = data_transformation.initiate_data_transformation(X_train, X_val, X_test, y_train, y_val, y_test)
        logger.info("Data transformation applied")

        return {}

    return PipelineStep(name=name, step_function=step_function)


if __name__ == "__main__":
    try:
        # Instantiate the pipeline
        data_transformation_pipeline = DataTransformationPipeline()

        # Add transformation step
        data_transformation_pipeline.add_step(create_data_transformation_step("Transform Data"))

        # Run the pipeline
        data_transformation_pipeline.run()

    except Exception as e:
        logger.error(f"Error during {PIPELINE_NAME} pipeline execution: {e}")
        raise CustomException(e, sys)
