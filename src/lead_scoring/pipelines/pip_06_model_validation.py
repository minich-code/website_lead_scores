import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

from src.lead_scoring.logger import logger
from src.lead_scoring.exception import CustomException
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.components.c_05_model_evaluation import ModelEvaluation
from typing import Callable, List

PIPELINE_NAME = "MODEL VALIDATION PIPELINE"


class PipelineStep:
    """
    Represents a step in the model evaluation pipeline.

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


class ModelEvaluationPipeline:
    """
    Orchestrates the model evaluation pipeline.

    Attributes:
        pipeline_name (str): The name of the pipeline.
        steps (List[PipelineStep]): A list of steps to be executed in the pipeline.

    Methods:
        add_step(step: PipelineStep):
            Adds a step to the pipeline.

        run():
            Executes all steps in the model evaluation pipeline.
    """

    def __init__(self, pipeline_name: str = PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self):
        """Executes the model evaluation pipeline."""
        try:
            logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")

            config_manager = ConfigurationManager()
            model_evaluation_config = config_manager.get_model_evaluation_config()

            pipeline_data = {
                'model_evaluation_config': model_evaluation_config,
            }

            for step in self.steps:
                pipeline_data = step.execute(**pipeline_data)

            logger.info(f"## ================ {self.pipeline_name} pipeline completed successfully =======================")

        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise


def create_model_evaluation_step(name: str) -> PipelineStep:
    """
    Creates a pipeline step for model evaluation.

    Parameters:
    - name (str): The name of the pipeline step.

    Returns:
    - PipelineStep: An instance of PipelineStep configured for model evaluation.
    """
    def step_function(model_evaluation_config):
        logger.info("Initializing model evaluation")
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.run_evaluation()
        logger.info("Model validation completed successfully")
        return {}

    return PipelineStep(name=name, step_function=step_function)


if __name__ == "__main__":
    try:
        # Instantiate pipeline
        model_evaluation_pipeline = ModelEvaluationPipeline()

        # Add steps to pipeline
        model_evaluation_pipeline.add_step(create_model_evaluation_step("Evaluate Model"))

        # Run pipeline
        model_evaluation_pipeline.run()

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME}: {e}")
        sys.exit(1)
