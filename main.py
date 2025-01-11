
import sys
from src.lead_scoring.logger import logger
from src.lead_scoring.exception import CustomException
from src.lead_scoring.pipelines.pip_01_data_ingestion import DataIngestionPipeline
from src.lead_scoring.pipelines.pip_02_data_validation import DataValidationPipeline
from src.lead_scoring.pipelines.pip_03_data_transformation import DataTransformationPipeline
# from src.lead_scoring.pipelines.pip_04_model_training import ModelTrainingPipeline
from typing import List



class PipelineOrchestrator:
    """Orchestrates the execution of multiple pipelines."""

    def __init__(self):
        self.pipelines = []
    
    def add_pipeline(self, pipeline):
         """Adds a pipeline to the list of pipelines to be executed"""
         self.pipelines.append(pipeline)

    def run_all(self):
      """Executes all pipelines in sequence"""
      try:
        for pipeline in self.pipelines:
          pipeline.run()
      except Exception as e:
          logger.error(f"Error in orchestrator: {e}")
          raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logger.info("## ================ Starting Entire Model Pipeline =======================")
        orchestrator = PipelineOrchestrator()
        orchestrator.add_pipeline(DataIngestionPipeline())
        orchestrator.add_pipeline(DataValidationPipeline())
        orchestrator.add_pipeline(DataTransformationPipeline())
        # orchestrator.add_pipeline(ModelTrainingPipeline())
        orchestrator.run_all()
        logger.info("## ================ Entire Model Pipeline completed successfully =======================")
    except Exception as e:
        logger.error(f"Error during entire pipeline execution: {e}")
        raise CustomException(e, sys)