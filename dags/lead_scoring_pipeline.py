from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.utils.taskgroup import TaskGroup

from src.lead_scoring.logger import logger
from src.lead_scoring.exception import CustomException
from src.lead_scoring.pipelines.pip_01_data_ingestion import DataIngestionPipeline
from src.lead_scoring.pipelines.pip_02_data_validation import DataValidationPipeline
from src.lead_scoring.pipelines.pip_03_data_transformation import DataTransformationPipeline
from src.lead_scoring.pipelines.pip_04_model_trainer import ModelTrainerPipeline
from src.lead_scoring.pipelines.pip_05_model_evaluation import ModelEvaluationPipeline
from src.lead_scoring.pipelines.pip_06_model_validation import ModelValidationPipeline


# python callables for each pipeline
def run_data_ingestion(**kwargs):
    DataIngestionPipeline().run()


def run_data_validation(**kwargs):
    DataValidationPipeline().run()


def run_data_transformation(**kwargs):  
    DataTransformationPipeline().run()


def run_model_trainer(**kwargs):
    ModelTrainerPipeline().run()


def run_model_evaluation(**kwargs):
    ModelEvaluationPipeline().run()


def run_model_validation(**kwargs):
    ModelValidationPipeline().run()


with DAG(
    dag_id="lead_scoring_pipeline",
    description="Lead Scoring Pipeline",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_view="graph",
    orientation="LR",
    tags=["lead_scoring"],
    default_args={
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "retry_exponential_backoff": True,
        "email_on_failure": True,
        "email_on_retry": True,
        "email": ["minichworks@example.com"],
        "email_subject": "[Lead Scoring Pipeline] {params.execution_date}",
    },
) as dag:

    # Task to trigger data ingestion pipeline
    with TaskGroup("data_quality_checks") as data_quality_checks:
        data_validation_task = PythonOperator(
            task_id = "data_validation",
            python_callable = run_data_validation
        )

        data_transformation = PythonOperator(
            task_id = "data_transformation",
            python_callable = run_data_transformation
        )

    # Main tasks 
    data_ingestion_task = PythonOperator(
        task_id = "data_ingestion",
        python_callable = run_data_ingestion

    )

    model_trainer_task = PythonOperator(
        task_id = "model_trainer",
        python_callable = run_model_trainer
    )

    model_evaluation_task = PythonOperator(
        task_id = "model_evaluation",
        python_callable = run_model_evaluation
    )

    model_validation_task = PythonOperator(
        task_id = "model_validation",
        python_callable = run_model_validation
    )

    # Task dependencies 
    data_ingestion_task >> data_quality_checks >> model_trainer_task >> model_evaluation_task >> model_validation_task

    # Sensor to detect new data availability 
    data_sensor = ExternalTaskSensor(
        task_id = "data_sensor",
        external_dag_id = "data_availability_check",
        external_task_id = "check_new_data",
        timeout = 3600,
        poke_interval = 300, 
        mode = "poke",
    )

    # Incorporate sensor in the workflow 
    data_sensor >> data_ingestion_task
