from dataclasses import dataclass 
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from sklearn.compose import ColumnTransformer


# Data ingestion 
@dataclass
class DataIngestionConfig:
    root_dir: str
    database_name: str
    collection_name: str
    batch_size: int
    mongo_uri: str

# Data Validation
@dataclass
class DataValidationConfig:
    root_dir: Path
    data_dir: Path
    val_status: Path
    all_schema: dict
    validated_data: Path
    profile_report_name: str

# Data Transformation

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: List[str]
    categorical_cols: List[str]
    target_col: str
    random_state: int

@dataclass
class TransformedData:
    preprocessor: ColumnTransformer
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_features_path: Path
    train_targets_path: Path
    model_name: str
    model_params: Dict[str, Any]
    project_name: str
    val_features_path: Path
    val_targets_path: Path


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    val_feature_path: Path
    val_targets_path: Path
    model_path: Path
    eval_scores_path: Path
    threshold_adjustment: Path
    precision_recall_path: Path


@dataclass
class ModelValidationConfig:
    root_dir: Path
    test_feature_path: Path
    test_targets_path: Path
    model_path: Path
    validation_scores_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path
    roc_curve_path: Path
    pr_curve_path: Path
    precision_recall_path: Path
    optimal_threshold: float
    project_name: str


