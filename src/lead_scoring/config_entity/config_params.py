from dataclasses import dataclass 
from pathlib import Path
from typing import Dict, List, Any
import pymongo
from pymongo import MongoClient
import pandas as pd 
import os 

# Data ingestion 
@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion process.

    """
    config_data: dict 

@dataclass
class DataValidationConfig:
    """
    Configuration for data validation process.

    """
    root_dir: Path
    val_status: str
    data_dir: Path
    all_schema: Dict[str, Any]
    critical_columns: List[str]