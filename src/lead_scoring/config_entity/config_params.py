from dataclasses import dataclass 
from pathlib import Path 
import pymongo
from pymongo import MongoClient
import pandas as pd 
import os 

# Data ingestion 
@dataclass
class DataIngestionConfig:
    config_data: dict 
