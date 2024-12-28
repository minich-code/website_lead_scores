import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

import pandas as pd
import numpy as np
import joblib 

from pathlib import Path 
from dataclasses import dataclass 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *


@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: list
    categorical_cols: list
    target_col : str


class ConfigurationManager:
    def __init__(self, data_preprocessing_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH):
        self.preprocessing_config = read_yaml(data_preprocessing_config)
        artifacts_root = self.preprocessing_config.artifacts_root
        create_directories([artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        logger.info("Getting data transformation configuration")

        transformation_config = self.preprocessing_config.data_transformation
        create_directories([transformation_config.root_dir])

        return DataTransformationConfig(
            root_dir = Path(transformation_config.root_dir),
            data_path = Path(transformation_config.data_path),
            numerical_cols = transformation_config.numerical_cols,
            categorical_cols = transformation_config.categorical_cols,
            target_col = transformation_config.target_col
        )
    

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config


    def get_transformer_object(self):  # Remove X_train and y_train
        logger.info("Creating transformer object")

        try:

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.config.numerical_cols),
                    ('cat', categorical_transformer, self.config.categorical_cols),

                ], remainder='passthrough'
            )

            return preprocessor
        except Exception as e:
            logger.error(f"Error creating transformer object: {str(e)}")
            raise CustomException(e, sys)
        

    def train_val_test_split(self):
        try:
            logger.info("Splitting data into train, validation and test sets")

            df = pd.read_parquet(self.config.data_path, engine="pyarrow")

            if self.config.target_col not in df.columns:
                raise ValueError(f"Target column {self.config.target_col} not found in data.")
            
            X = df.drop(columns=[self.config.target_col])
            y = df[self.config.target_col]


            logger.info("Splitting data into train and test sets")

            # Split into training and temporary sets (70% train, 30% temp)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

            # Split the temporary set into validation and test sets (50% each)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42) 

            logger.info(f"Saving the training, validation and testing data in artifacts file")

            # Save the target variables for each set 
            y_train.to_frame().to_parquet(self.config.root_dir / 'y_train.parquet', index=False)
            y_val.to_frame().to_parquet(self.config.root_dir / 'y_val.parquet', index=False)
            y_test.to_frame().to_parquet(self.config.root_dir / 'y_test.parquet', index=False)

            return X_train, X_val, X_test, y_train, y_val, y_test
        
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, X_train, X_val, X_test, y_train, y_val, y_test):
        try:
            logger.info("Initiating data transformation")

            # Get the preprocessor object
            preprocessor_obj = self.get_transformer_object() # Removed arguments here

            # Transform the training testing and validation data
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_val_transformed = preprocessor_obj.transform(X_val)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Save the preprocessing object to the artifacts file
            preprocessor_path = os.path.join(self.config.root_dir, 'preprocessor.joblib')
            save_object(obj=preprocessor_obj, file_path=preprocessor_path)

            # Save the transformed data to the artifacts file
            X_train_transformed_path = os.path.join(self.config.root_dir, 'X_train_transformed.joblib')
            X_val_transformed_path = os.path.join(self.config.root_dir, 'X_val_transformed.joblib')
            X_test_transformed_path = os.path.join(self.config.root_dir, 'X_test_transformed.joblib')

            joblib.dump(X_train_transformed, X_train_transformed_path)
            joblib.dump(X_val_transformed, X_val_transformed_path)
            joblib.dump(X_test_transformed, X_test_transformed_path)

            logger.info("Data transformation completed")

            return preprocessor_obj, X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config = data_transformation_config)
        X_train, X_val, X_test, y_train, y_val, y_test = data_transformation.train_val_test_split()
        preprocessor, X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test = \
            data_transformation.initiate_data_transformation(X_train, X_val, X_test, y_train, y_val, y_test) # Removed the argument

    except CustomException as e:
        logger.error(f"Error in data transformation: {str(e)}")
        sys.exit(1)