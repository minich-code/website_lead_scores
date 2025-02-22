import sys
sys.path.append('/home/western/ds_projects/website_lead_scores')

import pandas as pd
import joblib
import os
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.config_entity.config_params import *


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer_object(self) -> ColumnTransformer:
        logger.info("Creating transformer object")

        try:
            if not self.config.numerical_cols or not self.config.categorical_cols:
                raise ValueError("Numerical and categorical columns must be defined in the configuration.")

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
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
            logger.exception(f"Error creating transformer object: {str(e)}")
            raise CustomException(e, sys)

    def train_val_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        try:
            logger.info("Splitting data into train, validation, and test sets")

            if not os.path.exists(self.config.data_path):
                raise FileNotFoundError(f"Data path {self.config.data_path} does not exist.")
            df = pd.read_parquet(self.config.data_path, engine="pyarrow")

            if self.config.target_col not in df.columns:
                raise ValueError(f"Target column {self.config.target_col} not found in data.")

            X = df.drop(columns=[self.config.target_col])
            y = df[self.config.target_col]

            logger.info("Splitting data into train, validation, and test sets")

            # Split into training and temporary sets (70% train, 30% temp)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, stratify=y, random_state=self.config.random_state
            )

            # Split the temporary set into validation and test sets (50% each)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.config.random_state
            )

            logger.info("Saving the training, validation, and testing data in artifacts file")

            # Save the target variables for each set
            pd.DataFrame(y_train).to_parquet(self.config.root_dir / 'y_train.parquet', index=False)
            pd.DataFrame(y_val).to_parquet(self.config.root_dir / 'y_val.parquet', index=False)
            pd.DataFrame(y_test).to_parquet(self.config.root_dir / 'y_test.parquet', index=False)

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.exception(f"Error splitting data: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                                     X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
                                     y_test: pd.Series) -> TransformedData:
        try:
            logger.info("Initiating data transformation")

            # Get the preprocessor object
            preprocessor_obj = self.get_transformer_object()
            if not isinstance(preprocessor_obj, ColumnTransformer):
                raise TypeError("Expected a ColumnTransformer object")

            # Transform the training, testing, and validation data
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_val_transformed = preprocessor_obj.transform(X_val)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Save the preprocessing object to the artifacts file
            preprocessor_path = self.config.root_dir / 'preprocessor.joblib'
            joblib.dump(preprocessor_obj, preprocessor_path)  
            logger.info(f"Preprocessor saved to {preprocessor_path}")

            # Save the transformed data to the artifacts file
            X_train_transformed_path = self.config.root_dir / 'X_train_transformed.parquet'  
            X_val_transformed_path = self.config.root_dir / 'X_val_transformed.parquet'  
            X_test_transformed_path = self.config.root_dir / 'X_test_transformed.parquet'
            pd.DataFrame(X_train_transformed).to_parquet(X_train_transformed_path)  
            pd.DataFrame(X_val_transformed).to_parquet(X_val_transformed_path)  
            pd.DataFrame(X_test_transformed).to_parquet(X_test_transformed_path) 

            logger.info("Data transformation completed")

            return TransformedData(
                preprocessor=preprocessor_obj,
                X_train=pd.DataFrame(X_train_transformed),  
                X_val=pd.DataFrame(X_val_transformed),  
                X_test=pd.DataFrame(X_test_transformed),  
                y_train=y_train,
                y_val=y_val,
                y_test=y_test
            )

        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise CustomException(e, sys)

