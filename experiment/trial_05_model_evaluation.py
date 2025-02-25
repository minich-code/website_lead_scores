
import sys
sys.path.append("/home/western/DS_Projects/website_lead_scores")

from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, f1_score, precision_score, 
                             recall_score, roc_auc_score, accuracy_score)
import pandas as pd 
import numpy as np
import os 
import json 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    val_feature_path: Path
    val_targets_path: Path
    model_path: Path
    eval_scores_path: Path
    threshold_adjustment: Path
    precision_recall_path: Path


class ConfigurationManager:
    def __init__(self, model_evaluation_config: str = MODEL_EVALUATION_CONFIG_FILEPATH):
        self.evaluation_config = read_yaml(model_evaluation_config)
        artifacts_root = self.evaluation_config.artifacts_root
        create_directories([artifacts_root])

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        logger.info("Getting model evaluation configuration")

        eval_config = self.evaluation_config.model_evaluation
        create_directories([eval_config.root_dir])

        return ModelEvaluationConfig(
            root_dir = Path(eval_config.root_dir),
            val_feature_path = Path(eval_config.val_feature_path),
            val_targets_path = Path(eval_config.val_targets_path),
            model_path = Path(eval_config.model_path),
            eval_scores_path = Path(eval_config.eval_scores_path),
            threshold_adjustment = Path(eval_config.threshold_adjustment),
            precision_recall_path = Path(eval_config.precision_recall_path),
        )
    
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    @staticmethod
    def load_data(val_feature_path: Path, val_targets_path: Path):
        """Loads the validation data from the given file paths."""
        try:
            X_val_transformed = joblib.load(val_feature_path)
            y_val = pd.read_parquet(val_targets_path)
            logger.info("Validation data loaded successfully")
            return X_val_transformed, y_val
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def load_model(self, model_path: Path):
        """Loads the trained model."""
        try:
            model = joblib.load(str(model_path))
            logger.info("Model loaded successfully")

            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    def precision_recall_tradeoff(self, y_val, y_pred_proba):
        """Plots and saves the precision-recall tradeoff."""
        try:
            precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

            # Precision-recall tradeoff visualization
            plt.figure(figsize=(12, 8))
            plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
            plt.plot(thresholds, recall[:-1], label="Recall", color="green")
            plt.title("Precision-Recall Tradeoff Curve")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend()
            plt.grid()
            plt.axvline(x=0.43, color='red', linestyle='--', label='Cutoff: 0.32')
            plt.tight_layout()
            plt.savefig(self.config.precision_recall_path)
            plt.close()

            logger.info(f"Precision-recall tradeoff curve saved to {self.config.precision_recall_path}")
        except Exception as e:
            raise CustomException(f"Error plotting precision-recall tradeoff: {e}")

    
    def evaluate_model(self, model, X_val, y_val):
        """Evaluates the model on the validation set and adjusts threshold."""
        try:
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Threshold adjustment logic (unchanged)
            thresholds = np.linspace(0, 1, 101)
            metrics = []

            for threshold in thresholds:
                y_pred = (y_pred_proba > threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

                # Calculate metrics
                sensitivity = tp / (tp + fn)  # Recall
                specificity = tn / (tn + fp)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                metrics.append({
                    'Threshold': threshold,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'Accuracy': accuracy
                })

            metrics_df = pd.DataFrame(metrics)
            metrics_df['Diff_Sens_Spec'] = abs(metrics_df['Sensitivity'] - metrics_df['Specificity'])
            optimal_row = metrics_df.loc[metrics_df['Diff_Sens_Spec'].idxmin()]
            optimal_threshold = optimal_row['Threshold']

            plt.figure(figsize=(12, 8))
            plt.plot(metrics_df['Threshold'], metrics_df['Sensitivity'], label='Sensitivity', color='blue')
            plt.plot(metrics_df['Threshold'], metrics_df['Specificity'], label='Specificity', color='green')
            plt.plot(metrics_df['Threshold'], metrics_df['Accuracy'], label='Accuracy', color='red')
            plt.axvline(optimal_threshold, color='black', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
            plt.title('Sensitivity, Specificity, and Accuracy at Different Thresholds')
            plt.xlabel('Threshold')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid()
            plt.savefig(self.config.threshold_adjustment)
            plt.close()

            logger.info(f"Optimal Threshold: {optimal_threshold:.2f}")

            # Precision-Recall tradeoff visualization
            self.precision_recall_tradeoff(y_val, y_pred_proba)

            y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)

            scores = {
                "optimal_threshold": optimal_threshold,
                "f1_score": f1_score(y_val, y_pred_optimal),
                "precision": precision_score(y_val, y_pred_optimal),
                "recall": recall_score(y_val, y_pred_optimal),
                "accuracy": accuracy_score(y_val, y_pred_optimal)
            }

            save_json(path=Path(self.config.eval_scores_path), data=scores)

            logger.info("Model evaluated with threshold adjustment successfully")

            return y_pred_optimal, y_pred_proba, scores
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")

    def run_evaluation(self):
        """Runs the entire evaluation process."""
        try:
            X_val_transformed, y_val = self.load_data(
                self.config.val_feature_path, self.config.val_targets_path
            )
            model = self.load_model(self.config.model_path)
            
            y_pred, y_pred_proba, scores = self.evaluate_model(model, X_val_transformed, y_val)

            logger.info("Evaluation process completed successfully")
        except Exception as e:
            logger.error(f"Error in evaluation process: {str(e)}")

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.run_evaluation()
        logger.info("Model Evaluation Completed Successfully")
    except CustomException as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        sys.exit(1)


#----------------------------------------------------------Check this --------------------
# import sys
# sys.path.append("/home/western/DS_Projects/website_lead_scores")

# from pathlib import Path
# from dataclasses import dataclass
# from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
#                              precision_recall_curve, f1_score, precision_score, 
#                              recall_score, roc_auc_score, accuracy_score)
# import pandas as pd 
# import numpy as np
# import os 
# import json 
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib


# from src.lead_scoring.exception import CustomException
# from src.lead_scoring.logger import logger
# from src.lead_scoring.constants import *
# from src.lead_scoring.utils.commons import *


# @dataclass
# class ModelEvaluationConfig:
#     root_dir: Path
#     val_feature_path: Path
#     val_targets_path: Path
#     model_path: Path
#     eval_scores_path: Path
#     threshold_adjustment: Path
#     precision_recall_path: Path


# class ConfigurationManager:
#     def __init__(self, model_evaluation_config: str = MODEL_EVALUATION_CONFIG_FILEPATH):
#         self.evaluation_config = read_yaml(model_evaluation_config)
#         artifacts_root = self.evaluation_config.artifacts_root
#         create_directories([artifacts_root])

#     def get_model_evaluation_config(self) -> ModelEvaluationConfig:
#         logger.info("Getting model evaluation configuration")

#         eval_config = self.evaluation_config.model_evaluation
#         create_directories([eval_config.root_dir])

#         return ModelEvaluationConfig(
#             root_dir = Path(eval_config.root_dir),
#             val_feature_path = Path(eval_config.val_feature_path),
#             val_targets_path = Path(eval_config.val_targets_path),
#             model_path = Path(eval_config.model_path),
#             eval_scores_path = Path(eval_config.eval_scores_path),
#             threshold_adjustment = Path(eval_config.threshold_adjustment),
#             precision_recall_path = Path(eval_config.precision_recall_path),
#         )

# class ThresholdMetricsCalculator:
#     """Calculates metrics for different thresholds."""

#     @staticmethod
#     def calculate_metrics(y_true, y_pred_proba, thresholds):
#         metrics = []
#         for threshold in thresholds:
#             y_pred = (y_pred_proba > threshold).astype(int)
#             tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#             # Calculate metrics
#             sensitivity = tp / (tp + fn)  # Recall
#             specificity = tn / (tn + fp)
#             accuracy = (tp + tn) / (tp + tn + fp + fn)

#             metrics.append({
#                 'Threshold': threshold,
#                 'Sensitivity': sensitivity,
#                 'Specificity': specificity,
#                 'Accuracy': accuracy
#             })
#         return metrics

# class ThresholdPlotter:
#     """Plots threshold-based metrics."""

#     def __init__(self, output_path):
#         self.output_path = output_path

#     def plot_threshold_metrics(self, metrics_df, optimal_threshold):
#         try:
#             plt.figure(figsize=(12, 8))
#             plt.plot(metrics_df['Threshold'], metrics_df['Sensitivity'], label='Sensitivity', color='blue')
#             plt.plot(metrics_df['Threshold'], metrics_df['Specificity'], label='Specificity', color='green')
#             plt.plot(metrics_df['Threshold'], metrics_df['Accuracy'], label='Accuracy', color='red')
#             plt.axvline(optimal_threshold, color='black', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
#             plt.title('Sensitivity, Specificity, and Accuracy at Different Thresholds')
#             plt.xlabel('Threshold')
#             plt.ylabel('Metric Value')
#             plt.legend()
#             plt.grid()
#             plt.savefig(self.output_path)
#             plt.close()
#         except Exception as e:
#              raise CustomException(f"Error plotting threshold metrics: {e}")

# class ModelEvaluation:
#     def __init__(self, config: ModelEvaluationConfig):
#         self.config = config
#         self.threshold_metrics_calculator = ThresholdMetricsCalculator()
#         self.threshold_plotter = ThresholdPlotter(self.config.threshold_adjustment)

#     @staticmethod
#     def load_data(val_feature_path: Path, val_targets_path: Path):
#         """Loads the validation data from the given file paths."""
#         try:
#             X_val_transformed = joblib.load(val_feature_path)
#             y_val = pd.read_parquet(val_targets_path)
#             logger.info("Validation data loaded successfully")
#             return X_val_transformed, y_val
#         except Exception as e:
#             logger.error(f"Error loading data: {e}")
#             raise CustomException(e, sys)

#     @staticmethod
#     def load_model(model_path: Path):
#         """Loads the trained model."""
#         try:
#             model = joblib.load(str(model_path))
#             logger.info("Model loaded successfully")

#             return model
#         except Exception as e:
#             logger.error(f"Error loading model: {str(e)}")
#             raise CustomException(e, sys)
#     def precision_recall_tradeoff(self, y_val, y_pred_proba):
#         """Plots and saves the precision-recall tradeoff."""
#         try:
#             precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

#             # Precision-recall tradeoff visualization
#             plt.figure(figsize=(12, 8))
#             plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
#             plt.plot(thresholds, recall[:-1], label="Recall", color="green")
#             plt.title("Precision-Recall Tradeoff Curve")
#             plt.xlabel("Threshold")
#             plt.ylabel("Score")
#             plt.legend()
#             plt.grid()
#             plt.axvline(x=0.43, color='red', linestyle='--', label='Cutoff: 0.32')
#             plt.tight_layout()
#             plt.savefig(self.config.precision_recall_path)
#             plt.close()

#             logger.info(f"Precision-recall tradeoff curve saved to {self.config.precision_recall_path}")
#         except Exception as e:
#             raise CustomException(f"Error plotting precision-recall tradeoff: {e}")

#     def evaluate_model(self, model, X_val, y_val):
#         """Evaluates the model on the validation set and adjusts threshold."""
#         try:
#             y_pred_proba = model.predict_proba(X_val)[:, 1]
#             thresholds = np.linspace(0, 1, 101)
#             metrics = self.threshold_metrics_calculator.calculate_metrics(y_val, y_pred_proba, thresholds)
#             metrics_df = pd.DataFrame(metrics)
#             metrics_df['Diff_Sens_Spec'] = abs(metrics_df['Sensitivity'] - metrics_df['Specificity'])
#             optimal_row = metrics_df.loc[metrics_df['Diff_Sens_Spec'].idxmin()]
#             optimal_threshold = optimal_row['Threshold']
#             self.threshold_plotter.plot_threshold_metrics(metrics_df, optimal_threshold)

#             logger.info(f"Optimal Threshold: {optimal_threshold:.2f}")

#             # Precision-Recall tradeoff visualization
#             self.precision_recall_tradeoff(y_val, y_pred_proba)

#             y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)

#             scores = {
#                 "optimal_threshold": optimal_threshold,
#                 "f1_score": f1_score(y_val, y_pred_optimal),
#                 "precision": precision_score(y_val, y_pred_optimal),
#                 "recall": recall_score(y_val, y_pred_optimal),
#                 "accuracy": accuracy_score(y_val, y_pred_optimal)
#             }

#             save_json(path=Path(self.config.eval_scores_path), data=scores)

#             logger.info("Model evaluated with threshold adjustment successfully")

#             return y_pred_optimal, y_pred_proba, scores
#         except Exception as e:
#             logger.error(f"Error evaluating model: {str(e)}")
#             raise CustomException(e, sys)

#     def run_evaluation(self):
#         """Runs the entire evaluation process."""
#         try:
#             X_val_transformed, y_val = self.load_data(
#                 self.config.val_feature_path, self.config.val_targets_path
#             )
#             model = self.load_model(self.config.model_path)
            
#             y_pred, y_pred_proba, scores = self.evaluate_model(model, X_val_transformed, y_val)

#             logger.info("Evaluation process completed successfully")
#         except Exception as e:
#             logger.error(f"Error in evaluation process: {str(e)}")
#             raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         model_evaluation_config = config_manager.get_model_evaluation_config()
#         model_evaluation = ModelEvaluation(config=model_evaluation_config)
#         model_evaluation.run_evaluation()
#         logger.info("Model Evaluation Completed Successfully")
#     except CustomException as e:
#         logger.error(f"Error in model evaluation: {str(e)}")
#         sys.exit(1)