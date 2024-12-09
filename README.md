README: Lead Scoring Model for X Education
# Project Overview
This project aims to help X Education, an online education company, select the most promising leads that are most likely to convert into paying customers. The company wants to develop a Lead Scoring Model using a logistic regression approach to predict which leads are likely to convert and assign a lead score between 0 and 100. The model will help the company prioritize high-potential leads, allowing the sales team to focus on the leads with the highest chances of conversion.

# Problem Description
X Education sells online courses to industry professionals, attracting leads through various marketing channels like search engines and referrals. These leads visit the website, browse courses, and provide contact information (email or phone number) through forms. These leads are followed up by the sales team, who makes calls and sends emails to convert them into paying customers. Currently, the lead conversion rate is around 30%, but the CEO has set a target conversion rate of 80%.

The company needs a model to rank leads by their likelihood of converting. The Lead Scoring Model should assign a score based on various features, where a higher score indicates a higher likelihood of conversion.

# Goals
- Build a logistic regression model to predict lead conversion and assign a lead score between 0 and 100.
- Optimize the model to improve accuracy and conversion predictions based on past data.
- Adjust for future changes in requirements or thresholds, allowing flexibility for the company to update the model as needed.

# Steps Involved in the Project
## Data
Find data here https://www.kaggle.com/datasets/nagrajdesai/lead-score-case-study?select=Leads.csv 
## Data Preprocessing

## Data Cleaning: Removed missing values and irrelevant features.
1. Feature Selection: Selected relevant features based on domain knowledge and exploratory data analysis.
2. Data Scaling: Applied StandardScaler to scale the features, ensuring that the model treats all variables equally.
## Model Building

1. Gradient boosting Model: Used logistic regression to model the lead conversion probability, as it is well-suited for binary classification problems.
2. Train-Test Split: Split the dataset into training, validation, and test sets (70% train, 15% validation, 15% test) to evaluate model performance.
3. Model Training: Trained the model using the training data and evaluated its performance on the validation set.

## Model Evaluation

1. Precision-Recall and ROC Curves: Generated both the ROC and Precision-Recall curves to evaluate the model's ability to classify leads accurately.
2. AUC Calculation: Used the Area Under the Curve (AUC) for both ROC and Precision-Recall curves to quantify model performance.
3. Confusion Matrix & Metrics: Calculated key classification metrics such as Accuracy, Sensitivity (Recall), and Specificity.
4. Optimization of Thresholds: Tested different classification thresholds (e.g., 0.32, 0.36) to optimize the balance between precision and recall.

## Lead Scoring

1. Probability to Lead Score Mapping: Converted the predicted probabilities into a lead score between 0 and 100 by scaling and rounding the probabilities.
2. Predicted Labels: Applied cutoff thresholds (0.33, 0.36) to classify leads as likely to convert (1) or unlikely to convert (0).
3. Lead Scoring for the Test Set: After selecting the optimal cutoff threshold (0.36), the final predicted lead scores were calculated for the test set.

##Model Interpretation & Results

1. Lead Score Assignment: The final lead score was assigned to each lead based on the predicted probability.
2. Classification Report: Generated the classification report on the test set using the optimal cutoff to assess the model's performance with metrics like accuracy, precision, recall, and specificity.

# Next Steps
- Model Improvement: The model could be further improved using other machine learning algorithms such as Random Forests, Gradient Boosting.
- Hyperparameter Tuning: Fine-tuning the model using techniques like Grid Search or Random Search to find the best hyperparameters.
- Threshold Adjustment: Regularly adjust the classification thresholds to align with changes in business requirements or lead conversion targets.
- Continuous Monitoring: Continuously monitor the performance of the model and retrain it with new data to maintain prediction accuracy.

# Conclusion
The Lead Scoring Model developed for X Education will provide a data-driven approach to prioritize leads based on their likelihood of conversion. By assigning a lead score and classifying leads accordingly, the company can efficiently allocate resources to the most promising leads and increase its conversion rate.
