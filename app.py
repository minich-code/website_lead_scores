from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import pandas as pd
import os
import sys
from dotenv import load_dotenv

from src.lead_scoring.logger import logger
from src.lead_scoring.constants import PREDICTION_PIPELINE_CONFIG_FILEPATH
from src.lead_scoring.pipelines.pip_07_prediction_pipeline import ConfigurationManager, PredictionPipeline, InputDataHandler
from src.lead_scoring.exception import CustomException

load_dotenv()

# Initialize flask app
app = Flask(__name__)

# MongoDB configuration
MONGO_URI = os.environ.get('MONGO_URI')
DB_NAME = 'leads'
INPUT_COLLECTION = "leads_to_predict"
OUTPUT_COLLECTION = "predicted_leads"

# Create MongoDB client
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]  # Access as an attribute, not callable
    print(f"Connected to database: {db.name}")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {e}")
    sys.exit(1)


# Load configuration and prediction pipeline
try:
    config_manager = ConfigurationManager(PREDICTION_PIPELINE_CONFIG_FILEPATH)
    prediction_config = config_manager.get_prediction_pipeline_config()
    prediction_pipeline = PredictionPipeline(prediction_config)
except Exception as e:
    logger.error(f"Error initializing prediction pipeline: {e}")
    sys.exit(1)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/trigger_prediction', methods=['POST'])
def trigger_prediction():
    try:
        # Fetch data from MongoDB in batches
        logger.info("Fetching data from MongoDB in batches")
        batch_size = 20
        input_collection = db[INPUT_COLLECTION]
        output_collection = db[OUTPUT_COLLECTION]

        total_documents = input_collection.count_documents({})
        logger.info(f"Total documents {total_documents}")

        # Loop over the documents in batches
        for skip in range(0, total_documents, batch_size):
            # Get data in batches
            data_batch_cursor = input_collection.find({}, {"_id": 0}).skip(skip).limit(batch_size)
            data_batch = list(data_batch_cursor)

            # Check if data is available
            if not data_batch:
                logger.info(f"No more data available for the collection")
                break

            logger.info(f" Fetch batch: {len(data_batch)} documents")

            # Create an empty list to store combined data predictions
            combined_data = []

            # prepare data for prediction
            for document in data_batch:
                input_handler = InputDataHandler(**document)
                input_df = input_handler.get_data_as_df()

                # Make predictions
                predictions = prediction_pipeline.make_predictions(input_df)
                predictions = [int(pred) for pred in predictions]

                # Add predictions to data
                document['prediction'] = predictions[0]
                combined_data.append(document)

            # Store results in output collection
            if combined_data:
                output_collection.insert_many(combined_data)
                logger.info(f"Stored {len(combined_data)} predictions in output collection")

        return jsonify({"message": "Data prediction triggered successfully"}), 200

    except CustomException as e:
        logger.error(f"Error processing data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)