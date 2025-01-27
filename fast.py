from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
import pandas as pd 
import os 
import sys 
from dotenv import load_dotenv
from pathlib import Path

from src.lead_scoring.logger import logger
from src.lead_scoring.constants import PREDICTION_PIPELINE_CONFIG_FILEPATH
from src.lead_scoring.pipelines.pip_07_prediction_pipeline import ConfigurationManager, PredictionPipeline, InputDataHandler
from src.lead_scoring.exception import CustomException

load_dotenv()

# Initialize FastAPI 
app = FastAPI()

# Template configuration 
templates = Jinja2Templates(directory="templates")


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



try:
    config_manager = ConfigurationManager(PREDICTION_PIPELINE_CONFIG_FILEPATH)
    prediction_config = config_manager.get_prediction_pipeline_config()
    prediction_pipeline = PredictionPipeline(prediction_config)

except Exception as e:
    logger.error(f"Error initializing prediction pipeline: {e}")
    sys.exit(1)



@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/trigger_prediction")
async def trigger_prediction():
    try:
        # Fetch data from Mongo in batches 
        logger.info(f"Fetching data from MongoDB in batches")
        batch_size = 20
        input_collection = db[INPUT_COLLECTION]
        output_collection = db[OUTPUT_COLLECTION]

        total_documents = input_collection.count_documents({})
        logger.info(f"Total documents: {total_documents}")


        # Loop over the documents in batches 
        for skip in range(0, total_documents, batch_size):
            # Get data in batches 
            data_batch_cursor = input_collection.find({}, {"_id": 0}).skip(skip).limit(batch_size)
            data_batch = list(data_batch_cursor)

            # Check if data is available 
            if not data_batch:
                logger.info(f"No more data available for the collection")
                break

            logger.info(f"Fetching batch: {len(data_batch)} documents")

            # Create an empty list to store combined data predictions 
            combined_data = []

            # Prepare data for prediction 
            for document in data_batch:
                input_handler = InputDataHandler(**document)
                input_df = input_handler.get_data_as_df()

                # Make predictions 
                predictions = prediction_pipeline.predict(input_df)
                predictions = [int(pred) for pred in predictions]

                # Add predictions to data
                document['prediction'] = predictions[0]
                combined_data.append(document)

            # Store results in output collection 
            if combined_data:
                output_collection.insert_many(combined_data)
                logger.info(f"Stored {len(combined_data)} predictions in output collection")

        return Response(status_code=200, content="Data prediction triggered successfully")
    
    except CustomException as e:
        logger.error(f"Error processing data: {e}")
        return HTTPException(status_code=500, detail=str(e))
                