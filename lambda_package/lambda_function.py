import json
import os
from predict import predict_from_s3_json

# Optional: Environment variables for defaults
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "crowd-predictions-demo-2025")
S3_FOLDER = os.environ.get("S3_FOLDER", "predictions")
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "crowd-risk-option1-v2")

def lambda_handler(event, context):
    """
    Expects event to contain:
    {
        "input_bucket": "bucket-name",
        "input_key": "path/to/textract-output.json",
        "scenario": "general"  # or entry_rush, mid_event, evacuation
    }
    """
    try:
        input_bucket = event["input_bucket"]
        input_key = event["input_key"]
        scenario = event.get("scenario", "general").lower()
    except KeyError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Missing parameter: {str(e)}"})
        }

    try:
        df, mappings = predict_from_s3_json(
            input_bucket=input_bucket,
            input_key=input_key,
            scenario=scenario,
            endpoint_name=SAGEMAKER_ENDPOINT,
            output_bucket=OUTPUT_BUCKET,
            s3_folder=S3_FOLDER
        )

        response = {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Prediction completed successfully",
                "scenario": scenario,
                "predictions_csv": f"s3://{OUTPUT_BUCKET}/{S3_FOLDER}/prediction_{scenario}.csv",
                "matplotlib_plot": f"s3://{OUTPUT_BUCKET}/{S3_FOLDER}/risk_plot_{scenario}.png",
                "interactive_plot": f"s3://{OUTPUT_BUCKET}/{S3_FOLDER}/risk_plot_{scenario}.html",
                "mappings": mappings
            })
        }
        return response

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
