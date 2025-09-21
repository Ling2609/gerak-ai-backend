import boto3
import pandas as pd
from io import StringIO
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import json

def predict_from_s3_json(input_bucket, input_key, scenario, endpoint_name, 
                         output_bucket="crowd-predictions-demo-2025", s3_folder="predictions"):
    """
    Args:
        input_bucket (str): S3 bucket where input JSON is stored
        input_key (str): S3 key/path of the JSON file
        scenario (str): Scenario name ['general','entry_rush','mid_event','evacuation']
        endpoint_name (str): SageMaker endpoint name
        output_bucket (str): S3 bucket to save outputs
        s3_folder (str): Folder inside output bucket
    Returns:
        pd.DataFrame: Predictions + recommendations
        dict: Mappings for categorical columns
    """
    
    s3 = boto3.client("s3")
    
    response = s3.get_object(Bucket=input_bucket, Key=input_key)
    json_content = response['Body'].read().decode('utf-8')
    pdf_json = json.loads(json_content)

    rows = []
    for page in pdf_json.get("pages", []):
        row = {}
        for k, v in page.items():
            # Extract numeric if possible
            if isinstance(v, str) and ":" in v:
                try:
                    num = float(v.split(":")[1].strip())
                    row[k] = num
                except:
                    row[k] = 0
            elif isinstance(v, (int,float)):
                row[k] = v
            else:
                row[k] = v  # keep string
        row["Day_Hour"] = row.get("Day_Hour", 18)
        weather = page.get("Weather Severity","")
        if "Mild" in weather:
            row["Weather_Score"] = 0.5
        elif "Severe" in weather:
            row["Weather_Score"] = 1.0
        else:
            row["Weather_Score"] = 0.7
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    mappings = {}
    for col in df.columns:
        if df[col].dtype == object:
            unique_vals = sorted(df[col].dropna().unique())
            mapping = {val:i for i,val in enumerate(unique_vals)}
            df[col] = df[col].map(mapping)
            mappings[col] = mapping
    
    df_model_ready = df.copy()
    
    predictor = Predictor(endpoint_name=endpoint_name)
    predictor.serializer = CSVSerializer()
    predictions = predictor.predict(df_model_ready.values)
    if isinstance(predictions, bytes):
        predictions = predictions.decode("utf-8")
    
    risk_values = [float(x) for x in predictions.strip().split("\n")]
    df_model_ready["Congestion_Risk"] = risk_values

    def get_recommendation(risk, scenario):
        if scenario=="general":
            return "Safe to attend" if risk < 0.5 else "Moderate, consider early arrival"
        elif scenario=="entry_rush":
            return "Open extra gates"
        elif scenario=="mid_event":
            return "Redirect crowd to food/restroom areas"
        elif scenario=="evacuation":
            return "Activate emergency exits and guides"
        else:
            return "No recommendation"
    
    df_model_ready["Scenario"] = scenario
    df_model_ready["Recommendation"] = df_model_ready["Congestion_Risk"].apply(
        lambda r: get_recommendation(r, scenario)
    )

    csv_buffer = StringIO()
    df_model_ready.to_csv(csv_buffer, index=False)
    csv_key = f"{s3_folder}/prediction_{scenario}.csv"
    s3.put_object(Bucket=output_bucket, Key=csv_key, Body=csv_buffer.getvalue())
    
    plt.figure(figsize=(8,5))
    plt.bar(range(len(risk_values)), risk_values, color='skyblue')
    plt.xticks(range(len(risk_values)), [f"Row {i+1}" for i in range(len(risk_values))], rotation=45)
    plt.ylabel("Congestion Risk")
    plt.title(f"Scenario: {scenario}")
    plt.tight_layout()
    plt.savefig("/tmp/risk_plot.png")
    plt.close()
    s3.upload_file("/tmp/risk_plot.png", output_bucket, f"{s3_folder}/risk_plot_{scenario}.png")

    df_plot = df_model_ready.copy()
    df_plot["Expected_Attendance"] = df_plot.get("Capacity", df_plot.get("Expected_Attendance", [0]*len(df_plot)))
    fig = px.scatter(
        df_plot,
        x="Expected_Attendance",
        y="Congestion_Risk",
        color="Recommendation",
        hover_data=list(df_plot.columns),
        title=f"Crowd Congestion Risk - Scenario: {scenario}",
        size="Expected_Attendance",
        template="plotly_white"
    )
    html_file = f"/tmp/risk_plot_{scenario}.html"
    pio.write_html(fig, file=html_file, auto_open=False)
    s3.upload_file(html_file, output_bucket, f"{s3_folder}/risk_plot_{scenario}.html")
    
    print(f"✅ CSV saved: s3://{output_bucket}/{csv_key}")
    print(f"✅ Matplotlib plot saved: s3://{output_bucket}/{s3_folder}/risk_plot_{scenario}.png")
    print(f"✅ Interactive Plotly saved: s3://{output_bucket}/{s3_folder}/risk_plot_{scenario}.html")
    
    return df_model_ready, mappings
