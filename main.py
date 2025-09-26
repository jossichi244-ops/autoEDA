
from io import BytesIO
import json
from fastapi import FastAPI,File, Form, HTTPException, UploadFile, logger
from fastapi.responses import JSONResponse
from flask import app
import numpy as np
import pandas as pd
from modules.beyond_eda import beyond_eda
from modules.eda import analyze_column, clean_dataset, convert_numpy_types, descriptive_statistics, extract_eda_insights, generate_advanced_eda, generate_business_report, generate_relationships, generate_visualizations, infer_schema_from_df, inspect_dataset
from modules.prediction import auto_detect_target, detect_data_types, predict_from_df, read_file_to_df
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]

app = FastAPI(title="AutoEDA Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/parse-file")
async def parse_file(file: UploadFile = File(...)):
    name = file.filename.lower()
    content = await file.read()
    
    file_size_mb = len(content) / (1024 * 1024)
    logger.info(f"Received file: {file.filename} ({file_size_mb:.2f} MB)")

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(BytesIO(content), nrows=10000)
        elif name.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                df = pd.json_normalize(data[:10000])
            else:
                df = pd.json_normalize([data])
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(content), nrows=10000)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

    # ✅ Clean NaN/Inf → None
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.map(lambda x: None if isinstance(x, float) and np.isnan(x) else x)  

    schema = infer_schema_from_df(df)
    preview = df.head(10).to_dict(orient="records")
    understanding = []
    for col in df.columns:
        understanding.append(analyze_column(df[col]))
    
    # Step 2: Detailed Inspection
    inspection = inspect_dataset(df, max_sample=10)
    
    cleaned_result = clean_dataset(df, important_cols=[])
    preview = cleaned_result["cleaned_preview"]  

    descriptive = descriptive_statistics(df)

    visualizations = generate_visualizations(df)

    relationships = generate_relationships(df)

    advanced = generate_advanced_eda(df)
    print("parse_file: Running beyond_eda() for advanced analysis")
    try:
        advanced_analysis = beyond_eda(df)
    except Exception as e:
        advanced_analysis = {"error": str(e)}

    print("parse_file: Sending data to run_prediction for forecasting")
    prediction_result = await predict_from_df(df)
    print("parse_file: Received prediction result")
    result = {
        "schema": schema,
        "preview": preview,
        "understanding": understanding,
        "inspection": inspection,
        "cleaning": cleaned_result, 
        "descriptive": descriptive,
        "visualizations": visualizations,
        "relationships": relationships,
        "advanced": advanced,
        "advanced_analysis": advanced_analysis,
        "metadata": {
            "original_file_size_mb": round(file_size_mb, 2),
            "final_shape": df.shape,
            "sampled": file_size_mb > 10 
        }
    }

    insights = extract_eda_insights(result)

    result["insights"] = insights
    
    business_report = generate_business_report(result)

    result["business_report"] = business_report

    print("All columns:", df.columns.tolist())
    print("Numeric cols:", list(descriptive["numeric"].keys()))
    print("Categorical cols:", list(descriptive["categorical"].keys()))
    print("Unique counts per categorical column:")
    for col in list(descriptive["categorical"].keys()):
        print(f"  {col}: {df[col].nunique()} unique values")
    cleaned = convert_numpy_types(result)
    cleaned["prediction_result"] = prediction_result
    return JSONResponse(
        content=cleaned,
        media_type="application/json"
    )

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": "unknown", 
        "memory_usage_mb": "not tracked"
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "modules", "pipelineAutoML.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    pipeline_config = json.load(f)
    
@app.post("/api/prediction")
async def run_prediction(file: UploadFile = File(...), target: str = Form(None)):
    df = await read_file_to_df(file)

    # Detect types
    detected_types = detect_data_types(df, pipeline_config["data_type_detection"])

    # Determine target
    if target and target in df.columns:
        target_col = target
    else:
        target_col = auto_detect_target(df, detected_types)

    # Đánh dấu target
    if target_col:
        detected_types[target_col] = "target"

    # Run pipeline
    result = await predict_from_df(df, target_col=target_col)

    # Attach info
    result["target_col"] = target_col
    result["candidate_targets"] = [
        c for c in df.columns if "cluster" not in c.lower()
    ]
    return result

