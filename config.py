import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env automatically

# Folder paths
TEMP_FOLDER = "temp"
RESULT_FOLDER = "results"
LOG_FOLDER = "logs"
STATIC_FOLDER = "static"
MODELS_PATH = "models"

# Create folders if missing
for folder in [TEMP_FOLDER, RESULT_FOLDER, LOG_FOLDER, STATIC_FOLDER, MODELS_PATH]:
    os.makedirs(folder, exist_ok=True)

# Environment variables for Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_ACCESS_TOKEN = os.getenv("ADMIN_ACCESS_TOKEN", "default_admin")

# Model files in Supabase bucket
SUPABASE_MODELS = {
    "mosquito1": "mosquito_detection_1.pt",
    "mosquito2": "mosquito_detection_2.pt",
    "drone1": "drone_detection_1.pt",
    "drone2": "yolov8n.onnx",  # your ONNX model
}
