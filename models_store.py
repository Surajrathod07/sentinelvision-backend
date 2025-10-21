from pathlib import Path
import os
import threading
from ultralytics import YOLO
from supabase import create_client, SupabaseException
from config import SUPABASE_MODELS, MODELS_PATH, SUPABASE_URL, SUPABASE_KEY

# Supabase client
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SupabaseException("SUPABASE_URL or SUPABASE_KEY not set. Please set them in .env or environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Globals
MODELS = {}
MODEL_LOCKS = {}

def download_model(bucket: str, filename: str, local_path: str) -> str:
    """Download model from Supabase if not exists locally"""
    if os.path.exists(local_path):
        print(f"[INFO] Model already exists: {local_path}")
        return local_path

    print(f"[INFO] Downloading {filename} from Supabase bucket '{bucket}'...")
    res = supabase.storage.from_(bucket).download(filename)
    if not res:
        raise FileNotFoundError(f"Failed to download {filename} from Supabase bucket {bucket}")
    Path(local_path).write_bytes(res)
    print(f"[✅] Model downloaded: {local_path}")
    return local_path

def load_models():
    """Load all models into memory with thread-safe locks"""
    for key, file in SUPABASE_MODELS.items():
        local_path = os.path.join(MODELS_PATH, file)
        download_model("models", file, local_path)

        # Detect file type and load accordingly
        if local_path.endswith(".onnx"):
            model = YOLO(local_path, task="detect")  # explicitly set task for ONNX
        else:
            model = YOLO(local_path)  # PyTorch YOLO model

        MODELS[key] = model
        MODEL_LOCKS[key] = threading.Lock()

    print("[✅] All models loaded successfully.")

# Load models on import
load_models()
