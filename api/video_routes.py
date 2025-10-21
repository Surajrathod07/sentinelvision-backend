from fastapi import APIRouter, UploadFile, Form
from models_store import MODELS, MODEL_LOCKS

router = APIRouter()

@router.post("/process_video/")
async def process_video(file: UploadFile, model_name: str = Form(...)):
    if model_name not in MODELS:
        return {"error": "Model not found"}

    # Use thread lock for safe inference
    lock = MODEL_LOCKS[model_name]
    model = MODELS[model_name]

    with lock:
        results = model(file.file)  # YOLO inference

    return {"status": "success", "results": results.pandas().xyxy[0].to_dict(orient="records")}
