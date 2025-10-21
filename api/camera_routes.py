from fastapi import APIRouter, Form
from utils.detection import detect_camera
from utils.auth import verify_token

router = APIRouter()
from main import MODELS

@router.get("/camera/")
async def camera_endpoint(model_name: str = Form(...), token: str = Form(...)):
    verify_token(token)
    if model_name not in MODELS:
        return {"error": "Model not found"}
    detect_camera(MODELS[model_name])
    return {"status": "Camera stream ended"}
