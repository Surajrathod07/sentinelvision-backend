from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import onnxruntime as ort
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

app = FastAPI()

# --- CORS for frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Directory ---
MODEL_DIR = "models"
AVAILABLE_MODELS = [
    "mosquito_detection_1.pt",
    "mosquito_detection_2.pt",
    "drone_detection_1.pt",
    "yolov8n.onnx"
]

# --- Load Models ---
models = {}
for model_name in AVAILABLE_MODELS:
    path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(path):
        print(f"[⚠️] Missing model: {model_name}")
        continue

    try:
        if model_name.endswith(".onnx"):
            models[model_name] = ort.InferenceSession(path)
            print(f"[✅] Loaded ONNX model: {model_name}")
        else:
            models[model_name] = YOLO(path)
            print(f"[✅] Loaded YOLOv8 model: {model_name}")
    except Exception as e:
        print(f"[❌] Failed to load {model_name}: {e}")

# --- Request Schema ---
class PredictRequest(BaseModel):
    model_name: str
    image_base64: str

# --- Helper: Draw Boxes ---
def draw_boxes(image: Image.Image, detections, labels):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f"{labels[int(cls)]} {conf:.2f}" if labels else f"ID {int(cls)} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(y1 - 10, 0)), label, fill="red", font=font)
    return image

# --- Predict Endpoint ---
@app.post("/predict")
async def predict(req: PredictRequest):
    if req.model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name")

    image_bytes = base64.b64decode(req.image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    model = models[req.model_name]

    try:
        # --- For YOLOv8 (.pt) ---
        if req.model_name.endswith(".pt"):
            results = model(image)
            detections = results[0].boxes.data.cpu().numpy()
            labels = results[0].names

        # --- For ONNX ---
        elif req.model_name.endswith(".onnx"):
            sess = model
            input_name = sess.get_inputs()[0].name
            img_resized = image.resize((640, 640))
            img_np = np.array(img_resized).transpose(2, 0, 1)
            img_np = np.expand_dims(img_np, 0).astype(np.float32) / 255.0

            outputs = sess.run(None, {input_name: img_np})
            detections = outputs[0][0] if isinstance(outputs[0], list) else outputs[0]
            labels = None

        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")

        # --- Annotate Image ---
        annotated = draw_boxes(image.copy(), detections, labels)

        # --- Convert to base64 ---
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG")
        annotated_base64 = base64.b64encode(buf.getvalue()).decode()

        # --- Build response ---
        return {
            "model": req.model_name,
            "detections": len(detections),
            "annotated_image": annotated_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "🟢 SentinelVision API running with YOLOv8 + ONNX models!"}
