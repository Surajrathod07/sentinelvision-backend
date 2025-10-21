# api/image_routes.py
from fastapi import APIRouter, UploadFile, Form, HTTPException, BackgroundTasks
from starlette.concurrency import run_in_threadpool
from utils.file_handling import save_upload_file, remove_file
from utils.auth import verify_token
from utils.detection import detect_image as detect_image_sync

# Import MODELS and MODEL_LOCKS from models_store
from models_store import MODELS, MODEL_LOCKS

router = APIRouter()

@router.post("/detect/image/")
async def detect_image_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = None,
    model_name: str = Form(...),
    token: str = Form(...)
):
    # Authentication
    try:
        verify_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    # Check model
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    # Save uploaded file
    file_path = await save_upload_file(file)

    # Run detection safely with lock
    model = MODELS[model_name]
    lock = MODEL_LOCKS.get(model_name)

    def sync_wrapper(m, l, path):
        if l:
            with l:
                return detect_image_sync(m, path)
        else:
            return detect_image_sync(m, path)

    annotated_path = await run_in_threadpool(sync_wrapper, model, lock, file_path)

    # Cleanup uploaded file
    background_tasks.add_task(remove_file, file_path)

    # Return result
    return {"annotated_file": annotated_path}
