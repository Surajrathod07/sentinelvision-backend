# utils/file_handling.py
import os
import uuid
from pathlib import Path
from fastapi import UploadFile

UPLOAD_DIR = Path("/tmp/sentinel_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save UploadFile to a temporary path and return the path string.
    """
    suffix = Path(upload_file.filename).suffix or ""
    filename = f"{uuid.uuid4().hex}{suffix}"
    out_path = UPLOAD_DIR / filename

    # stream write
    with out_path.open("wb") as buffer:
        while True:
            chunk = await upload_file.read(1024*1024)
            if not chunk:
                break
            buffer.write(chunk)
    return str(out_path)

def remove_file(path: str):
    try:
        os.remove(path)
    except Exception:
        pass
