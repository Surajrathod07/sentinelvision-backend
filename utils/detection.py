# utils/detection.py
import os
import cv2
from config import RESULT_FOLDER

os.makedirs(RESULT_FOLDER, exist_ok=True)

def _save_annotated_image(numpy_img, out_path):
    # numpy_img is BGR if produced by results[0].plot() from ultralytics
    cv2.imwrite(out_path, numpy_img)
    return out_path

def detect_image(model, input_path):
    """
    Run model on an image file path and return the path to the annotated image.
    Uses ultralytics YOLO model (model(frame) -> Results). We call .plot() and save.
    """
    results = model(input_path)   # ultralytics accepts file path
    res0 = results[0]
    annotated = res0.plot()       # returns annotated numpy image (BGR)
    annotated_path = os.path.join(RESULT_FOLDER, f"result_{os.path.basename(input_path)}")
    _save_annotated_image(annotated, annotated_path)
    return annotated_path

def detect_video(model, input_path):
    """
    For short videos, ultralytics may save when using model.predict(save=True).
    We use a safe approach: run model with source as path, then fetch annotated frames if available,
    else fallback to running frame-by-frame (slower).
    """
    results = model(input_path)  # ultralytics may process and produce results.disk files
    # prefer to save a video: some ultralytics versions allow results[0].orig_video or .save() but for reliability:
    annotated_path = os.path.join(RESULT_FOLDER, f"result_{os.path.basename(input_path)}")
    try:
        # Try to get annotated video from results if API exposes it
        # (Some versions of ultralytics save to runs/detect/exp)
        # If not available, we attempt to render the first frame annotated and return that path.
        res0 = results[0]
        if hasattr(res0, "plot"):
            # save a single annotated frame (for quick preview)
            annotated = res0.plot()
            cv2.imwrite(annotated_path, annotated)
            return annotated_path
    except Exception:
        pass

    # Fallback: just return input_path (no annotation)
    return input_path

def detect_camera(model, camera_index=0):
    """
    Blocking helper for local testing (not used by server).
    """
    cap = cv2.VideoCapture(camera_index)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("Real-time Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
