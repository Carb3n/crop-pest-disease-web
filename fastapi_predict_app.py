"""
FastAPI Prediction Server for AgroShield Crop Pest & Disease Detection
========================================================================
Wraps the YOLOv8s cotton pest detector and EfficientNetB0 plant disease
classifier (from crop-pest-disease-web) behind a single /predict endpoint
that matches the contract expected by the Node/Express website's
services/modelService.js (callExternalModel + normalizePrediction).

Contract:
  POST /predict
    multipart/form-data:
      image      (file, required)
      crop_type  (string, optional)

  Response JSON (any of these field names are recognized by modelService.js):
    {
      "name": "Tomato Early Blight",
      "type": "disease" | "pest" | "healthy",
      "confidence": 0.94,
      "severity": "moderate",           # best-effort, optional
      "description": "...",
      "model_version": "yolo-effnet-v1"
    }

Run locally:
    pip install -r requirements-api.txt
    uvicorn fastapi_predict_app:app --host 0.0.0.0 --port 8000

Then in the website's .env:
    MODEL_PROVIDER=external
    MODEL_API_URL=http://localhost:8000/predict
"""

import io
import logging

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agroshield-api")

app = FastAPI(title="AgroShield Prediction API", version="1.0.0")

# -----------------------------------------------------------------------
# Model paths (same as crop-pest-disease-web/app.py)
# -----------------------------------------------------------------------
PEST_MODEL_PATH = "models/pest/best.pt"
DISEASE_MODEL_PATH = "models/disease/plant_disease_efficientnet_v2_best.keras"
DISEASE_CLASSES_PATH = "models/disease/plant_disease_class_names_v2.npy"
DISEASE_IMG_SIZE = (224, 224)

# Confidence thresholds below which a prediction is not trusted enough
# to surface as the primary result.
PEST_CONF_THRESHOLD = 0.25
DISEASE_CONF_THRESHOLD = 0.40

# -----------------------------------------------------------------------
# Lazy-loaded model singletons (loaded once at first request / startup)
# -----------------------------------------------------------------------
_pest_model = None
_disease_model = None
_disease_class_names = None


def get_pest_model():
    global _pest_model
    if _pest_model is None:
        from ultralytics import YOLO
        logger.info("Loading YOLOv8s pest model...")
        _pest_model = YOLO(PEST_MODEL_PATH)
    return _pest_model


def get_disease_model():
    global _disease_model
    if _disease_model is None:
        import tensorflow as tf
        logger.info("Loading EfficientNetB0 disease model...")
        _disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    return _disease_model


def get_disease_class_names():
    global _disease_class_names
    if _disease_class_names is None:
        _disease_class_names = np.load(DISEASE_CLASSES_PATH, allow_pickle=True)
    return _disease_class_names


@app.on_event("startup")
def preload_models():
    # Preload both models at server startup so the first request isn't slow.
    get_pest_model()
    get_disease_model()
    get_disease_class_names()
    logger.info("All models loaded and ready.")


# -----------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------
def run_pest_detection(image: Image.Image, conf_threshold: float = PEST_CONF_THRESHOLD):
    model = get_pest_model()
    results = model.predict(image, conf=conf_threshold, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append((label, conf))

    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


def run_disease_classification(image: Image.Image, top_k: int = 3):
    import tensorflow as tf

    model = get_disease_model()
    class_names = get_disease_class_names()

    img = image.convert("RGB").resize(DISEASE_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    return [(str(class_names[i]), float(preds[i])) for i in top_indices]


def parse_disease_label(raw_label: str):
    """
    Disease class names look like 'Tomato___Late_blight' or
    'Apple___healthy'. Split into crop, condition, and derive a type.
    """
    crop, _, condition = raw_label.partition("___")
    condition_readable = condition.replace("_", " ").strip()
    is_healthy = "healthy" in condition.lower()
    name = f"{crop} Healthy" if is_healthy else f"{crop} {condition_readable}"
    pred_type = "healthy" if is_healthy else "disease"
    return name, pred_type, condition_readable


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "AgroShield Prediction API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    crop_type: str = Form(default=None),
):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.error(f"Failed to read uploaded image: {exc}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        # Run pest detection first.
        pest_detections = run_pest_detection(pil_image)
        top_pest = pest_detections[0] if pest_detections else None

        # Run disease classification.
        disease_preds = run_disease_classification(pil_image, top_k=3)
        top_disease_label, top_disease_conf = disease_preds[0]
        disease_name, disease_type, condition = parse_disease_label(top_disease_label)

        # Decide which result to surface as primary:
        # if a pest is detected above threshold, prefer it; otherwise use
        # the disease/healthy classification.
        if top_pest and top_pest[1] >= PEST_CONF_THRESHOLD:
            label, conf = top_pest
            result = {
                "name": label,
                "type": "pest",
                "confidence": round(conf, 4),
                "severity": "unknown",
                "description": f"Detected {label} via YOLOv8s cotton pest model.",
                "model_version": "yolov8s-pest-v1",
            }
        else:
            result = {
                "name": disease_name,
                "type": disease_type,
                "confidence": round(top_disease_conf, 4),
                "severity": "unknown",
                "description": (
                    f"Classified via EfficientNetB0 (48-class PlantVillage + Wheat "
                    f"taxonomy). Raw label: {top_disease_label}."
                ),
                "model_version": "efficientnetb0-disease-v2",
            }

        return JSONResponse(content=result)

    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
