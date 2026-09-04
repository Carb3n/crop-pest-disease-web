import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Crop Pest & Disease Detection",
    page_icon="\U0001F33E",
    layout="centered",
)

PEST_MODEL_PATH = "models/pest/best.pt"
DISEASE_MODEL_PATH = "models/disease/plant_disease_efficientnet_v2_best.keras"
DISEASE_CLASSES_PATH = "models/disease/plant_disease_class_names_v2.npy"
DISEASE_IMG_SIZE = (224, 224)


@st.cache_resource(show_spinner="Loading pest detection model...")
def load_pest_model():
    from ultralytics import YOLO
    return YOLO(PEST_MODEL_PATH)


@st.cache_resource(show_spinner="Loading disease classification model...")
def load_disease_model():
    import tensorflow as tf
    return tf.keras.models.load_model(DISEASE_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_disease_class_names():
    return np.load(DISEASE_CLASSES_PATH, allow_pickle=True)


def run_pest_detection(image: Image.Image, conf_threshold: float = 0.25):
    model = load_pest_model()
    results = model.predict(image, conf=conf_threshold, verbose=False)
    result = results[0]
    annotated = result.plot()  # BGR numpy array with boxes drawn
    annotated = annotated[:, :, ::-1]  # BGR -> RGB for st.image

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append((label, conf))

    return annotated, detections


def run_disease_classification(image: Image.Image, top_k: int = 3):
    """Classify plant disease using EfficientNetB0 with correct preprocessing."""
    import tensorflow as tf

    model = load_disease_model()
    class_names = load_disease_class_names()


    img = image.convert("RGB").resize(DISEASE_IMG_SIZE)


    arr = np.array(img, dtype=np.float32)


    arr = tf.keras.applications.efficientnet.preprocess_input(arr)


    arr = np.expand_dims(arr, axis=0)


    preds = model.predict(arr, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    return [(class_names[i], float(preds[i])) for i in top_indices]


st.title("\U0001F33E Crop Pest & Disease Detection")
st.caption(
    "SIH26131 \u2014 Government of Maharashtra | "
    "Cotton pest detection (YOLOv8s) + plant disease classification (EfficientNetB0)"
)

tab_pest, tab_disease = st.tabs(
    ["\U0001F41E Cotton Pest Detection", "\U0001F33F Plant Disease Classification"]
)

with tab_pest:
    st.subheader("Cotton Pest Detection")
    st.write("Upload a field image of cotton crops to detect and localize pests.")

    conf_threshold = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="pest_conf"
    )
    pest_file = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png"], key="pest_upload"
    )

    if pest_file is not None:
        image = Image.open(pest_file).convert("RGB")
        with st.spinner("Running detection..."):
            annotated, detections = run_pest_detection(image, conf_threshold)

        st.image(annotated, caption="Detections", width=700)

        if detections:
            st.write("**Detected pests:**")
            for label, conf in detections:
                st.write(f"- {label} \u2014 {conf:.1%} confidence")
        else:
            st.info("No pests detected above the confidence threshold.")

with tab_disease:
    st.subheader("Plant Disease Classification")
    st.write(
        "Upload a leaf image to classify the plant disease "
        "(48 classes, PlantVillage + Wheat taxonomy)."
    )

    disease_file = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png"], key="disease_upload"
    )

    if disease_file is not None:
        image = Image.open(disease_file).convert("RGB")
        st.image(image, caption="Uploaded image", width=700)

        with st.spinner("Classifying..."):
            top_preds = run_disease_classification(image, top_k=3)

        st.write("**Top predictions:**")
        for label, conf in top_preds:
            crop, _, condition = label.partition("___")
            condition = condition.replace("_", " ")
            st.write(f"- **{crop}** \u2014 {condition} ({conf:.1%})")

st.divider()
st.caption(
    "Built by Shivansh \u00b7 SIH26131 AI/ML pipeline \u00b7 "
    "YOLOv8s + EfficientNetB0"
)