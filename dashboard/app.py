from __future__ import annotations

import importlib.util
import traceback

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from inference import (
        DashboardDependencyError,
        DashboardPipeline,
        draw_annotations,
        predictions_to_rows,
        topk_text,
    )
except Exception as e:
    st.error(f"Failed to import inference module: {str(e)}")
    st.error(traceback.format_exc())
    st.stop()


st.set_page_config(page_title="FGVD Vehicle Dashboard", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline(variant: str, load_detector: bool) -> DashboardPipeline:
    return DashboardPipeline(variant=variant, load_detector=load_detector)


def read_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def show_image(image, **kwargs) -> None:
    try:
        st.image(image, use_container_width=True, **kwargs)
    except TypeError:
        st.image(image, use_column_width=True, **kwargs)


st.title("FGVD Vehicle Classification Dashboard")
st.caption("Upload a road image, detect vehicle crops, and classify each crop at L1, L2, and L3.")

with st.sidebar:
    st.header("Pipeline")
    model_choice = st.radio(
        "Model option",
        ["Our best model", "Paper model"],
        index=1,
        help="Our best: YOLO best(m) + deep SGCN/RF. Paper: YOLOv8n + raw SGCN.",
    )
    variant = "improved" if model_choice == "Our best model" else "paper"
    yolo_available = importlib.util.find_spec("ultralytics") is not None
    st.markdown(
        "**Our best model** uses `best(m).pt`, deep SGCN for L1, and RF-deep for L2/L3.\n\n"
        "**Paper model** uses `best_yolov8n.pt` and raw SGCN checkpoints for L1/L2/L3."
    )
    if not yolo_available:
        st.warning("YOLO detection needs `ultralytics`. Whole-image mode is enabled for this environment.")
    conf = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
    max_det = st.slider("Maximum detections", 1, 30, 10, 1)
    full_image_mode = st.checkbox(
        "Classify whole image as one crop",
        value=not yolo_available,
        help="Useful for testing cropped vehicle images or when YOLO dependencies are unavailable.",
    )

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

image_rgb = read_image(uploaded)
left, right = st.columns([1, 1])
with left:
    st.subheader("Input")
    show_image(image_rgb)

run = st.button("Run detection and classification", type="primary")
if not run:
    st.stop()

try:
    with st.spinner("Loading checkpoints and running inference..."):
        pipeline = load_pipeline(variant, load_detector=not full_image_mode)
        predictions = pipeline.predict_image(image_rgb, conf=conf, max_det=max_det)
except DashboardDependencyError as exc:
    st.error(str(exc))
    st.code("pip install -r dashboard/requirements.txt", language="bash")
    st.stop()
except Exception as exc:
    st.exception(exc)
    st.stop()

if not predictions:
    st.warning("No vehicle detections found. Lower the confidence threshold or use whole-image mode.")
    st.stop()

annotated = draw_annotations(image_rgb, predictions)
with right:
    st.subheader("Annotated Output")
    show_image(annotated)

st.subheader("Predictions")
rows = predictions_to_rows(predictions)
df = pd.DataFrame(rows)
for col in ["detector_conf", "L1_conf", "L2_conf", "L3_conf"]:
    if col in df:
        df[col] = df[col].map(lambda x: f"{x:.2%}")
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Top-3 Details")
for idx, pred in enumerate(predictions, start=1):
    with st.expander(f"Vehicle {idx}: {pred.detection.xyxy}"):
        st.write(f"Detector: `{pred.detection.class_name}` ({pred.detection.confidence:.2%})")
        st.write(f"L1 top-3: {topk_text(pred.l1, 'L1')}")
        st.write(f"L2 top-3: {topk_text(pred.l2, 'L2')}")
        st.write(f"L3 top-3: {topk_text(pred.l3, 'L3')}")
