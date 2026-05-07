from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import (
    DashboardDependencyError,
    DashboardPipeline,
    draw_annotations,
    predictions_to_rows,
    topk_text,
)


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
st.caption("Upload a vehicle image and classify it at L1, L2, and L3.")

with st.sidebar:
    st.header("Pipeline")
    model_choice = st.radio(
        "Model option",
        ["Our best model", "Paper model"],
        index=1,
        help="Our best: deep SGCN. Paper: raw SGCN.",
    )
    variant = "improved" if model_choice == "Our best model" else "paper"
    yolo_available = importlib.util.find_spec("ultralytics") is not None
    st.markdown(
        "**Our best model** uses deep SGCN checkpoints for L1/L2/L3.\n\n"
        "**Paper model** uses raw SGCN checkpoints for L1/L2/L3."
    )
    if yolo_available:
        st.caption("Optional detector mode is available in this environment.")
        conf = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
        max_det = st.slider("Maximum detections", 1, 30, 10, 1)
        full_image_mode = st.checkbox(
            "Classify whole image as one crop",
            value=False,
            help="Use this for cropped vehicle images.",
        )
    else:
        st.info("Detector mode is disabled for this deployment. Upload a cropped vehicle image for best results.")
        conf = 0.25
        max_det = 1
        full_image_mode = True

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

image_rgb = read_image(uploaded)
left, right = st.columns([1, 1])
with left:
    st.subheader("Input")
    show_image(image_rgb)

run_label = "Run classification" if full_image_mode else "Run detection and classification"
run = st.button(run_label, type="primary")
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
    st.warning("No vehicle detections found. Lower the confidence threshold or classify the whole image.")
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
