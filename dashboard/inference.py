from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from torch_geometric.data import Data


DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
FEATURE_DIR = DASHBOARD_DIR / "feature_extraction "
FEATURE_SCRIPT = FEATURE_DIR / "single_image_feature_extractor.py"
FEATURE_REFINER_SCRIPT = FEATURE_DIR / "feature_refiner.py"
DEEP_FEATURE_CKPT = FEATURE_DIR / "best_multilevel.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fgvd_utils import SGCNModel, load_skeleton_edge_index  # noqa: E402


class DashboardDependencyError(RuntimeError):
    pass


@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class LevelPrediction:
    label: str
    confidence: float
    topk: list[tuple[str, float]]


@dataclass
class VehiclePrediction:
    detection: Detection
    l1: LevelPrediction
    l2: LevelPrediction
    l3: LevelPrediction


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _download_from_huggingface(repo_id: str, filename: str, cache_dir: Path | None = None) -> Path:
    """Download a file from Hugging Face Hub and cache it locally."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir) if cache_dir else None)
        return Path(path)
    except Exception as e:
        raise DashboardDependencyError(
            f"Failed to download {filename} from Hugging Face Hub ({repo_id}). "
            f"Please ensure the model repository is public and accessible. Error: {str(e)}"
        )


@lru_cache(maxsize=1)
def _feature_module():
    _require_file(FEATURE_SCRIPT)
    spec = importlib.util.spec_from_file_location("single_image_feature_extractor", FEATURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import feature extractor from {FEATURE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _feature_refiner_module():
    _require_file(FEATURE_REFINER_SCRIPT)
    spec = importlib.util.spec_from_file_location("feature_refiner", FEATURE_REFINER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import feature refiner from {FEATURE_REFINER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _edge_index() -> torch.Tensor:
    try:
        return load_skeleton_edge_index()
    except Exception:
        edges: list[tuple[int, int]] = []
        side = 64
        for r in range(side):
            for c in range(side):
                src = r * side + c
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < side and 0 <= cc < side:
                            edges.append((src, rr * side + cc))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _load_yolo(path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DashboardDependencyError(
            "ultralytics is not installed. Install dashboard requirements with "
            "`pip install -r dashboard/requirements.txt`."
        ) from exc
    return YOLO(str(_require_file(path)))


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def _clamp_box(box: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _normalize_feature_channels(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Match saved raw-feature scale: each non-RGB channel is in [0, 1]."""
    arr = np.asarray(arr, dtype=np.float32)
    mins = arr.min(axis=0, keepdims=True)
    maxs = arr.max(axis=0, keepdims=True)
    denom = maxs - mins
    out = np.zeros_like(arr, dtype=np.float32)
    np.divide(arr - mins, denom, out=out, where=denom > eps)
    return np.clip(out, 0.0, 1.0)


def _allowed_label_mask(labels: list[str], allowed_prefix: str | None) -> np.ndarray:
    mask = np.ones(len(labels), dtype=bool)
    if not allowed_prefix:
        return mask

    prefixes = [allowed_prefix]
    base = allowed_prefix[:-2] if allowed_prefix.endswith("::") else allowed_prefix
    parts = base.split("::")
    if len(parts) > 1:
        prefixes.append(f"{parts[0]}::")

    for prefix in prefixes:
        mask = np.array([str(label).startswith(prefix) for label in labels], dtype=bool)
        if not mask.any():
            continue
        return mask

    return np.ones(len(labels), dtype=bool)


def _topk_from_scores(
    labels: list[str],
    scores: np.ndarray,
    k: int = 3,
    allowed_prefix: str | None = None,
) -> LevelPrediction:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    mask = _allowed_label_mask(labels, allowed_prefix)
    candidate_idx = np.flatnonzero(mask)
    if candidate_idx.size == 0:
        candidate_idx = np.arange(len(scores))
    k = min(k, len(candidate_idx))
    local_idx = np.argsort(-scores[candidate_idx])[:k]
    idx = candidate_idx[local_idx]
    topk = [(labels[int(i)], float(scores[int(i)])) for i in idx]
    label, confidence = topk[0] if topk else ("unknown", 0.0)
    return LevelPrediction(label=label, confidence=confidence, topk=topk)


def _softmax_prediction(
    labels: list[str],
    logits: torch.Tensor,
    k: int = 3,
    allowed_prefix: str | None = None,
) -> LevelPrediction:
    probs = torch.softmax(logits.detach().cpu().float(), dim=-1).numpy().reshape(-1)
    return _topk_from_scores(labels, probs, k=k, allowed_prefix=allowed_prefix)


def _rf_prediction(
    bundle: dict,
    x: np.ndarray,
    k: int = 3,
    allowed_prefix: str | None = None,
) -> LevelPrediction:
    model = bundle["model"]
    pca = bundle.get("pca")
    labels = list(bundle["label_classes"])
    features = x.reshape(1, -1).astype(np.float32)
    if pca is not None:
        features = pca.transform(features).astype(np.float32)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[0]
    else:
        pred = int(model.predict(features)[0])
        scores = np.zeros(len(labels), dtype=np.float32)
        scores[pred] = 1.0
    return _topk_from_scores(labels, scores, k=k, allowed_prefix=allowed_prefix)


def _display_leaf(label: str, level: str) -> str:
    parts = str(label).split("::")
    if level == "L1":
        return parts[0]
    if level == "L2" and len(parts) >= 2:
        return parts[1]
    if level == "L3" and len(parts) >= 3:
        return parts[2]
    return str(label)


def _make_graph_data(x_np: np.ndarray, raw_rgb_np: np.ndarray, device: torch.device) -> Data:
    x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
    raw_x = torch.from_numpy(np.ascontiguousarray(raw_rgb_np)).float()
    data = Data(
        x=x,
        raw_x=raw_x,
        edge_index=_edge_index().clone(),
        batch=torch.zeros(x.shape[0], dtype=torch.long),
    )
    return data.to(device)


class SGCNPredictor:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        ckpt = torch.load(_require_file(checkpoint_path), map_location="cpu")
        sig = ckpt.get("run_signature", {})
        labels = list(ckpt.get("label_classes", []))
        if not labels:
            raise ValueError(f"Checkpoint has no label_classes: {checkpoint_path}")
        self.labels = labels
        self.level = str(sig.get("level", ""))
        self.feature_source = str(sig.get("feature_source", ""))
        self.device = device
        in_channels = self._infer_in_channels(ckpt)
        hidden_dim = int(sig.get("hidden_dim", 64))
        num_layers = int(sig.get("num_layers", 2))
        dropout = float(sig.get("dropout", 0.5))
        edge_sigma = float(sig.get("edge_sigma", 0.5))
        self.model = SGCNModel(in_channels, len(labels), hidden_dim, num_layers, dropout, edge_sigma)
        state = ckpt.get("best_state") or ckpt.get("model_state")
        self.model.load_state_dict(state)
        self.model.to(device).eval()

    @staticmethod
    def _infer_in_channels(ckpt: dict) -> int:
        state = ckpt.get("best_state") or ckpt.get("model_state")
        weight = state.get("layers.0.conv.lin.weight")
        if weight is None:
            sig = ckpt.get("run_signature") or {}
            if sig.get("feature_source") == "deep":
                return 64
            if sig.get("level") == "L1":
                return 8
            return 7
        return int(weight.shape[1])

    def predict(
        self,
        x_np: np.ndarray,
        raw_rgb_np: np.ndarray,
        allowed_prefix: str | None = None,
    ) -> LevelPrediction:
        data = _make_graph_data(x_np, raw_rgb_np, self.device)
        with torch.no_grad():
            logits = self.model(data)
        return _softmax_prediction(self.labels, logits[0], k=3, allowed_prefix=allowed_prefix)


class DeepFeatureExtractor:
    def __init__(self, device: torch.device):
        ckpt = torch.load(_require_file(DEEP_FEATURE_CKPT), map_location="cpu")
        self.device = device
        if int(ckpt.get("in_channels", 8)) != 8:
            raise ValueError(f"Expected 8-channel deep refiner checkpoint, got {ckpt.get('in_channels')}")
        refiner = _feature_refiner_module()
        self.model = refiner.MultiLevelDeepFeatureRefiner(
            num_classes=ckpt["num_classes"],
            D=int(ckpt.get("D", 64)),
        )
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.to(device).eval()

    def extract(self, raw8: np.ndarray) -> np.ndarray:
        grid = raw8.reshape(64, 64, raw8.shape[1]).transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(grid)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            feat = self.model(x, return_spatial=True)
        arr = feat.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return arr.reshape(64 * 64, -1).astype(np.float32)


class DashboardPipeline:
    def __init__(self, variant: str, load_detector: bool = True):
        if variant not in {"paper", "improved"}:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
        if load_detector:
            detector_name = "best_yolov8n.pt" if variant == "paper" else "best(m).pt"
            self.detector = _load_yolo(DASHBOARD_DIR / "detection" / detector_name)

        class_root = DASHBOARD_DIR / "classification" / ("paperbased" if variant == "paper" else "improved")
        if variant == "paper":
            self.l1 = SGCNPredictor(class_root / "L1.pt", self.device)
            self.l2 = SGCNPredictor(class_root / "L2.pt", self.device)
            self.l3 = SGCNPredictor(class_root / "L3.pt", self.device)
            self.deep_extractor = None
            self.rf_l2 = None
            self.rf_l3 = None
        else:
            l2_file = class_root / "L2.joblib"
            l3_file = class_root / "L3.joblib"
            if not l2_file.exists() or not l3_file.exists():
                try:
                    hf_repo_id = "anamikarajesh/FGVD-improved-models"
                    l2_file = _download_from_huggingface(hf_repo_id, "L2.joblib", cache_dir=class_root)
                    l3_file = _download_from_huggingface(hf_repo_id, "L3.joblib", cache_dir=class_root)
                except DashboardDependencyError:
                    raise DashboardDependencyError(
                        "The 'improved' model variant requires L2.joblib and L3.joblib files. "
                        "These are hosted on Hugging Face Hub but are not yet available. "
                        "Please use the 'Paper model' variant or set up your own Hugging Face repository. "
                        "See: https://huggingface.co for details."
                    )
            self.l1 = SGCNPredictor(class_root / "L1.pt", self.device)
            self.l2 = None
            self.l3 = None
            self.deep_extractor = DeepFeatureExtractor(self.device)
            self.rf_l2 = joblib.load(l2_file)
            self.rf_l3 = joblib.load(l3_file)

    def detect(self, image_rgb: np.ndarray, conf: float = 0.25, max_det: int = 10) -> list[Detection]:
        if self.detector is None:
            h, w = image_rgb.shape[:2]
            return [Detection((0, 0, w, h), 1.0, -1, "full_image")]
        result = self.detector.predict(source=image_rgb, conf=conf, max_det=max_det, verbose=False)[0]
        detections: list[Detection] = []
        names = getattr(self.detector, "names", {}) or {}
        if result.boxes is None:
            return detections
        h, w = image_rgb.shape[:2]
        for box in result.boxes:
            xyxy = _clamp_box(box.xyxy[0].detach().cpu().numpy(), w, h)
            cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
            det_conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            cls_name = str(names.get(cls_id, cls_id))
            detections.append(Detection(xyxy, det_conf, cls_id, cls_name))
        return detections

    def _extract_raw(self, image_rgb: np.ndarray, xyxy: tuple[int, int, int, int]):
        mod = _feature_module()
        image_bgr = _to_bgr(image_rgb)
        cropped_bgr = mod.crop_and_resize(image_bgr, *xyxy)
        rgb = mod.extract_rgb(cropped_bgr).astype(np.float32)
        gabor = _normalize_feature_channels(mod.extract_gabor(cropped_bgr))
        sobel = _normalize_feature_channels(mod.extract_sobel(cropped_bgr))
        raw8 = np.concatenate([rgb, gabor, sobel], axis=1).astype(np.float32)
        raw7 = np.concatenate([rgb, gabor], axis=1).astype(np.float32)
        crop_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        return raw8, raw7, rgb, crop_rgb

    def classify_detection(self, image_rgb: np.ndarray, detection: Detection) -> VehiclePrediction:
        raw8, raw7, rgb, _ = self._extract_raw(image_rgb, detection.xyxy)
        if self.variant == "paper":
            l1 = self.l1.predict(raw8, rgb)
            l2 = self.l2.predict(raw7, rgb, allowed_prefix=f"{l1.label}::")
            l3 = self.l3.predict(raw7, rgb, allowed_prefix=f"{l2.label}::")
        else:
            deep = self.deep_extractor.extract(raw8)
            pooled = deep.mean(axis=0, dtype=np.float32)
            l1 = self.l1.predict(deep, rgb)
            l2 = _rf_prediction(self.rf_l2, pooled, k=3, allowed_prefix=f"{l1.label}::")
            l3 = _rf_prediction(self.rf_l3, pooled, k=3, allowed_prefix=f"{l2.label}::")
        return VehiclePrediction(detection=detection, l1=l1, l2=l2, l3=l3)

    def predict_image(self, image_rgb: np.ndarray, conf: float = 0.25, max_det: int = 10) -> list[VehiclePrediction]:
        detections = self.detect(image_rgb, conf=conf, max_det=max_det)
        return [self.classify_detection(image_rgb, det) for det in detections]


def annotation_text(pred: VehiclePrediction) -> str:
    l1 = _display_leaf(pred.l1.label, "L1")
    l2 = _display_leaf(pred.l2.label, "L2")
    l3 = _display_leaf(pred.l3.label, "L3")
    return f"L1:{l1} | L2:{l2} | L3:{l3}"


def draw_annotations(image_rgb: np.ndarray, predictions: list[VehiclePrediction]) -> Image.Image:
    image = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["#e74c3c", "#2e86de", "#27ae60", "#8e44ad", "#d35400", "#16a085"]

    for i, pred in enumerate(predictions):
        x1, y1, x2, y2 = pred.detection.xyxy
        color = colors[i % len(colors)]
        text = annotation_text(pred)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        y_text = max(0, y1 - text_h - 6)
        draw.rectangle([x1, y_text, x1 + text_w + 8, y_text + text_h + 6], fill=color)
        draw.text((x1 + 4, y_text + 3), text, fill="white", font=font)
    return image


def predictions_to_rows(predictions: list[VehiclePrediction]) -> list[dict]:
    rows = []
    for idx, pred in enumerate(predictions, start=1):
        rows.append(
            {
                "vehicle": idx,
                "detector_label": pred.detection.class_name,
                "detector_conf": pred.detection.confidence,
                "bbox": pred.detection.xyxy,
                "L1": _display_leaf(pred.l1.label, "L1"),
                "L1_conf": pred.l1.confidence,
                "L2": _display_leaf(pred.l2.label, "L2"),
                "L2_conf": pred.l2.confidence,
                "L3": _display_leaf(pred.l3.label, "L3"),
                "L3_conf": pred.l3.confidence,
            }
        )
    return rows


def topk_text(pred: LevelPrediction, level: str) -> str:
    return ", ".join(f"{_display_leaf(label, level)} ({score:.2%})" for label, score in pred.topk)
