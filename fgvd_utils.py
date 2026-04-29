"""
fgvd_utils.py
=============
Shared utilities for FGVD vehicle classification experiments.

Public surface
--------------
Data / labels:
    load_metadata, build_level_labels, apply_case, parent_label_for,
    check_group_leakage

Tail merging:
    tail_merge

Feature loading:
    load_raw_stacked, load_deep_features, pool_deep,
    LEVEL_RAW_FEATURES

Graph helpers:
    load_skeleton_edge_index, compute_gaussian_edge_weights

Datasets:
    FGVDGraphDataset, FGVDPooledDataset

Models:
    SGCNModel, GATModel, DeepMLP

Hierarchy:
    build_parent_index, MaskedHierarchicalCE, hierarchical_predict

Long-tail losses:
    LogitAdjustedCE, FocalLoss, compute_class_balanced_weights

Training:
    run_epoch, fit_with_resume, evaluate

Metrics:
    top_k_accuracy, stratified_accuracy_by_support, per_parent_macro_f1,
    classification_summary

Plotting / artifacts:
    plot_learning_curves, save_run_artifacts

High-level entry point:
    run_experiment
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


# ============================================================
# Constants
# ============================================================

ROOT = Path("FGVD_Graph_Handover")
METADATA_PATH = ROOT / "metadata.csv"
RAW_FEAT_ROOT = ROOT / "raw_features"
DEEP_FEAT_ROOT = ROOT / "deep_features"
SKELETON_PATH = ROOT / "master_grid_skeleton.npz"
LEGACY_ADJ_PATH = ROOT / "master_grid_adj.npz"

DEFAULT_DEEP_SUBDIR = "multilevel"
DEFAULT_EDGE_SIGMA = 0.5
LEVEL_RAW_FEATURES = {
    "L1": ("rgb", "gabor", "sobel"),
    "L2": ("rgb", "gabor"),
    "L3": ("rgb", "gabor"),
}

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data / labels
# ============================================================

TW_SET = {"motorcycle", "scooter"}
THW_SET = {"autorickshaw", "auto", "threewheeler", "three_wheeler"}


def _norm_l1(x: str) -> str:
    return str(x).strip().lower().replace("-", "").replace(" ", "")


def load_metadata(path: Path = METADATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["vehicle_id", "split", "L1", "L2", "L3", "image_stem"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def build_level_labels(df: pd.DataFrame, level: str) -> np.ndarray:
    """Hierarchical string label: L1, L1::L2, or L1::L2::L3."""
    if level == "L1":
        return df["L1"].to_numpy(dtype=object)
    if level == "L2":
        return (df["L1"] + "::" + df["L2"]).to_numpy(dtype=object)
    if level == "L3":
        return (df["L1"] + "::" + df["L2"] + "::" + df["L3"]).to_numpy(dtype=object)
    raise ValueError("level must be L1/L2/L3")


def parent_label_for(label: str, level: str) -> str:
    """Return the parent label for a hierarchical string label.
    L2 label 'car::Honda' -> 'car'. L3 'car::Honda::City' -> 'car::Honda'."""
    if level in ("L1",):
        return label
    parts = label.split("::")
    if level == "L2":
        return parts[0]
    if level == "L3":
        return "::".join(parts[:2])
    raise ValueError(f"Unknown level {level}")


def apply_case(
    df: pd.DataFrame, base_labels: np.ndarray, level: str, case_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one of the binary/case-collapse modes from the original notebooks."""
    l1_norm = df["L1"].map(_norm_l1).to_numpy()
    is_tw = np.isin(l1_norm, list(TW_SET))
    is_thw = np.isin(l1_norm, list(THW_SET))
    is_car = l1_norm == "car"

    keep_all = np.ones(len(df), dtype=bool)

    if case_name == "all":
        return base_labels, keep_all
    if case_name == "tw_vs_car":
        keep = is_tw | is_car
        return base_labels[keep], keep
    if case_name == "tw_vs_all":
        y = np.where(is_tw, "two_wheeler" if level == "L1" else base_labels, "other")
        return y, keep_all
    if case_name == "thw_vs_all":
        y = np.where(is_thw, "three_wheeler" if level == "L1" else base_labels, "other")
        return y, keep_all
    if case_name == "tw_thw_vs_all":
        group = is_tw | is_thw
        y = np.where(group, "two_or_three_wheeler" if level == "L1" else base_labels, "other")
        return y, keep_all
    raise ValueError(f"Unknown case_name: {case_name}")


def check_group_leakage(df: pd.DataFrame, group_col: str = "image_stem") -> dict[str, int]:
    splits = ["train", "val", "test"]
    groups = {s: set(df[df.split == s][group_col]) for s in splits}
    return {
        "train_val": len(groups["train"] & groups["val"]),
        "train_test": len(groups["train"] & groups["test"]),
        "val_test": len(groups["val"] & groups["test"]),
        "n_train": len(groups["train"]),
        "n_val": len(groups["val"]),
        "n_test": len(groups["test"]),
    }


# ============================================================
# Tail merging
# ============================================================

def tail_merge(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_min: int,
    level: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Merge classes with <n_min train samples into '<parent>::other'.
    Returns merged label arrays + report dict (orig_classes, kept_classes,
    coverage_train/val/test).
    """
    counts = Counter(y_train)
    kept_set = {c for c, n in counts.items() if n >= n_min}

    def _remap_one(lbl: str) -> str:
        if lbl in kept_set:
            return lbl
        parent = parent_label_for(lbl, level)
        return f"{parent}::other"

    f = np.vectorize(_remap_one, otypes=[object])
    y_tr_m = f(y_train)
    y_va_m = f(y_val)
    y_te_m = f(y_test)

    report = {
        "n_min": n_min,
        "orig_classes": int(len(counts)),
        "kept_classes": int(len(kept_set)),
        "merged_classes": int(len(set(y_tr_m))),
        "coverage_train": float(np.mean(np.isin(y_train, list(kept_set)))),
        "coverage_val": float(np.mean(np.isin(y_val, list(kept_set)))),
        "coverage_test": float(np.mean(np.isin(y_test, list(kept_set)))),
    }
    return y_tr_m, y_va_m, y_te_m, report


# ============================================================
# Feature loading
# ============================================================

def _raw_path(vid: str, split: str, feat: str) -> Path:
    return RAW_FEAT_ROOT / feat / split / f"{vid}.npy"


def _deep_path(vid: str, split: str, subdir: str) -> Path:
    return DEEP_FEAT_ROOT / subdir / split / f"{vid}.npy"


def load_raw_stacked(vid: str, split: str, feature_names: Sequence[str]) -> np.ndarray:
    parts = [
        np.asarray(np.load(_raw_path(vid, split, f), mmap_mode="r"), dtype=np.float32)
        for f in feature_names
    ]
    return np.concatenate(parts, axis=1)


def load_deep_features(vid: str, split: str, subdir: str = DEFAULT_DEEP_SUBDIR) -> np.ndarray:
    return np.asarray(np.load(_deep_path(vid, split, subdir), mmap_mode="r"), dtype=np.float32)


def pool_deep(arr: np.ndarray) -> np.ndarray:
    """Mean-pool deep features over nodes -> (D,) vector."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features, got {arr.shape}")
    return arr.mean(axis=0, dtype=np.float32)


def has_features(vid: str, split: str, feature_source: str, deep_subdir: str,
                 raw_feature_names: Sequence[str]) -> bool:
    if feature_source == "deep":
        return _deep_path(vid, split, deep_subdir).exists()
    if feature_source == "raw":
        return all(_raw_path(vid, split, f).exists() for f in raw_feature_names)
    raise ValueError(f"Unknown feature_source: {feature_source}")


# ============================================================
# Graph helpers
# ============================================================

@lru_cache(maxsize=1)
def load_skeleton_edge_index() -> torch.Tensor:
    """Load 8-neighbour skeleton connectivity (weights ignored)."""
    if SKELETON_PATH.exists():
        path = SKELETON_PATH
    elif LEGACY_ADJ_PATH.exists():
        path = LEGACY_ADJ_PATH
    else:
        raise FileNotFoundError(f"No adjacency file found at {SKELETON_PATH} or {LEGACY_ADJ_PATH}")
    coo = sp.load_npz(path).tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


def compute_gaussian_edge_weights(
    raw_x: torch.Tensor, edge_index: torch.Tensor, sigma: float = DEFAULT_EDGE_SIGMA
) -> torch.Tensor:
    """exp(-||x_u - x_v||^2 / (2 sigma^2)) per edge.

    Computes RGB-similarity-style weights dynamically from raw node features.
    Used by SGCN-raw, SGCN-deep and GAT to standardise on dynamic weighting.
    """
    u, v = edge_index[0], edge_index[1]
    diff = raw_x[u] - raw_x[v]
    dist2 = (diff * diff).sum(dim=1)
    sigma_sq = max(sigma * sigma, 1e-12)
    return torch.exp(-dist2 / (2.0 * sigma_sq))


# ============================================================
# Datasets
# ============================================================

@dataclass
class SampleSpec:
    vehicle_id: str
    split: str


class FGVDGraphDataset(torch.utils.data.Dataset):
    """Returns PyG Data with x, raw_x, edge_index, y.

    `raw_x` is always present (uses RGB only — same 3-channel basis the paper
    uses for edge weights). This lets every graph model compute dynamic edge
    weights on the GPU without baking them into adjacency files.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        level: str,
        feature_source: str,
        deep_subdir: str = DEFAULT_DEEP_SUBDIR,
        edge_feature_names: Sequence[str] = ("rgb",),
    ):
        self.df = df.reset_index(drop=True)
        self.y = y
        self.level = level
        self.feature_source = feature_source
        self.deep_subdir = deep_subdir
        self.edge_feature_names = tuple(edge_feature_names)
        self._edge_index = load_skeleton_edge_index()

    def __len__(self) -> int:
        return len(self.df)

    def _load_x(self, vid: str, split: str) -> np.ndarray:
        if self.feature_source == "deep":
            return load_deep_features(vid, split, self.deep_subdir)
        # raw: stack the level's raw features
        return load_raw_stacked(vid, split, LEVEL_RAW_FEATURES[self.level])

    def __getitem__(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        vid = str(row["vehicle_id"])
        split = str(row["split"])

        x_np = self._load_x(vid, split)
        raw_np = load_raw_stacked(vid, split, self.edge_feature_names)

        x = torch.from_numpy(np.ascontiguousarray(x_np))
        raw_x = torch.from_numpy(np.ascontiguousarray(raw_np))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return Data(x=x, raw_x=raw_x, edge_index=self._edge_index, y=y)


class FGVDPooledDataset(torch.utils.data.Dataset):
    """Returns (mean-pooled deep feature, label) — for the CNN/MLP baseline."""

    def __init__(self, df: pd.DataFrame, y: np.ndarray, deep_subdir: str = DEFAULT_DEEP_SUBDIR):
        self.df = df.reset_index(drop=True)
        self.y = y
        self.deep_subdir = deep_subdir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        vid = str(row["vehicle_id"])
        split = str(row["split"])
        arr = load_deep_features(vid, split, self.deep_subdir)
        x = torch.from_numpy(np.ascontiguousarray(pool_deep(arr)))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


# ============================================================
# Models
# ============================================================

class _SGCNLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.5):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch, add_self_loops=True, normalize=True)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_weight=edge_attr)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SGCNModel(nn.Module):
    """Spatial GCN. Computes Gaussian edge weights from raw_x dynamically.

    Works for both 'raw' and 'deep' node-feature variants — only the
    in_channels and the source of `data.x` differ between the two.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        edge_sigma: float = DEFAULT_EDGE_SIGMA,
    ):
        super().__init__()
        self.edge_sigma = edge_sigma
        layers = []
        for i in range(num_layers):
            layers.append(_SGCNLayer(in_channels if i == 0 else hidden_dim, hidden_dim, dropout))
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, raw_x, batch = data.x, data.edge_index, data.raw_x, data.batch
        edge_attr = compute_gaussian_edge_weights(raw_x, edge_index, self.edge_sigma)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GATModel(nn.Module):
    """GAT with dynamic edge weights as edge_attr (1-d).

    ``hidden_dim`` is treated as the total hidden width. PyG's GATConv
    interprets ``out_channels`` as the per-head width when ``concat=True``, so
    using hidden_dim directly there multiplies activation memory by ``heads``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
        heads: int = 8,
        dropout: float = 0.5,
        edge_sigma: float = DEFAULT_EDGE_SIGMA,
    ):
        super().__init__()
        self.edge_sigma = edge_sigma
        self.dropout = dropout
        self.head_dim = max(1, (hidden_dim + heads - 1) // heads)
        self.gat_hidden_dim = self.head_dim * heads
        self.conv1 = GATConv(in_channels, self.head_dim, heads=heads, dropout=dropout, edge_dim=1, concat=True)
        self.bn1 = nn.BatchNorm1d(self.gat_hidden_dim)
        self.conv2 = GATConv(self.gat_hidden_dim, hidden_dim, heads=1, dropout=dropout, edge_dim=1, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, raw_x, batch = data.x, data.edge_index, data.raw_x, data.batch
        edge_w = compute_gaussian_edge_weights(raw_x, edge_index, self.edge_sigma).view(-1, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_w)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_w)
        x = self.bn2(x)
        x = F.elu(x)
        g = global_mean_pool(x, batch)
        return self.classifier(g)


class DeepMLP(nn.Module):
    """MLP over mean-pooled deep features. Used by cnn.ipynb baseline."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Hierarchy helpers: opt-in masked-softmax loss
# ============================================================

def build_parent_index(class_names: Sequence[str], level: str) -> np.ndarray:
    """Returns int array of length len(class_names): parent index for each class.
    Parent index is encoded against the *unique parents* in this label space.
    Returns parent_idx_per_class plus the parent name list via tuple form.
    """
    parents = [parent_label_for(c, level) for c in class_names]
    uniq_parents = sorted(set(parents))
    p_to_idx = {p: i for i, p in enumerate(uniq_parents)}
    return np.array([p_to_idx[p] for p in parents], dtype=np.int64), uniq_parents


class MaskedHierarchicalCE(nn.Module):
    """Cross-entropy where logits whose parent ≠ true parent are masked out.

    Forces the network to compete only within the correct parent group during
    training, which is the single-multi-head equivalent of cascaded prediction.

    Implementation note: we cannot pass ``-inf`` masked logits to
    ``F.cross_entropy(label_smoothing=...)`` — the smoothing term sums
    ``log_softmax`` across all classes, including masked positions whose
    softmax is 0 → ``log(0) = -inf`` → loss = inf. We compute log-softmax
    manually over the un-masked subset and apply smoothing inside that group.
    """

    def __init__(self, parent_per_class: torch.Tensor, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        # parent_per_class: LongTensor (C,)  giving parent group id for each class
        self.register_buffer("parent_per_class", parent_per_class.long())
        self.weight = weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_parent = self.parent_per_class[targets]                  # (B,)
        mask = self.parent_per_class.unsqueeze(0) == target_parent.unsqueeze(1)  # (B, C) bool

        # Replace masked positions with very negative finite value so log_softmax stays finite.
        very_neg = torch.finfo(logits.dtype).min / 4
        masked_logits = logits.masked_fill(~mask, very_neg)
        log_probs = F.log_softmax(masked_logits, dim=1)                 # (B, C)

        # NLL on the true class — guaranteed un-masked since parent matches itself.
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)     # (B,)

        if self.label_smoothing > 0:
            # Mean log-prob within the un-masked group only.
            zeros = torch.zeros_like(log_probs)
            in_group_lp = torch.where(mask, log_probs, zeros)           # (B, C)
            n_in_group = mask.sum(dim=1).clamp(min=1).to(log_probs.dtype)
            mean_lp = in_group_lp.sum(dim=1) / n_in_group               # (B,)
            eps = self.label_smoothing
            sample_loss = (1.0 - eps) * nll + eps * (-mean_lp)
        else:
            sample_loss = nll

        if self.weight is not None:
            sample_loss = sample_loss * self.weight[targets]

        return sample_loss.mean()


def hierarchical_predict(
    logits: torch.Tensor, parent_per_class: torch.Tensor, predicted_parent: torch.Tensor
) -> torch.Tensor:
    """At inference: given a parent prediction, mask out logits with non-matching parents
    and return argmax. Used when we have a separate L1 head; here we just take the
    argmax over all logits since training already enforced the parent constraint.
    """
    mask = parent_per_class.unsqueeze(0) == predicted_parent.unsqueeze(1)
    masked = logits.masked_fill(~mask, float("-inf"))
    return masked.argmax(dim=1)


# ============================================================
# Long-tail losses
# ============================================================

class LogitAdjustedCE(nn.Module):
    """Menon et al. (ICLR'21). Adds tau * log(prior) to logits during training."""

    def __init__(self, class_freq: torch.Tensor, tau: float = 1.0, weight: torch.Tensor | None = None):
        super().__init__()
        prior = class_freq / class_freq.sum()
        self.register_buffer("log_prior", torch.log(prior + 1e-12))
        self.tau = tau
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj = logits + self.tau * self.log_prior
        return F.cross_entropy(adj, targets, weight=self.weight)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_class_balanced_weights(class_freq: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Cui et al. (CVPR'19). Effective-number weighting."""
    eff = 1.0 - np.power(beta, np.asarray(class_freq, dtype=np.float64))
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    w = w / w.mean()
    return w.astype(np.float32)


# ============================================================
# Training loop (PyG models + plain torch)
# ============================================================

def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _make_grad_scaler(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _is_pyg_batch(batch) -> bool:
    return isinstance(batch, Data)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device=DEVICE,
    *,
    scaler=None,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    device = torch.device(device)
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, n = 0.0, 0
    preds_all, targets_all = [], []
    amp_enabled = bool(use_amp and device.type == "cuda")
    grad_accum_steps = max(1, int(grad_accum_steps))
    pending_backward_steps = 0

    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        if _is_pyg_batch(batch):
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            targets = batch.y.view(-1)
            inputs = batch
        else:
            x, y = batch
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))
            targets = y
            inputs = x

        with torch.set_grad_enabled(training):
            with _autocast_context(device, amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if training:
                loss_for_backward = loss / grad_accum_steps
                if scaler is not None and amp_enabled:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()
                pending_backward_steps += 1

                if pending_backward_steps >= grad_accum_steps:
                    if scaler is not None and amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending_backward_steps = 0

        bs = targets.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds_all.append(logits.argmax(dim=1).detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

    if training and pending_backward_steps > 0:
        if scaler is not None and amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(n, 1)
    preds = np.concatenate(preds_all) if preds_all else np.array([], dtype=np.int64)
    targets = np.concatenate(targets_all) if targets_all else np.array([], dtype=np.int64)
    acc = accuracy_score(targets, preds) if len(targets) else 0.0
    macro = f1_score(targets, preds, average="macro", zero_division=0) if len(targets) else 0.0
    return avg_loss, acc, macro, preds, targets


def fit_with_resume(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    optimizer,
    criterion,
    target_total_epochs: int,
    ckpt_dir: Path,
    label_classes: Sequence[str],
    run_signature: dict | None = None,
    resume: bool = True,
    print_every: int = 5,
    device=DEVICE,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
) -> tuple[nn.Module, dict]:
    device = torch.device(device)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"
    scaler = _make_grad_scaler(device, use_amp)

    history = {k: [] for k in ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1"]}
    start_epoch = 1
    best_val_acc = -1.0
    best_state = None

    if resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location="cpu")
        label_match = ckpt.get("label_classes") == list(label_classes)
        if label_match:
            if run_signature is not None and ckpt.get("run_signature") != run_signature:
                print("Checkpoint training config differs — starting fresh.")
                ckpt_sig = ckpt.get("run_signature")
                if ckpt_sig is not None:
                    changed = [
                        k for k, v in run_signature.items()
                        if ckpt_sig.get(k) != v
                    ]
                    if changed:
                        print(f"  Changed keys: {', '.join(changed[:6])}")
                else:
                    print("  Existing checkpoint has no run signature.")
                ckpt = None
        else:
            print("Checkpoint label space differs — starting fresh.")

        if ckpt is not None and label_match:
            model_state = ckpt.get("model_state", {})
            current_state = model.state_dict()
            mismatched = [
                (k, tuple(v.shape), tuple(current_state[k].shape))
                for k, v in model_state.items()
                if k in current_state and tuple(v.shape) != tuple(current_state[k].shape)
            ]
            missing = [k for k in current_state if k not in model_state]
            unexpected = [k for k in model_state if k not in current_state]

            if mismatched or missing or unexpected:
                print("Checkpoint model architecture differs — starting fresh.")
                if mismatched:
                    k, old_shape, new_shape = mismatched[0]
                    print(f"  First mismatch: {k} checkpoint={old_shape} current={new_shape}")
            else:
                model.load_state_dict(model_state)
                optimizer.load_state_dict(ckpt["optim_state"])
                scaler_state = ckpt.get("scaler_state")
                if scaler_state is not None and scaler.is_enabled():
                    scaler.load_state_dict(scaler_state)
                history = ckpt.get("history", history)
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                best_val_acc = float(ckpt.get("best_val_acc", -1.0))
                best_state = ckpt.get("best_state", None)
                print(f"Resumed from {last_ckpt} (epoch {start_epoch - 1}).")

    for epoch in range(start_epoch, target_total_epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=use_amp, grad_accum_steps=grad_accum_steps,
        )
        va_loss, va_acc, va_f1, _, _ = run_epoch(
            model, val_loader, criterion, None, device, use_amp=use_amp,
        )

        for k, v in zip(
            ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1"],
            [epoch, tr_loss, va_loss, tr_acc, va_acc, tr_f1, va_f1],
        ):
            history[k].append(float(v) if k != "epoch" else int(v))

        improved = va_acc > best_val_acc
        if improved:
            best_val_acc = float(va_acc)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {"epoch": epoch, "best_val_acc": best_val_acc, "best_state": best_state,
                 "history": history, "label_classes": list(label_classes),
                 "run_signature": run_signature},
                best_ckpt,
            )

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(),
             "best_val_acc": best_val_acc, "best_state": best_state, "history": history,
             "label_classes": list(label_classes),
             "run_signature": run_signature,
             "scaler_state": scaler.state_dict() if scaler.is_enabled() else None},
            last_ckpt,
        )

        if epoch == start_epoch or epoch % print_every == 0 or epoch == target_total_epochs:
            print(f"Epoch {epoch:03d} | train_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
                  f"| train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate(model, loader, criterion=None, k_list=(1, 3, 5), device=DEVICE,
             use_amp: bool = False) -> dict:
    """Top-k accuracy + macro/weighted F1 + raw predictions."""
    device = torch.device(device)
    model.eval()
    all_logits, all_targets = [], []
    total_loss, n = 0.0, 0
    amp_enabled = bool(use_amp and device.type == "cuda")

    with torch.no_grad():
        for batch in loader:
            if _is_pyg_batch(batch):
                batch = batch.to(device, non_blocking=(device.type == "cuda"))
                targets = batch.y.view(-1)
                with _autocast_context(device, amp_enabled):
                    logits = model(batch)
            else:
                x, y = batch
                x = x.to(device, non_blocking=(device.type == "cuda"))
                y = y.to(device, non_blocking=(device.type == "cuda"))
                targets = y
                with _autocast_context(device, amp_enabled):
                    logits = model(x)

            if criterion is not None:
                with _autocast_context(device, amp_enabled):
                    loss = criterion(logits, targets)
                bs = targets.size(0)
                total_loss += loss.item() * bs
                n += bs

            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    n_classes = logits.shape[1]

    out = {"loss": total_loss / max(n, 1) if criterion is not None else None}

    pred1 = logits.argmax(dim=1).numpy()
    targets_np = targets.numpy()
    out["acc"] = float(accuracy_score(targets_np, pred1))
    out["macro_f1"] = float(f1_score(targets_np, pred1, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(targets_np, pred1, average="weighted", zero_division=0))

    for k in k_list:
        if k > n_classes:
            out[f"top{k}_acc"] = out["acc"]
            continue
        topk = logits.topk(k, dim=1).indices.numpy()
        hits = (topk == targets_np[:, None]).any(axis=1)
        out[f"top{k}_acc"] = float(hits.mean())

    out["preds"] = pred1
    out["targets"] = targets_np
    return out


# ============================================================
# Metrics
# ============================================================

def stratified_accuracy_by_support(
    y_true: np.ndarray, y_pred: np.ndarray, train_counts: dict[int, int],
    bins: tuple = (0, 20, 100, np.inf),
) -> dict[str, float]:
    """Many/medium/few-shot stratified accuracy, by training support."""
    bin_names = ["few(<20)", "med(20-100)", "many(100+)"]
    out = {}
    for lo, hi, name in zip(bins[:-1], bins[1:], bin_names):
        in_bin = np.array([train_counts.get(int(t), 0) >= lo and train_counts.get(int(t), 0) < hi
                           for t in y_true])
        if in_bin.sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(accuracy_score(y_true[in_bin], y_pred[in_bin]))
    return out


def per_parent_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, parent_per_class: np.ndarray
) -> dict[int, float]:
    """Macro-F1 within each parent group."""
    out = {}
    for p in np.unique(parent_per_class):
        member_classes = np.where(parent_per_class == p)[0]
        mask = np.isin(y_true, member_classes)
        if mask.sum() == 0:
            continue
        out[int(p)] = float(f1_score(y_true[mask], y_pred[mask], average="macro",
                                     labels=member_classes, zero_division=0))
    return out


def classification_summary(y_true_labels, y_pred_labels) -> str:
    return classification_report(y_true_labels, y_pred_labels, digits=4, zero_division=0)


# ============================================================
# Plotting / artifacts
# ============================================================

def plot_learning_curves(history: dict, title: str, out_path: Path | None = None) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="val_loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train_acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="val_acc", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_run_artifacts(run_dir: Path, history: dict, metrics: dict, le: LabelEncoder,
                       extra: dict | None = None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {k: v for k, v in metrics.items() if k not in ("preds", "targets")},
        "history": history,
        "classes": le.classes_.tolist() if le is not None else None,
    }
    if extra is not None:
        payload.update(extra)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# High-level entry point
# ============================================================

@dataclass
class ExperimentConfig:
    method: str            # "sgcn", "gat", "rf", "cnn"
    level: str             # "L1", "L2", "L3"
    feature_source: str    # "raw" or "deep"
    case: str = "all"
    deep_subdir: str = DEFAULT_DEEP_SUBDIR
    edge_sigma: float = DEFAULT_EDGE_SIGMA
    hierarchical: bool | None = None  # None defaults to flat CE; set True to opt into MaskedHierarchicalCE
    tail_merge_min: int | None = None
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    heads: int = 8                  # GAT
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    rf_pca_dim: int = 256           # used for raw RF
    use_logit_adjustment: bool = False
    label_smoothing: float = 0.05
    num_workers: int = 0
    resume: bool = True
    print_every: int = 5
    grad_accum_steps: int = 1
    use_amp: bool = True
    seed: int = SEED


def _autoset_hierarchy(cfg: ExperimentConfig) -> ExperimentConfig:
    if cfg.hierarchical is None:
        cfg.hierarchical = False
    if cfg.tail_merge_min is None and cfg.level in ("L2", "L3"):
        cfg.tail_merge_min = 20
    return cfg


def _filter_to_existing_features(df, feature_source, deep_subdir, level):
    raw_names = LEVEL_RAW_FEATURES[level]
    keep = []
    for r in df.itertuples(index=False):
        if has_features(str(r.vehicle_id), str(r.split), feature_source, deep_subdir, raw_names):
            keep.append(True)
        else:
            keep.append(False)
    return df.loc[keep].reset_index(drop=True), np.array(keep, dtype=bool)


def _prepare_label_space(cfg, train_df, val_df, test_df):
    """Build merged labels + LabelEncoder."""
    y_tr = build_level_labels(train_df, cfg.level)
    y_va = build_level_labels(val_df, cfg.level)
    y_te = build_level_labels(test_df, cfg.level)

    y_tr, _ = apply_case(train_df, y_tr, cfg.level, cfg.case)
    y_va, _ = apply_case(val_df, y_va, cfg.level, cfg.case)
    y_te, _ = apply_case(test_df, y_te, cfg.level, cfg.case)

    tail_report = None
    if cfg.tail_merge_min is not None and cfg.level in ("L2", "L3"):
        y_tr, y_va, y_te, tail_report = tail_merge(y_tr, y_va, y_te, cfg.tail_merge_min, cfg.level)

    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_va, y_te]))
    return le.transform(y_tr), le.transform(y_va), le.transform(y_te), le, tail_report


def _make_loaders(cfg, train_df, val_df, test_df, y_tr, y_va, y_te, pyg: bool):
    if pyg:
        ds_tr = FGVDGraphDataset(train_df, y_tr, cfg.level, cfg.feature_source, cfg.deep_subdir)
        ds_va = FGVDGraphDataset(val_df, y_va, cfg.level, cfg.feature_source, cfg.deep_subdir)
        ds_te = FGVDGraphDataset(test_df, y_te, cfg.level, cfg.feature_source, cfg.deep_subdir)
        loader_cls = PyGDataLoader
    else:
        ds_tr = FGVDPooledDataset(train_df, y_tr, cfg.deep_subdir)
        ds_va = FGVDPooledDataset(val_df, y_va, cfg.deep_subdir)
        ds_te = FGVDPooledDataset(test_df, y_te, cfg.deep_subdir)
        loader_cls = torch.utils.data.DataLoader

    pin = (DEVICE.type == "cuda")
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=pin)
    train_loader = loader_cls(ds_tr, shuffle=True, **common)
    val_loader = loader_cls(ds_va, shuffle=False, **common)
    test_loader = loader_cls(ds_te, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def _build_criterion(cfg, y_train: np.ndarray, num_classes: int, le: LabelEncoder) -> nn.Module:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    cb_weights = torch.tensor(compute_class_balanced_weights(counts), device=DEVICE)

    # Opt-in only. MaskedHierarchicalCE assumes the true parent is known while
    # training, so plain global top-k accuracy can look artificially poor.
    if cfg.hierarchical and cfg.level in ("L2", "L3"):
        parent_idx, _ = build_parent_index(le.classes_.tolist(), cfg.level)
        parent_t = torch.tensor(parent_idx, device=DEVICE)
        return MaskedHierarchicalCE(parent_t, weight=cb_weights, label_smoothing=cfg.label_smoothing)

    if cfg.use_logit_adjustment:
        freq = torch.tensor(counts, device=DEVICE)
        return LogitAdjustedCE(freq, tau=1.0, weight=cb_weights)

    return nn.CrossEntropyLoss(weight=cb_weights, label_smoothing=cfg.label_smoothing)


def _run_signature(cfg: ExperimentConfig, n_classes: int) -> dict:
    """Fields that must match to safely resume a torch checkpoint."""
    return {
        "method": cfg.method,
        "level": cfg.level,
        "feature_source": cfg.feature_source,
        "case": cfg.case,
        "deep_subdir": cfg.deep_subdir,
        "n_classes": int(n_classes),
        "tail_merge_min": cfg.tail_merge_min,
        "hierarchical": bool(cfg.hierarchical),
        "use_logit_adjustment": bool(cfg.use_logit_adjustment),
        "label_smoothing": float(cfg.label_smoothing),
        "hidden_dim": int(cfg.hidden_dim),
        "num_layers": int(cfg.num_layers),
        "heads": int(cfg.heads),
        "dropout": float(cfg.dropout),
        "edge_sigma": float(cfg.edge_sigma),
    }


# ----- RF path (sklearn) -----

def _run_rf(cfg: ExperimentConfig, train_df, val_df, test_df,
            y_tr, y_va, y_te, le: LabelEncoder, ckpt_dir: Path) -> dict:
    """Train RF on pooled deep features OR PCA-reduced raw features."""
    def _build_X(df, split_name):
        rows = []
        for r in df.itertuples(index=False):
            vid, sp_name = str(r.vehicle_id), str(r.split)
            if cfg.feature_source == "deep":
                arr = load_deep_features(vid, sp_name, cfg.deep_subdir)
                rows.append(pool_deep(arr))
            else:
                arr = load_raw_stacked(vid, sp_name, LEVEL_RAW_FEATURES[cfg.level])
                rows.append(arr.reshape(-1).astype(np.float32))
        return np.stack(rows, axis=0).astype(np.float32)

    print(f"Building features (RF/{cfg.feature_source}) ...")
    X_tr = _build_X(train_df, "train")
    X_va = _build_X(val_df, "val")
    X_te = _build_X(test_df, "test")
    print(f"  X_tr={X_tr.shape}  X_va={X_va.shape}  X_te={X_te.shape}")

    pca = None
    if cfg.feature_source == "raw" and X_tr.shape[1] > cfg.rf_pca_dim:
        pca = PCA(n_components=cfg.rf_pca_dim, random_state=cfg.seed)
        X_tr = pca.fit_transform(X_tr).astype(np.float32)
        X_va = pca.transform(X_va).astype(np.float32)
        X_te = pca.transform(X_te).astype(np.float32)
        print(f"  PCA -> {X_tr.shape[1]} dims (explained var={pca.explained_variance_ratio_.sum():.4f})")

    rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=cfg.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_tr, y_tr)

    proba_te = rf.predict_proba(X_te)
    pred_te = proba_te.argmax(axis=1)
    pred_va = rf.predict(X_va)

    val_acc = float(accuracy_score(y_va, pred_va))
    test_acc = float(accuracy_score(y_te, pred_te))
    test_f1 = float(f1_score(y_te, pred_te, average="macro", zero_division=0))
    test_wf1 = float(f1_score(y_te, pred_te, average="weighted", zero_division=0))

    n_classes = len(le.classes_)
    topk = {1: test_acc}
    for k in (3, 5):
        if k <= n_classes:
            top_idx = np.argsort(-proba_te, axis=1)[:, :k]
            topk[k] = float((top_idx == y_te[:, None]).any(axis=1).mean())
        else:
            topk[k] = test_acc

    metrics = {
        "val_acc": val_acc,
        "acc": test_acc,
        "top1_acc": topk[1],
        "top3_acc": topk[3],
        "top5_acc": topk[5],
        "macro_f1": test_f1,
        "weighted_f1": test_wf1,
    }

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "pca": pca, "label_classes": le.classes_.tolist()},
                ckpt_dir / "best.joblib")

    print(classification_summary(le.inverse_transform(y_te), le.inverse_transform(pred_te)))
    return {"metrics": metrics, "preds": pred_te, "targets": y_te}


# ----- main dispatcher -----

def run_experiment(cfg: ExperimentConfig, *, ckpt_root: Path = Path("checkpoints"),
                   plot_root: Path = Path("plots")) -> dict:
    set_seed(cfg.seed)
    cfg = _autoset_hierarchy(cfg)

    print(f"\n=== {cfg.method.upper()} | {cfg.level} | {cfg.feature_source} | "
          f"hierarchical={cfg.hierarchical} | tail_min={cfg.tail_merge_min} ===")

    md = load_metadata()
    train_df = md[md.split == "train"].reset_index(drop=True)
    val_df = md[md.split == "val"].reset_index(drop=True)
    test_df = md[md.split == "test"].reset_index(drop=True)

    train_df, _ = _filter_to_existing_features(train_df, cfg.feature_source, cfg.deep_subdir, cfg.level)
    val_df, _ = _filter_to_existing_features(val_df, cfg.feature_source, cfg.deep_subdir, cfg.level)
    test_df, _ = _filter_to_existing_features(test_df, cfg.feature_source, cfg.deep_subdir, cfg.level)

    y_tr, y_va, y_te, le, tail_report = _prepare_label_space(cfg, train_df, val_df, test_df)
    n_classes = len(le.classes_)
    print(f"Samples: train={len(train_df)} val={len(val_df)} test={len(test_df)} | classes={n_classes}")
    if tail_report is not None:
        print(f"Tail-merge: kept {tail_report['kept_classes']}/{tail_report['orig_classes']} classes "
              f"| coverage train={tail_report['coverage_train']:.3f} "
              f"val={tail_report['coverage_val']:.3f} test={tail_report['coverage_test']:.3f}")

    tag = f"{cfg.level}_{cfg.case}"
    ckpt_dir = ckpt_root / cfg.method / cfg.feature_source / tag
    plot_path = plot_root / cfg.method / cfg.feature_source / f"{tag}_curves.png"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    extras = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
              "tail_report": tail_report}

    if cfg.method == "rf":
        out = _run_rf(cfg, train_df, val_df, test_df, y_tr, y_va, y_te, le, ckpt_dir)
        save_run_artifacts(ckpt_dir, history={}, metrics=out["metrics"], le=le, extra=extras)
        return {"metrics": out["metrics"], "le": le, "ckpt_dir": ckpt_dir,
                "preds": out["preds"], "targets": out["targets"]}

    # ---- torch path: SGCN / GAT / CNN ----
    pyg = cfg.method in ("sgcn", "gat")
    train_loader, val_loader, test_loader = _make_loaders(
        cfg, train_df, val_df, test_df, y_tr, y_va, y_te, pyg=pyg
    )

    sample = next(iter(train_loader))
    if pyg:
        in_channels = sample.x.shape[1]
    else:
        in_channels = sample[0].shape[1]

    if cfg.method == "sgcn":
        model = SGCNModel(in_channels, n_classes, cfg.hidden_dim, cfg.num_layers,
                          cfg.dropout, cfg.edge_sigma).to(DEVICE)
    elif cfg.method == "gat":
        model = GATModel(in_channels, n_classes, cfg.hidden_dim, cfg.heads,
                         cfg.dropout, cfg.edge_sigma).to(DEVICE)
    elif cfg.method == "cnn":
        model = DeepMLP(in_channels, n_classes, hidden=512, dropout=cfg.dropout).to(DEVICE)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = _build_criterion(cfg, y_tr, n_classes, le)
    if cfg.grad_accum_steps > 1:
        print(f"Gradient accumulation: micro_batch={cfg.batch_size} "
              f"effective_batch={cfg.batch_size * cfg.grad_accum_steps}")
    if cfg.use_amp and DEVICE.type == "cuda":
        print("AMP mixed precision: enabled")

    model, history = fit_with_resume(
        model, train_loader, val_loader,
        optimizer=optimizer, criterion=criterion,
        target_total_epochs=cfg.epochs, ckpt_dir=ckpt_dir,
        label_classes=le.classes_.tolist(), run_signature=_run_signature(cfg, n_classes),
        resume=cfg.resume,
        print_every=cfg.print_every,
        use_amp=cfg.use_amp,
        grad_accum_steps=cfg.grad_accum_steps,
    )

    # final test eval (un-masked CE for fair top-k)
    test_metrics = evaluate(
        model, test_loader, criterion=nn.CrossEntropyLoss(), k_list=(1, 3, 5),
        use_amp=cfg.use_amp,
    )
    test_metrics["val_acc_best"] = float(max(history["val_acc"])) if history["val_acc"] else None

    print(f"\nTest acc={test_metrics['acc']:.4f} | top3={test_metrics['top3_acc']:.4f} "
          f"| top5={test_metrics['top5_acc']:.4f} | macro_f1={test_metrics['macro_f1']:.4f}")
    print(classification_summary(
        le.inverse_transform(test_metrics["targets"]),
        le.inverse_transform(test_metrics["preds"]),
    ))

    plot_learning_curves(history, title=f"{cfg.method} | {cfg.level} | {cfg.feature_source}",
                         out_path=plot_path)

    metrics_to_save = {k: v for k, v in test_metrics.items() if k not in ("preds", "targets")}
    save_run_artifacts(ckpt_dir, history=history, metrics=metrics_to_save, le=le, extra=extras)

    return {"metrics": metrics_to_save, "history": history, "le": le, "ckpt_dir": ckpt_dir,
            "preds": test_metrics["preds"], "targets": test_metrics["targets"]}
