"""
deep_feature_refiner_multilevel.py
====================================
"World 2" Deep Feature Refiner — Multi-Level Joint Training
-------------------------------------------------------------
KEY CHANGES vs single-level version
-------------------------------------
1. Input channels: always 8 (RGB=3 + Gabor=4 + Sobel=1) for ALL levels.
2. Shared Inception backbone across L1 / L2 / L3.
3. Three separate classification heads — one per label level.
4. Joint training loss = L1_loss + L2_loss + L3_loss (equal weights).
5. Single checkpoint stores backbone + all 3 heads.
6. Feature extraction outputs ONE (4096, D) file per vehicle (shared
   backbone means the spatial features are level-agnostic).

Pipeline summary
----------------
  .npy files (4096, 8)           ← always 8-channel baseline (RGB+Gabor+Sobel)
       │
       ▼  reshape
  (64, 64, 8) → (8, 64, 64)     ← spatial feature cube
       │
       ▼  Shared ZeroStride Inception backbone
  (B, 128, 64, 64)
       │
       ├─ 1×1 Proj → (B, D, 64, 64) → reshape → (B, 4096, D)  [SGCN handover]
       │
       ├─ Head-L1: GAP → FC(128 → n_L1)   [training only]
       ├─ Head-L2: GAP → FC(128 → n_L2)   [training only]
       └─ Head-L3: GAP → FC(128 → n_L3)   [training only]

Usage — training (all 3 levels jointly)
-----------------------------------------
    python deep_feature_refiner_multilevel.py train \\
        --handover_dir  /path/to/FGVD_Graph_Handover \\
        --output_dir    /path/to/FGVD_Graph_Handover \\
        --epochs        30 \\
        --batch_size    64 \\
        --lr            1e-3 \\
        --D             64

Usage — inference (extract shared deep features)
-------------------------------------------------
    python deep_feature_refiner_multilevel.py extract \\
        --handover_dir  /path/to/FGVD_Graph_Handover \\
        --output_dir    /path/to/FGVD_Graph_Handover \\
        --checkpoint    /path/to/FGVD_Graph_Handover/checkpoints/best_multilevel.pt \\
        --D             64

Output layout
-------------
    FGVD_Graph_Handover/
    ├── deep_features/
    │   └── multilevel/
    │       ├── train/  <vehicle_id>.npy   shape (4096, D)
    │       ├── val/
    │       └── test/
    └── checkpoints/
        └── best_multilevel.pt
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_H, TARGET_W = 64, 64
N_NODES            = TARGET_H * TARGET_W          # 4096
IN_CHANNELS        = 8                            # RGB(3)+Gabor(4)+Sobel(1) for ALL levels
LABEL_LEVELS       = ["L1", "L2", "L3"]
SPLITS             = ["train", "val", "test"]


# ===========================================================================
# 1.  DATASET  — always loads 8-channel stack
# ===========================================================================

class FGVDMultiLevelDataset(Dataset):
    """
    Loads 8-channel baseline features and returns labels for all three
    hierarchy levels simultaneously.

    Parameters
    ----------
    handover_dir  : Path   Root of FGVD_Graph_Handover/
    split         : str    "train" | "val" | "test"
    label_maps    : dict   {level_str → {class_str → int}}
    """

    def __init__(self, handover_dir: Path, split: str, label_maps: Dict[str, dict]):
        self.rf         = handover_dir / "raw_features"
        self.split      = split
        self.label_maps = label_maps

        meta_path = handover_dir / "metadata.csv"
        self.records = []
        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                # Only keep samples that have a valid label at every level
                labels = {}
                valid  = True
                for lvl in LABEL_LEVELS:
                    lstr = row.get(lvl, "")
                    if lstr not in label_maps[lvl]:
                        valid = False
                        break
                    labels[lvl] = label_maps[lvl][lstr]
                if valid:
                    self.records.append({
                        "vehicle_id": row["vehicle_id"],
                        "labels":     labels,
                    })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        vid = rec["vehicle_id"]
        sp  = self.split

        # --- Always 8-channel stack: RGB + Gabor + Sobel ---
        rgb   = np.load(self.rf / "rgb"   / sp / f"{vid}.npy")   # (4096, 3)
        gabor = np.load(self.rf / "gabor" / sp / f"{vid}.npy")   # (4096, 4)
        sobel = np.load(self.rf / "sobel" / sp / f"{vid}.npy")   # (4096, 1)

        stacked = np.concatenate([rgb, gabor, sobel], axis=1).astype(np.float32)  # (4096, 8)

        # Reshape → (C, H, W)
        cube = stacked.reshape(TARGET_H, TARGET_W, -1)   # (64, 64, 8)
        cube = np.transpose(cube, (2, 0, 1))             # (8, 64, 64)

        x      = torch.from_numpy(cube)
        y_L1   = torch.tensor(rec["labels"]["L1"], dtype=torch.long)
        y_L2   = torch.tensor(rec["labels"]["L2"], dtype=torch.long)
        y_L3   = torch.tensor(rec["labels"]["L3"], dtype=torch.long)

        return x, y_L1, y_L2, y_L3, vid


def build_label_maps(handover_dir: Path) -> Dict[str, dict]:
    """
    Build label → int mappings for all three levels from the training split.
    Alphabetically sorted for reproducibility.
    """
    sets = {lvl: set() for lvl in LABEL_LEVELS}
    with open(handover_dir / "metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                for lvl in LABEL_LEVELS:
                    lstr = row.get(lvl, "")
                    if lstr:
                        sets[lvl].add(lstr)
    return {
        lvl: {lbl: idx for idx, lbl in enumerate(sorted(sets[lvl]))}
        for lvl in LABEL_LEVELS
    }


# ===========================================================================
# 2.  ARCHITECTURE — Shared Backbone + 3 Heads
# ===========================================================================

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      stride=1, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ZeroStrideInceptionBlock(nn.Module):
    """
    Four-branch Inception block — zero spatial downsampling.
      B1: 1×1
      B2: 1×1 → 3×3 dilated (d=2)
      B3: 1×1 → 5×5 dilated (d=2)
      B4: 1×1 projection
    Output channels = b1_ch + b2_ch + b3_ch + b4_ch
    """

    def __init__(self, in_ch, b1_ch, b2_reduce, b2_ch, b3_reduce, b3_ch, b4_ch):
        super().__init__()
        self.branch1 = ConvBnRelu(in_ch, b1_ch, 1)
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_ch, b2_reduce, 1),
            ConvBnRelu(b2_reduce, b2_ch, 3, dilation=2),
        )
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_ch, b3_reduce, 1),
            ConvBnRelu(b3_reduce, b3_ch, 5, dilation=2),
        )
        self.branch4 = ConvBnRelu(in_ch, b4_ch, 1)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ], dim=1)


class MultiLevelDeepFeatureRefiner(nn.Module):
    """
    Shared Inception backbone with THREE classification heads.

    Spatial path (SGCN handover)
    ----------------------------
      Input (8, 64, 64)
        → InceptionBlock1 (8  → 64)
        → InceptionBlock2 (64 → 128)
        → 1×1 Proj (128 → D)
        → output (D, 64, 64)  i.e. (4096, D) per sample

    Classification heads (training only — detached from spatial path)
    -----------------------------------------------------------------
      After Block2 output (128, 64, 64):
        → GAP → Flatten → Dropout → FC(128 → n_L1)   [Head L1]
        → GAP → Flatten → Dropout → FC(128 → n_L2)   [Head L2]
        → GAP → Flatten → Dropout → FC(128 → n_L3)   [Head L3]

    Parameters
    ----------
    num_classes : dict  {"L1": int, "L2": int, "L3": int}
    D           : int   feature dimension per pixel (default 64)
    """

    def __init__(self, num_classes: Dict[str, int], D: int = 64):
        super().__init__()
        self.D = D

        # ---- Shared Backbone ----
        # Block 1: 8 → 64 channels  (16+24+16+8)
        self.inception1 = ZeroStrideInceptionBlock(
            in_ch=IN_CHANNELS,
            b1_ch=16, b2_reduce=8,  b2_ch=24,
            b3_reduce=4, b3_ch=16,  b4_ch=8,
        )
        # Block 2: 64 → 128 channels  (32+48+32+16)
        self.inception2 = ZeroStrideInceptionBlock(
            in_ch=64,
            b1_ch=32, b2_reduce=16, b2_ch=48,
            b3_reduce=8, b3_ch=32,  b4_ch=16,
        )

        # 1×1 projection → D features per pixel (spatial handover)
        self.proj = nn.Sequential(
            nn.Conv2d(128, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
        )

        # ---- Three Classification Heads ----
        def _head(n):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(128, n),
            )

        self.head_L1 = _head(num_classes["L1"])
        self.head_L2 = _head(num_classes["L2"])
        self.head_L3 = _head(num_classes["L3"])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, return_spatial: bool = False):
        """
        Parameters
        ----------
        x              : (B, 8, 64, 64)
        return_spatial : True  → returns (B, D, 64, 64)  [inference]
                         False → returns (logits_L1, logits_L2, logits_L3, proj)
                                  [training; proj kept for gradient flow]
        """
        f1   = self.inception1(x)    # (B,  64, 64, 64)
        f2   = self.inception2(f1)   # (B, 128, 64, 64)
        proj = self.proj(f2)         # (B,   D, 64, 64)

        if return_spatial:
            return proj

        logits_L1 = self.head_L1(f2)
        logits_L2 = self.head_L2(f2)
        logits_L3 = self.head_L3(f2)
        return logits_L1, logits_L2, logits_L3, proj   # proj keeps grad alive


# ===========================================================================
# 3.  TRAINING
# ===========================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Multi-Level Joint Training  device={device}")
    print(f"{'='*60}")

    handover_dir = Path(args.handover_dir)
    output_dir   = Path(args.output_dir)
    ckpt_dir     = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Label maps for all 3 levels ---
    label_maps  = build_label_maps(handover_dir)
    num_classes = {lvl: len(label_maps[lvl]) for lvl in LABEL_LEVELS}
    for lvl, n in num_classes.items():
        print(f"  Classes {lvl}: {n}")

    # --- Datasets ---
    train_ds = FGVDMultiLevelDataset(handover_dir, "train", label_maps)
    val_ds   = FGVDMultiLevelDataset(handover_dir, "val",   label_maps)
    print(f"  Train: {len(train_ds)}   Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = MultiLevelDeepFeatureRefiner(num_classes, D=args.D).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # Verify zero-downsampling
    with torch.no_grad():
        dummy   = torch.zeros(1, IN_CHANNELS, TARGET_H, TARGET_W).to(device)
        spatial = model(dummy, return_spatial=True)
        assert spatial.shape == (1, args.D, TARGET_H, TARGET_W), \
            f"Downsampling detected! Got {tuple(spatial.shape)}"
    print(f"  ✓ Zero-downsampling verified: {tuple(spatial.shape)}")

    # --- Optimiser ---
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Loss weights — equal by default; tune if needed
    loss_weights = {"L1": 1.0, "L2": 4.0, "L3": 8.0}

    best_avg_acc = 0.0
    best_ckpt    = ckpt_dir / "best_multilevel.pt"
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        # Do not load optimizer to allow LR Restart
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resuming from epoch {ckpt['epoch']} with NEW learning rate.")
    for epoch in range(1, args.epochs + 1):

        # ---- Train ----
        model.train()
        train_stats = {lvl: {"loss": 0.0, "correct": 0, "total": 0} for lvl in LABEL_LEVELS}

        for x, y_L1, y_L2, y_L3, _ in tqdm(
                train_dl, desc=f"  Epoch {epoch}/{args.epochs} [train]",
                leave=False, unit="batch"):

            x    = x.to(device)
            ys   = {"L1": y_L1.to(device),
                    "L2": y_L2.to(device),
                    "L3": y_L3.to(device)}

            optimiser.zero_grad(set_to_none=True)
            logits_L1, logits_L2, logits_L3, _proj = model(x, return_spatial=False)
            logits = {"L1": logits_L1, "L2": logits_L2, "L3": logits_L3}

            # Joint loss
            total_loss = sum(
                loss_weights[lvl] * criterion(logits[lvl], ys[lvl])
                for lvl in LABEL_LEVELS
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()

            # Track per-level stats
            for lvl in LABEL_LEVELS:
                with torch.no_grad():
                    lv = criterion(logits[lvl], ys[lvl])
                train_stats[lvl]["loss"]    += lv.item() * x.size(0)
                train_stats[lvl]["correct"] += (logits[lvl].argmax(1) == ys[lvl]).sum().item()
                train_stats[lvl]["total"]   += x.size(0)

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_correct = {lvl: 0 for lvl in LABEL_LEVELS}
        val_total   = 0

        with torch.no_grad():
            for x, y_L1, y_L2, y_L3, _ in val_dl:
                x  = x.to(device)
                ys = {"L1": y_L1.to(device),
                      "L2": y_L2.to(device),
                      "L3": y_L3.to(device)}
                logits_L1, logits_L2, logits_L3, _ = model(x, return_spatial=False)
                logits = {"L1": logits_L1, "L2": logits_L2, "L3": logits_L3}
                for lvl in LABEL_LEVELS:
                    val_correct[lvl] += (logits[lvl].argmax(1) == ys[lvl]).sum().item()
                val_total += x.size(0)

        # Summary
        val_accs  = {lvl: 100.0 * val_correct[lvl] / val_total for lvl in LABEL_LEVELS}
        avg_acc   = sum(val_accs.values()) / 3
        lr_now    = scheduler.get_last_lr()[0]

        train_acc_str = "  ".join(
            f"{lvl}={100*train_stats[lvl]['correct']/train_stats[lvl]['total']:.1f}%"
            for lvl in LABEL_LEVELS
        )
        val_acc_str = "  ".join(
            f"{lvl}={val_accs[lvl]:.1f}%" for lvl in LABEL_LEVELS
        )

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train [{train_acc_str}] | "
              f"val   [{val_acc_str}] | "
              f"avg_val={avg_acc:.2f}% | lr={lr_now:.2e}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "label_maps":  label_maps,
                "num_classes": num_classes,
                "in_channels": IN_CHANNELS,
                "D":           args.D,
                "val_accs":    val_accs,
                "avg_val_acc": avg_acc,
            }, best_ckpt)
            print(f"    ★ New best avg_val_acc={avg_acc:.2f}%  "
                  f"[{val_acc_str}] — saved.")

    print(f"\n  Best avg val accuracy: {best_avg_acc:.2f}%")
    print(f"  Checkpoint: {best_ckpt}")
    return str(best_ckpt)


# ===========================================================================
# 4.  FEATURE EXTRACTION
# ===========================================================================

def extract(args):
    """
    Load a trained checkpoint and extract shared deep features for all splits.
    Saves ONE (4096, D) .npy file per vehicle — works for all label levels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Extracting shared deep features  device={device}")
    print(f"{'='*60}")

    handover_dir = Path(args.handover_dir)
    output_dir   = Path(args.output_dir)

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    label_maps  = ckpt["label_maps"]
    num_classes = ckpt["num_classes"]
    D           = ckpt["D"]
    print(f"  Best avg val acc: {ckpt['avg_val_acc']:.2f}%  (epoch {ckpt['epoch']})")
    print(f"  Val accs: " + "  ".join(
        f"{lvl}={ckpt['val_accs'][lvl]:.1f}%" for lvl in LABEL_LEVELS))

    # --- Model ---
    model = MultiLevelDeepFeatureRefiner(num_classes, D=D).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Output directory (single shared folder) ---
    deep_root = output_dir / "deep_features" / "multilevel"
    for split in SPLITS:
        (deep_root / split).mkdir(parents=True, exist_ok=True)

    # --- Extract per split ---
    for split in SPLITS:
        ds = FGVDMultiLevelDataset(handover_dir, split, label_maps)
        dl = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
        print(f"\n  Split '{split}': {len(ds)} crops …")

        with torch.no_grad():
            for x, *_, vids in tqdm(dl, desc=f"  {split}", unit="batch"):
                x       = x.to(device)
                spatial = model(x, return_spatial=True)      # (B, D, 64, 64)

                B    = spatial.size(0)
                feat = spatial.permute(0, 2, 3, 1)           # (B, H, W, D)
                feat = feat.reshape(B, N_NODES, D)            # (B, 4096, D)
                feat = feat.cpu().numpy().astype(np.float32)

                for i, vid in enumerate(vids):
                    np.save(str(deep_root / split / f"{vid}.npy"), feat[i])

    print(f"\n  Deep features saved to: {deep_root}")
    print(f"  Shape per file: (4096, {D})")
    _print_handover_note(deep_root, D)


def _print_handover_note(deep_root: Path, D: int):
    print(f"""
{'='*60}
HANDOVER NOTE FOR SGCN PERSON
{'='*60}
One set of deep features works for ALL label levels (L1, L2, L3).

    import numpy as np, scipy.sparse as sp
    from pathlib import Path

    root = Path("{deep_root.parent.parent}")
    adj  = sp.load_npz(root / "master_grid_adj.npz")  # unchanged

    # Load deep features  (shape: 4096 × {D})
    vehicle_id = "train_00042_003"
    node_feats = np.load(
        root / "deep_features" / "multilevel" / "train" / f"{{vehicle_id}}.npy"
    )  # shape ({N_NODES}, {D})

    # Use the SAME node_feats for L1, L2, and L3 SGCN runs.
    # Only the label file changes between levels.
{'='*60}
""")


# ===========================================================================
# 5.  ARCHITECTURE SUMMARY
# ===========================================================================

def summarise(args):
    dummy_num_classes = {"L1": 6, "L2": 57, "L3": 217}
    model = MultiLevelDeepFeatureRefiner(dummy_num_classes, D=args.D)

    total = trainable = 0
    for p in model.parameters():
        total     += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"\n{'='*60}")
    print(f"MultiLevelDeepFeatureRefiner  in_ch={IN_CHANNELS}  D={args.D}")
    print(f"{'='*60}")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")

    x = torch.zeros(1, IN_CHANNELS, TARGET_H, TARGET_W)
    with torch.no_grad():
        f1   = model.inception1(x)
        f2   = model.inception2(f1)
        proj = model.proj(f2)
        lL1  = model.head_L1(f2)
        lL2  = model.head_L2(f2)
        lL3  = model.head_L3(f2)

    print(f"\n  Shape trace (batch_size=1):")
    print(f"    Input            : {tuple(x.shape)}")
    print(f"    After Block 1    : {tuple(f1.shape)}")
    print(f"    After Block 2    : {tuple(f2.shape)}")
    print(f"    After 1×1 Proj   : {tuple(proj.shape)}  ← SGCN node features")
    print(f"    Logits L1        : {tuple(lL1.shape)}")
    print(f"    Logits L2        : {tuple(lL2.shape)}")
    print(f"    Logits L3        : {tuple(lL3.shape)}")

    B, D, H, W = proj.shape
    flat = proj.permute(0, 2, 3, 1).reshape(B, H*W, D)
    print(f"\n  SGCN-ready tensor  : {tuple(flat.shape)}  (N_nodes={H*W}, D={D})")
    print(f"  Spatial preserved  : H={TARGET_H}, W={TARGET_W}  ✓")
    print(f"{'='*60}\n")


# ===========================================================================
# 6.  CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="World 2 — Multi-Level Deep Feature Refiner (L1+L2+L3 jointly)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train")
    p_train.add_argument("--handover_dir", required=True)
    p_train.add_argument("--output_dir",   required=True)
    p_train.add_argument("--epochs",     type=int,   default=30)
    p_train.add_argument("--batch_size", type=int,   default=64)
    p_train.add_argument("--lr",         type=float, default=1e-3)
    p_train.add_argument("--D",          type=int,   default=64,
                         help="Output feature dim per pixel.")
    p_train.add_argument("--resume", type=str, default=None, 
                         help="Path to best_multilevel.pt to resume from")
    # ---- extract ----
    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--handover_dir", required=True)
    p_ext.add_argument("--output_dir",   required=True)
    p_ext.add_argument("--checkpoint",   required=True)
    p_ext.add_argument("--batch_size", type=int, default=64)
    p_ext.add_argument("--D",          type=int, default=64)

    # ---- summarise ----
    p_sum = sub.add_parser("summarise")
    p_sum.add_argument("--D", type=int, default=64)

    args = parser.parse_args()

    if   args.command == "train":     train(args)
    elif args.command == "extract":   extract(args)
    elif args.command == "summarise": summarise(args)


if __name__ == "__main__":
    main()